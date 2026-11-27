"""KolektorSDD: samples, pixel masks, the official folds and leakage-safe folds.

Dataset facts this module relies on (each one is checked, not assumed):

* 50 parts ``kos01`` ... ``kos50``; each holds ``PartN.jpg`` images and a
  ``PartN_label.bmp`` mask of the same size for every image.
* Masks: an all-zero grayscale BMP for a normal image, a palette BMP for a
  defective one. A pixel is defective when its value is above zero, which gives
  the same answer whether a reader returns palette indices or grey levels.
* The official 3-fold split (``splits/split.pyb``, a Python 2 pickle holding
  ``[train_folds, test_folds, all_parts]``) splits by part.

The official folds leak: ``kos26`` is a byte-identical copy of ``kos01`` and
``kos34`` of ``kos27`` (every image, including the defective one), and the
folds put each pair on both sides of a train/test boundary. ``part_relations``
finds such relations from the data (identical bytes, identical pixels, and any
near-duplicate pairs supplied by the detector), and ``corrected_folds`` removes
them: whole-part copies are dropped (``drop_copies``, keeping the lowest part
number) or kept on one side (``keep_together``); parts related in any other way
are always kept together.

The split file is never given to ``pickle``: ``decode_split_pickle`` interprets
the handful of data-only opcodes the file uses (lists, tuples, strings, memo)
and raises on anything else, so a tampered file cannot run code.
"""

from __future__ import annotations

import hashlib
import pickletools
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PART_RE = re.compile(r"kos\d\d")
IMAGE_RE = re.compile(r"Part(\d+)\.jpg")
MASK_RE = re.compile(r"Part(\d+)_label\.bmp")


class UnsafePickle(ValueError):
    """The split file contains an opcode outside the data-only subset."""


class DatasetError(ValueError):
    """The files on disk do not match the documented dataset layout."""


# --------------------------------------------------------------------------
# Official split file: a data-only pickle interpreter
# --------------------------------------------------------------------------
_MARK = object()
_STRING_OPS = {"STRING", "BINSTRING", "SHORT_BINSTRING", "UNICODE", "BINUNICODE",
               "SHORT_BINUNICODE", "BINUNICODE8"}
_PUT_OPS = {"PUT", "BINPUT", "LONG_BINPUT"}
_GET_OPS = {"GET", "BINGET", "LONG_BINGET"}


def decode_split_pickle(data: bytes):
    """Evaluate a pickle that may contain only lists, tuples and strings.

    ``pickletools.genops`` only parses (it never imports or calls anything);
    this function then applies the opcodes itself. Any opcode that could build
    another kind of object (GLOBAL, REDUCE, BUILD, INST, OBJ, ...) raises
    :class:`UnsafePickle`.
    """
    stack: list = []
    memo: Dict[int, object] = {}

    def pop_mark() -> list:
        for i in range(len(stack) - 1, -1, -1):
            if stack[i] is _MARK:
                items = stack[i + 1:]
                del stack[i:]
                return items
        raise UnsafePickle("MARK not found")

    try:
        ops = list(pickletools.genops(data))
    except ValueError as e:   # truncated or garbled stream
        raise UnsafePickle(f"malformed pickle: {e}") from e
    for op, arg, _pos in ops:
        name = op.name
        if name in ("PROTO", "FRAME"):
            continue
        if name == "MARK":
            stack.append(_MARK)
        elif name in _STRING_OPS:
            stack.append(arg if isinstance(arg, str) else bytes(arg).decode("latin-1"))
        elif name == "EMPTY_LIST":
            stack.append([])
        elif name == "LIST":
            stack.append(pop_mark())
        elif name == "EMPTY_TUPLE":
            stack.append(())
        elif name == "TUPLE":
            stack.append(tuple(pop_mark()))
        elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
            n = int(name[-1])
            items = stack[-n:]
            del stack[-n:]
            stack.append(tuple(items))
        elif name == "APPEND":
            v = stack.pop()
            if not isinstance(stack[-1], list):
                raise UnsafePickle("APPEND to a non-list")
            stack[-1].append(v)
        elif name == "APPENDS":
            items = pop_mark()
            if not isinstance(stack[-1], list):
                raise UnsafePickle("APPENDS to a non-list")
            stack[-1].extend(items)
        elif name in _PUT_OPS:
            memo[int(arg)] = stack[-1]
        elif name == "MEMOIZE":
            memo[len(memo)] = stack[-1]
        elif name in _GET_OPS:
            if int(arg) not in memo:
                raise UnsafePickle(f"memo key {arg} not defined")
            stack.append(memo[int(arg)])
        elif name == "STOP":
            if len(stack) != 1:
                raise UnsafePickle("malformed pickle: stack not empty at STOP")
            return stack[0]
        else:
            raise UnsafePickle(f"opcode {name} is not allowed in a split file")
    raise UnsafePickle("no STOP opcode")  # pragma: no cover (genops already requires STOP)


@dataclass
class OfficialFolds:
    train: List[List[str]]
    test: List[List[str]]
    parts: List[str]


def _check_part_list(xs, what: str) -> List[str]:
    if not isinstance(xs, (list, tuple)) or not all(isinstance(x, str) and PART_RE.fullmatch(x) for x in xs):
        raise DatasetError(f"{what} is not a list of part names")
    if len(set(xs)) != len(xs):
        raise DatasetError(f"{what} lists a part twice")
    return list(xs)


def load_official_folds(path: str | Path) -> OfficialFolds:
    """Read ``split.pyb`` safely and check its structure completely."""
    obj = decode_split_pickle(Path(path).read_bytes())
    if not isinstance(obj, (list, tuple)) or len(obj) != 3:
        raise DatasetError("split file must hold [train_folds, test_folds, all_parts]")
    train_f, test_f, allp = obj
    allp = _check_part_list(allp, "all_parts")
    if not isinstance(train_f, (list, tuple)) or not isinstance(test_f, (list, tuple)) \
            or len(train_f) != len(test_f) or not train_f:
        raise DatasetError("train and test folds must be lists of equal length")
    train = [_check_part_list(t, f"train fold {k}") for k, t in enumerate(train_f)]
    test = [_check_part_list(t, f"test fold {k}") for k, t in enumerate(test_f)]
    universe = set(allp)
    seen: set = set()
    for k, (tr, te) in enumerate(zip(train, test)):
        if set(tr) & set(te):
            raise DatasetError(f"fold {k}: a part is in both train and test")
        if set(tr) | set(te) != universe:
            raise DatasetError(f"fold {k}: train and test do not cover all parts")
        if seen & set(te):
            raise DatasetError(f"fold {k}: test folds overlap")
        seen |= set(te)
    if seen != universe:
        raise DatasetError("test folds do not cover all parts")
    return OfficialFolds(train=train, test=test, parts=allp)


# --------------------------------------------------------------------------
# Samples and masks
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class KolektorSample:
    part: str
    index: int
    image: str
    mask: str
    defective: bool
    defect_px: int
    height: int
    width: int

    @property
    def id(self) -> str:
        return f"{self.part}/Part{self.index}"


def read_mask(path: str | Path) -> np.ndarray:
    """Boolean defect mask: True where the stored value is above zero."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise DatasetError(f"cannot read mask {path}")
    return m > 0


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise DatasetError(f"cannot read image {path}")
    return img


def load_kolektorsdd(root: str | Path) -> List[KolektorSample]:
    """Every image with its mask; any stray, missing or unpaired file raises."""
    root = Path(root)
    parts = sorted(p for p in root.iterdir() if p.is_dir() and PART_RE.fullmatch(p.name))
    others = sorted(p.name for p in root.iterdir() if p.is_dir() and not PART_RE.fullmatch(p.name)
                    and p.name != "splits")
    if not parts:
        raise DatasetError(f"no kosNN folders in {root}")
    if others:
        raise DatasetError(f"unexpected folders in {root}: {others}")
    samples: List[KolektorSample] = []
    for pdir in parts:
        images, masks = {}, {}
        for f in pdir.iterdir():
            mi, mm = IMAGE_RE.fullmatch(f.name), MASK_RE.fullmatch(f.name)
            if mi:
                images[int(mi.group(1))] = f
            elif mm:
                masks[int(mm.group(1))] = f
            else:
                raise DatasetError(f"unexpected file {f}")
        if set(images) != set(masks):
            raise DatasetError(f"{pdir.name}: images {sorted(images)} and masks {sorted(masks)} differ")
        for idx in sorted(images):
            img = read_image(images[idx])
            m = read_mask(masks[idx])
            if img.shape != m.shape:
                raise DatasetError(f"{images[idx]}: image {img.shape} and mask {m.shape} differ in size")
            n = int(m.sum())
            samples.append(KolektorSample(pdir.name, idx, str(images[idx]), str(masks[idx]),
                                          n > 0, n, img.shape[0], img.shape[1]))
    return samples


# --------------------------------------------------------------------------
# Relations between parts and corrected folds
# --------------------------------------------------------------------------
class _UF:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)

    def groups(self) -> List[List[str]]:
        out: Dict[str, List[str]] = {}
        for x in self.p:
            out.setdefault(self.find(x), []).append(x)
        return sorted((sorted(g) for g in out.values() if len(g) > 1), key=lambda g: g[0])


def _file_sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pixel_sha(path: str) -> str:
    img = read_image(path)
    return hashlib.sha256(str(img.shape).encode() + img.tobytes()).hexdigest()


@dataclass
class PartRelation:
    parts: List[str]
    kind: str                    # copy (every image identical) | related (shares some content)
    evidence: List[str]          # identical-file | identical-pixels | near-duplicate


def part_relations(samples: Sequence[KolektorSample],
                   near_pairs: Sequence[Tuple[str, str]] = ()) -> List[PartRelation]:
    """Groups of parts that share image content.

    Parts are linked when any image has identical bytes or identical decoded
    pixels to an image of another part, or when the near-duplicate detector
    paired two of their images (``near_pairs``: sample ids such as
    ``kos01/Part5``). A group is a ``copy`` when all its parts hold the same
    multiset of images (pixel-identical); otherwise ``related``.
    """
    by_id = {s.id: s for s in samples}
    parts = sorted({s.part for s in samples})
    uf = _UF(parts)
    evidence: Dict[frozenset, set] = {}

    def link(a: str, b: str, why: str):
        if a != b:
            uf.union(a, b)
            evidence.setdefault(frozenset((a, b)), set()).add(why)

    pix: Dict[str, str] = {}
    for kind, fn in (("identical-file", _file_sha), ("identical-pixels", _pixel_sha)):
        seen: Dict[str, str] = {}
        for s in samples:
            h = fn(s.image)
            if kind == "identical-pixels":
                pix[s.id] = h
            if h in seen:
                link(seen[h], s.part, kind)
            else:
                seen[h] = s.part
    for a, b in near_pairs:
        if a not in by_id or b not in by_id:
            raise DatasetError(f"near-duplicate pair names unknown samples: {a}, {b}")
        link(by_id[a].part, by_id[b].part, "near-duplicate")

    content: Dict[str, Tuple[str, ...]] = {}
    for p in parts:
        content[p] = tuple(sorted(pix[s.id] for s in samples if s.part == p))
    out = []
    for g in uf.groups():
        ev = sorted(set().union(*(v for k, v in evidence.items() if k <= set(g))))
        kind = "copy" if len({content[p] for p in g}) == 1 else "related"
        out.append(PartRelation(parts=g, kind=kind, evidence=ev))
    return out


@dataclass
class Folds:
    train: List[List[str]]
    val: List[List[str]]
    test: List[List[str]]
    dropped: List[str]
    policy: str


def _fold_of(test: List[List[str]], part: str) -> int:
    for k, t in enumerate(test):
        if part in t:
            return k
    raise DatasetError(f"{part} is in no test fold")


def corrected_folds(official: OfficialFolds, relations: Sequence[PartRelation],
                    policy: str = "drop_copies", val_frac: float = 0.2,
                    seed: int = 42) -> Folds:
    """Official folds with every cross-split relation removed, plus validation parts.

    * ``drop_copies``: of each ``copy`` group only the lowest part number stays.
    * ``keep_together`` (and every ``related`` group under either policy): the
      whole group moves to the test fold of its lowest-numbered part.

    Validation parts are carved from each fold's training parts (``val_frac``
    of them, at least one), drawing whole groups so a group is never split
    between train and validation. The draw depends only on ``seed`` and the
    fold index.
    """
    if policy not in ("drop_copies", "keep_together"):
        raise ValueError(f"unknown policy {policy}")
    dropped: set = set()
    units: List[List[str]] = []
    for r in relations:
        keep = sorted(r.parts)
        if r.kind == "copy" and policy == "drop_copies":
            dropped |= set(keep[1:])
            keep = keep[:1]
        if len(keep) > 1:
            units.append(keep)
    test = [sorted(set(t) - dropped) for t in official.test]
    for unit in units:
        home = _fold_of(official.test, unit[0])
        for k in range(len(test)):
            test[k] = sorted((set(test[k]) - set(unit)) | (set(unit) if k == home else set()))
    kept = sorted(set(official.parts) - dropped)
    unit_of = {p: tuple(u) for u in units for p in u}
    train, val = [], []
    for k in range(len(test)):
        pool = sorted(set(kept) - set(test[k]))
        atoms = sorted({unit_of.get(p, (p,)) for p in pool})
        rng = np.random.default_rng([seed, k])
        order = [atoms[i] for i in rng.permutation(len(atoms))]
        n_val = max(1, int(round(val_frac * len(pool))))
        v: List[str] = []
        for a in order:
            if len(v) >= n_val:
                break
            v.extend(a)
        val.append(sorted(v))
        train.append(sorted(set(pool) - set(v)))
    folds = Folds(train=train, val=val, test=test, dropped=sorted(dropped), policy=policy)
    check_folds(folds, kept, relations)
    return folds


def check_folds(folds: Folds, kept: Sequence[str], relations: Sequence[PartRelation]) -> None:
    """Raise unless every fold partitions the kept parts and no relation crosses."""
    kept_set = set(kept)
    seen_test: set = set()
    for k, (tr, va, te) in enumerate(zip(folds.train, folds.val, folds.test)):
        sides = [set(tr), set(va), set(te)]
        if sum(len(s) for s in sides) != len(set().union(*sides)):
            raise DatasetError(f"fold {k}: a part is on two sides")
        if set().union(*sides) != kept_set:
            raise DatasetError(f"fold {k}: sides do not cover the kept parts")
        if set(folds.dropped) & set().union(*sides):
            raise DatasetError(f"fold {k}: a dropped part is still used")
        if seen_test & set(te):
            raise DatasetError(f"fold {k}: test folds overlap")
        seen_test |= set(te)
        for r in relations:
            live = [p for p in r.parts if p in kept_set]
            if sum(1 for s in sides if s & set(live)) > 1:
                raise DatasetError(f"fold {k}: related parts {live} are split across sides")
    if seen_test != kept_set:
        raise DatasetError("test folds do not cover the kept parts")


def crossing_relations(train: Sequence[str], test: Sequence[str],
                       relations: Sequence[PartRelation]) -> List[List[str]]:
    """Relations with parts on both sides of a train/test boundary."""
    tr, te = set(train), set(test)
    return [r.parts for r in relations if set(r.parts) & tr and set(r.parts) & te]


def fold_summary(samples: Sequence[KolektorSample], parts: Sequence[str]) -> Dict:
    ps = set(parts)
    sel = [s for s in samples if s.part in ps]
    return {"parts": len(ps), "images": len(sel), "defective": sum(s.defective for s in sel)}


def split_report(samples: Sequence[KolektorSample], official: OfficialFolds,
                 relations: Sequence[PartRelation], folds: Folds) -> Dict:
    """Official versus corrected folds, with every leaking relation named."""
    rep = {"relations": [asdict(r) for r in relations], "policy": folds.policy,
           "dropped_parts": folds.dropped, "official": [], "corrected": []}
    for k in range(len(official.test)):
        rep["official"].append({
            "fold": k, "train": fold_summary(samples, official.train[k]),
            "test": fold_summary(samples, official.test[k]),
            "crossing": crossing_relations(official.train[k], official.test[k], relations)})
        rep["corrected"].append({
            "fold": k, "train": fold_summary(samples, folds.train[k]),
            "val": fold_summary(samples, folds.val[k]),
            "test": fold_summary(samples, folds.test[k]),
            "crossing": crossing_relations(folds.train[k] + folds.val[k], folds.test[k], relations),
            "val_parts": folds.val[k], "test_parts": folds.test[k]})
    return rep


def mask_iou(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    union = int((a | b).sum())
    return None if union == 0 else int((a & b).sum()) / union
