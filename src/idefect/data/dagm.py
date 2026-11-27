"""DAGM 2007: samples, weak ellipse labels and a leakage-safe validation split.

Layout per class (checked strictly; Windows ``Thumbs.db`` caches are ignored):

    ClassN/{Train,Test}/NNNN.PNG
    ClassN/{Train,Test}/Label/Labels.txt
    ClassN/{Train,Test}/Label/NNNN_label.PNG     (defective images only)

``Labels.txt`` starts with a header line ``1``; every other line is
``id <tab> flag <tab> NNNN.PNG <tab> 0 <tab> NNNN_label.PNG | 0``. The parser
accepts exactly that and cross-checks the listing against the folder in both
directions, so an image without a label line, a label line without an image,
or a mask without a defect flag all stop the load.

The official Train/Test split is kept as the test protocol. Validation images
come only from the Train side, stratified by defect flag, per class; any
near-duplicate groups supplied by the caller are kept whole (never split
between train and validation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from idefect.data.kolektor import DatasetError

SPLITS = ("Train", "Test")
_NAME_RE = re.compile(r"(\d{4})\.PNG")


@dataclass(frozen=True)
class DAGMSample:
    cls: int
    split: str            # Train | Test (official)
    name: str             # NNNN.PNG
    image: str
    label: Optional[str]  # mask path for defective images
    defective: bool

    @property
    def id(self) -> str:
        return f"Class{self.cls}/{self.split}/{self.name}"


@dataclass(frozen=True)
class LabelLine:
    id: str
    defective: bool
    name: str
    label: Optional[str]


def parse_labels(path: str | Path) -> List[LabelLine]:
    """Parse one ``Labels.txt`` strictly; any deviation raises."""
    lines = Path(path).read_text(encoding="ascii").splitlines()
    if not lines or lines[0].strip() != "1":
        raise DatasetError(f"{path}: first line must be the header '1'")
    out, names = [], set()
    for no, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        f = line.rstrip("\r\n").split("\t")
        while f and f[-1].strip() == "":
            f.pop()
        if len(f) != 5:
            raise DatasetError(f"{path}:{no}: expected 5 fields, got {f}")
        rid, flag, name, zero, label = (x.strip() for x in f)
        m = _NAME_RE.fullmatch(name)
        if not m or rid != m.group(1) or zero != "0" or flag not in ("0", "1"):
            raise DatasetError(f"{path}:{no}: malformed line {line!r}")
        if flag == "1" and label != f"{rid}_label.PNG":
            raise DatasetError(f"{path}:{no}: defective image without its label file")
        if flag == "0" and label != "0":
            raise DatasetError(f"{path}:{no}: normal image with a label file")
        if name in names:
            raise DatasetError(f"{path}:{no}: {name} listed twice")
        names.add(name)
        out.append(LabelLine(rid, flag == "1", name, label if flag == "1" else None))
    return out


def load_dagm(root: str | Path, classes: Sequence[int] = range(1, 11)) -> List[DAGMSample]:
    root = Path(root)
    samples: List[DAGMSample] = []
    for c in classes:
        for split in SPLITS:
            d = root / f"Class{c}" / split
            labels = parse_labels(d / "Label" / "Labels.txt")
            on_disk = {f.name for f in d.iterdir() if f.is_file() and f.name.lower() != "thumbs.db"}
            listed = {r.name for r in labels}
            if on_disk != listed:
                raise DatasetError(f"{d}: images and Labels.txt differ "
                                   f"(only on disk {sorted(on_disk - listed)[:5]}, "
                                   f"only listed {sorted(listed - on_disk)[:5]})")
            masks_disk = {f.name for f in (d / "Label").iterdir()
                          if f.is_file() and f.name.lower() != "thumbs.db"} - {"Labels.txt"}
            masks_listed = {r.label for r in labels if r.label}
            if masks_disk != masks_listed:
                raise DatasetError(f"{d}/Label: mask files and defect flags differ")
            others = sorted(p.name for p in d.iterdir() if p.is_dir() and p.name != "Label")
            if others:
                raise DatasetError(f"{d}: unexpected folders {others}")
            for r in labels:
                samples.append(DAGMSample(c, split, r.name, str(d / r.name),
                                          str(d / "Label" / r.label) if r.label else None, r.defective))
    return samples


def read_label(path: Optional[str], shape=(512, 512)) -> np.ndarray:
    """Boolean weak label (ellipse); all False for a normal image."""
    if path is None:
        return np.zeros(shape, bool)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise DatasetError(f"cannot read label {path}")
    return m > 0


def validation_split(samples: Sequence[DAGMSample], val_frac: float = 0.2, seed: int = 42,
                     groups: Sequence[Sequence[str]] = ()) -> Dict[str, str]:
    """{sample id: train | val | test}.

    Official Test stays test. For each class and each defect flag, ``val_frac``
    of the Train images (rounded) go to validation. ``groups`` lists sample
    ids that must stay together; a group is an atom of the draw, belongs to the
    defective stratum if any member is defective, and may not straddle the
    official Train/Test boundary (that would be leakage the official split
    already contains, which the caller must resolve first).
    """
    by_id = {s.id: s for s in samples}
    group_of: Dict[str, tuple] = {}
    for g in groups:
        g = tuple(sorted(g))
        if len({by_id[i].split for i in g}) > 1 or len({by_id[i].cls for i in g}) > 1:
            raise DatasetError(f"group {g} crosses classes or the official Train/Test split")
        for i in g:
            if i in group_of:
                raise DatasetError(f"{i} is in two groups")
            group_of[i] = g
    assign = {s.id: "test" for s in samples if s.split == "Test"}
    for c in sorted({s.cls for s in samples}):
        train = [s for s in samples if s.cls == c and s.split == "Train"]
        atoms = sorted({group_of.get(s.id, (s.id,)) for s in train})
        for flag in (False, True):
            stratum = [a for a in atoms if any(by_id[i].defective for i in a) == flag]
            n_target = int(round(val_frac * sum(len(a) for a in stratum)))
            rng = np.random.default_rng([seed, c, int(flag)])
            n = 0
            for j in rng.permutation(len(stratum)):
                part = "val" if n < n_target else "train"
                if part == "val":
                    n += len(stratum[j])
                for i in stratum[j]:
                    assign[i] = part
    return assign
