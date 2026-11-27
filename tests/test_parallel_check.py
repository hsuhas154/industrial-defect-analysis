"""Parallel DAGM check: same results as the serial path, ledger rules, status files."""

import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from idefect.config import ContentScreenConfig, DuplicateConfig
from idefect.data import duplicates as dup
from idefect.data.phase4_dagm_worker import _first_difference, ram_estimates
from idefect.data.phase4_leakage import content_check
from idefect.data.phase4_runtime import Ledger, Status, pid_alive
from idefect.data.similarity_screen import gpu_bytes, similarity_matrix


def _tex(seed, shape=(96, 96), sigma=1.0):
    rng = np.random.default_rng(seed)
    x = cv2.GaussianBlur((rng.random(shape) * 255).astype(np.uint8), (0, 0), sigma)
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)


def _image_set(n=14):
    """Overlapping crops under flips and rotations, mixed with unrelated textures."""
    big = _tex(1, (260, 260))
    out = []
    for k in range(n):
        if k % 3 == 2:
            out.append(_tex(100 + k))
        else:
            y, x = (k * 13) % 150, (k * 29) % 150
            crop = big[y:y + 96, x:x + 96]
            out.append(np.ascontiguousarray(dup.dihedral(crop, k % 8)))
    return out


def test_parallel_e1_equals_serial():
    cfg = DuplicateConfig()
    imgs = _image_set()
    idx = list(range(len(imgs)))
    seen = []
    serial = dup.exact_overlap_candidates(imgs, idx, cfg, workers=1, progress=lambda d, t: seen.append((d, t)),
                                          chunk=3)
    par = dup.exact_overlap_candidates(imgs, idx, cfg, workers=3, chunk=2)
    assert len(serial) > 5
    assert par == serial and list(par) == list(serial)          # same content, same order
    assert seen[-1] == (len(imgs), len(imgs))
    assert dup.exact_overlap_pairs(imgs, cfg, workers=3) == dup.exact_overlap_pairs(imgs, cfg)


def test_similarity_progress_does_not_change_scores():
    rng = np.random.default_rng(0)
    ts = [rng.normal(size=(10, 10)).astype(np.float32) for _ in range(5)]
    calls = []
    a = similarity_matrix(ts, (0, 2, 4, 6), 0.5, "cpu")
    b = similarity_matrix(ts, (0, 2, 4, 6), 0.5, "cpu", progress=lambda d, t: calls.append((d, t)))
    for k in ("score", "transform", "dy", "dx"):
        assert np.array_equal(a[k], b[k])
    assert calls == [(i, 5) for i in range(1, 6)]


def test_gpu_bytes():
    # 1,150 thumbnails of 64 x 64, all 8 transforms: canvas 128 x 128
    spectra = 3 * 1150 * 8 * 128 * 65 * 8
    work = 512 * 128 * 128 * 4 * 24
    assert gpu_bytes([(64, 64)] * 1150, tuple(range(8))) == spectra + work
    # a rot90 of a tall image needs a wider canvas
    assert gpu_bytes([(100, 40)], (1,)) > gpu_bytes([(100, 40)], (0,))


def test_content_check_hooks_do_not_change_result():
    imgs = _image_set(12)
    ids = [f"img{i}" for i in range(len(imgs))]
    sc = ContentScreenConfig(exact_stride=3, coarse_factor=4.0, fine_factor=2.0, top_k=3, planted_sources=3)
    plain = content_check(imgs, ids, ids, sc, DuplicateConfig(), (0, 2, 4, 6), "t", cache_dir=None, seed=1)
    stages, gates = [], []

    from contextlib import contextmanager

    @contextmanager
    def gate(stage, **info):
        gates.append((stage, sorted(info)))
        yield

    hooked = content_check(imgs, ids, ids, sc, DuplicateConfig(), (0, 2, 4, 6), "t", cache_dir=None, seed=1,
                           progress=lambda s, d, t: stages.append((s, d, t)), gate=gate, e1_workers=3)
    assert json.dumps(plain) == json.dumps(hooked)
    assert gates == [("e1", []), ("coarse", ["thumbs", "transforms"])]
    last = {}
    for s, d, t in stages:
        last[s] = (d, t)
    n = len(imgs)
    assert last["coarse"] == (n * (n - 1) // 2,) * 2
    assert last["e1"] == (n, n) and last["planted"] == (3, 3)
    assert last["fine"][0] == last["fine"][1] == plain["n_candidates"]


def test_cache_write_is_atomic(tmp_path):
    imgs = _image_set(6)
    ids = [f"img{i}" for i in range(len(imgs))]
    sc = ContentScreenConfig(exact_stride=3, coarse_factor=4.0, top_k=2, planted_sources=2)
    out = content_check(imgs, ids, ids, sc, DuplicateConfig(), (0,), "t", cache_dir=str(tmp_path))
    files = sorted(p.name for p in tmp_path.iterdir())
    assert len(files) == 1 and files[0].endswith(".json")      # no partial file left behind
    assert json.loads((tmp_path / files[0]).read_text()) == json.loads(json.dumps(out))


@pytest.fixture
def live_pids():
    procs = [subprocess.Popen(["sleep", "60"]) for _ in range(3)]
    yield [p.pid for p in procs]
    for p in procs:
        p.kill()
        p.wait()


def test_ledger_budget_and_order(tmp_path, live_pids):
    a, b, c = live_pids
    led = Ledger.create(tmp_path, {"ram": 10})
    assert led.try_set("ram", 6, "a", pid=a)
    assert not led.try_set("ram", 5, "b", pid=b, since=1.0)      # 6 + 5 > 10: waits
    assert not led.try_set("ram", 4, "c", pid=c, since=2.0)      # would fit, but b has waited longer
    assert led.try_set("ram", 1, "a", pid=a)                     # a decrease always succeeds
    assert led.try_set("ram", 5, "b", pid=b, since=1.0)          # 1 + 5 fits
    assert led.try_set("ram", 4, "c", pid=c, since=2.0)          # 1 + 5 + 4 fits, nobody earlier
    assert not led.try_set("ram", 5, "c", pid=c, since=3.0)      # 1 + 5 + 5 > 10
    assert led.snapshot()["held"][f"{c}/ram"]["bytes"] == 4      # a refused increase keeps the old value


def test_ledger_no_deadlock_when_all_holders_wait(tmp_path, live_pids):
    a, b, _ = live_pids
    led = Ledger.create(tmp_path, {"ram": 10})
    assert led.try_set("ram", 6, "a", pid=a)
    assert led.try_set("ram", 3, "b", pid=b)
    assert not led.try_set("ram", 9, "b more", pid=b, since=1.0)  # a holds 6 and is running: wait
    # a now asks for more as well; its only other holder (b) is waiting, so nothing could move
    assert led.try_set("ram", 9, "a more", pid=a, since=0.5)


def test_ledger_drops_dead_and_zombie_holders(tmp_path):
    led = Ledger.create(tmp_path, {"ram": 10})
    done = subprocess.Popen(["true"])
    done.wait()
    zombie = subprocess.Popen(["sleep", "60"])
    zombie.kill()
    import time

    time.sleep(0.3)                     # killed but not reaped yet: a zombie
    assert not pid_alive(done.pid) and not pid_alive(zombie.pid) and pid_alive(os.getpid())
    led.try_set("ram", 9, "dead", pid=done.pid)
    led.try_set("ram", 9, "zombie", pid=zombie.pid)
    assert led.snapshot()["held"] == {}
    assert led.try_set("ram", 9, "me")
    zombie.wait()


def test_status_file_and_watchdog_cancel(tmp_path):
    st = Status(tmp_path / "s.json", stall=60, cls=3, mode="run")
    st.stage("e1", 10)
    st.progress(4)
    st.finish("done", summary={"x": 1})
    d = json.loads((tmp_path / "s.json").read_text())
    assert d["state"] == "done" and d["summary"] == {"x": 1} and "e1" in d["stages"]


def test_first_difference_and_estimates():
    a = {"x": [1, 2, {"y": 0.5}], "z": "s"}
    assert _first_difference(a, json.loads(json.dumps(a))) == []
    b = {"x": [1, 2, {"y": 0.6}], "z": "s"}
    assert _first_difference(a, b) == ["/x[2]/y: 0.5 vs 0.6"]
    e = ram_estimates(2300, 2300 * 512 * 512, 12)
    assert e["base"] < e["gpu"] < e["e1"]
