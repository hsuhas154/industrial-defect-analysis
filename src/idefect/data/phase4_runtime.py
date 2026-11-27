"""Process plumbing for the parallel DAGM check: ledger, heartbeats, system readout.

Nothing here changes a result. It decides when work may start (so memory
cannot overflow), notices when work stops moving (so a hang costs minutes,
not hours) and shows what is going on.

* ``Ledger``: RAM and GPU memory reservations shared by all worker processes
  through one JSON file under an ``flock``. A worker asks before a memory-heavy
  step and waits while the step does not fit the budget. Entries of processes
  that died are dropped automatically, so a crash can never leave memory
  reserved. Requests are served first come, first served, so a large step
  cannot be starved by small ones.
* ``Status``: each worker rewrites a small JSON status file (stage, progress,
  memory) at most once per second. Every write re-arms a ``faulthandler``
  watchdog: if no progress is reported for ``stall`` seconds, the Python stack
  of every thread is written to the worker log and the worker exits, so the
  orchestrator can retry it in a fresh process.
* ``die_with_parent``: a process that loses its parent is killed by the
  kernel (``PR_SET_PDEATHSIG``), so no orphan keeps running and holding memory.
* System readout from ``/proc`` and ``nvidia-smi`` (CPU, RAM, swap, GPU, per
  process-group memory and CPU).
"""

from __future__ import annotations

import fcntl
import faulthandler
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional

GB = 1 << 30


# --------------------------------------------------------------------------
# Processes
# --------------------------------------------------------------------------
def die_with_parent(sig: int = signal.SIGKILL) -> None:
    """Ask the kernel to send ``sig`` to this process when its parent exits (Linux)."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, sig)   # PR_SET_PDEATHSIG
    except OSError:
        pass


def pid_alive(pid: int) -> bool:
    """True if ``pid`` runs and is not a zombie."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            state = fh.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return state not in ("Z", "X")


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def read_json(path: Path) -> Optional[Dict]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------
# Shared memory ledger
# --------------------------------------------------------------------------
class Ledger:
    """RAM and GPU reservations shared between processes (see the module docstring)."""

    def __init__(self, run_dir: str | Path):
        self.path = Path(run_dir) / "ledger.json"
        self.lock = Path(run_dir) / "ledger.lock"

    @staticmethod
    def create(run_dir: str | Path, budgets: Dict[str, int]) -> "Ledger":
        led = Ledger(run_dir)
        led.lock.touch()
        with led._locked() as st:
            st.clear()
            st.update({"budget": dict(budgets), "held": {}, "waiting": {}})
        return led

    @contextmanager
    def _locked(self):
        with open(self.lock, "a+") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                st = read_json(self.path) or {"budget": {}, "held": {}, "waiting": {}}
                for part in ("held", "waiting"):
                    st[part] = {k: v for k, v in st[part].items() if pid_alive(int(k.split("/")[0]))}
                yield st
                write_json_atomic(self.path, st)
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    def try_set(self, resource: str, nbytes: int, label: str = "", pid: int | None = None,
                since: float | None = None) -> bool:
        """Set this process's reservation of ``resource``; False if an increase must wait.

        A decrease always succeeds. An increase is granted when everything
        held by other processes plus ``nbytes`` fits the budget (or nothing
        else is held, so one step can always run alone) and no other process
        has been waiting longer for the same resource. If every other holder
        is itself waiting, the request is granted too (no deadlock; the
        orchestrator's memory guard still watches the machine).
        """
        pid = os.getpid() if pid is None else pid
        key = f"{pid}/{resource}"
        since = time.time() if since is None else since
        with self._locked() as st:
            cur = st["held"].get(key, {}).get("bytes", 0)
            if nbytes <= cur:
                st["held"][key] = {"bytes": int(nbytes), "label": label}
                st["waiting"].pop(key, None)
                return True
            held = {k: v for k, v in st["held"].items() if k != key and k.endswith(f"/{resource}")}
            others = sum(v["bytes"] for v in held.values())
            earlier = any(v["since"] < since for k, v in st["waiting"].items()
                          if k != key and k.endswith(f"/{resource}"))
            # holders that are themselves waiting cannot release anything: if every
            # other holder waits, nothing would ever move, so the first in line goes
            waiting_pids = {k.split("/")[0] for k in st["waiting"] if k != key}
            stuck = all(k.split("/")[0] in waiting_pids for k in held)
            budget = st["budget"].get(resource, 0)
            if not earlier and (others + nbytes <= budget or others == 0 or stuck):
                st["held"][key] = {"bytes": int(nbytes), "label": label}
                st["waiting"].pop(key, None)
                return True
            st["waiting"][key] = {"bytes": int(nbytes), "label": label, "since": since}
            return False

    def set(self, resource: str, nbytes: int, label: str = "",
            on_wait: Callable[[str], None] | None = None, poll: float = 1.0) -> float:
        """Block until the reservation is granted; returns the seconds waited."""
        t0 = time.time()
        while not self.try_set(resource, nbytes, label, since=t0):
            if on_wait is not None:
                on_wait(f"waiting for {nbytes / GB:.1f} GB of {resource.upper()} ({label})")
            time.sleep(poll)
        return time.time() - t0

    def snapshot(self) -> Dict:
        with self._locked() as st:
            return json.loads(json.dumps(st))


# --------------------------------------------------------------------------
# Worker status and stall watchdog
# --------------------------------------------------------------------------
def rss_bytes(pid: int | str = "self") -> int:
    try:
        with open(f"/proc/{pid}/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return 0


class Status:
    """One worker's status file plus the stall watchdog (see the module docstring)."""

    def __init__(self, path: str | Path, stall: float, **fixed):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stall = float(stall)
        now = time.time()
        self.d = {**fixed, "pid": os.getpid(), "state": "running", "stage": "start", "done": 0,
                  "total": 0, "started": now, "stage_started": now, "updated": now, "message": "",
                  "stages": {}, "rss_gb": 0.0, "peak_rss_gb": 0.0}
        self._last = 0.0
        faulthandler.enable(file=sys.stderr, all_threads=True)
        self.write(force=True)

    @property
    def stage_name(self) -> str:
        return self.d["stage"]

    def stage(self, name: str, total: int = 0, message: str = "") -> None:
        now = time.time()
        prev = self.d["stage"]
        self.d["stages"][prev] = round(now - self.d["stage_started"], 1)
        self.d.update(stage=name, done=0, total=int(total), stage_started=now, message=message,
                      rate_from=None)
        print(f"[{time.strftime('%H:%M:%S')}] stage {name}" + (f" ({total:,})" if total else "")
              + (f": {message}" if message else ""), flush=True)
        self.write(force=True)

    def progress(self, done: int, total: int | None = None) -> None:
        if self.d.get("rate_from") is None:
            # the rate (and so the ETA) is measured from the first report of the stage, so
            # set-up work before it (the E1 table, the reference spectra) does not distort it
            self.d["rate_from"] = [time.time(), int(done)]
        self.d["done"] = int(done)
        if total is not None:
            self.d["total"] = int(total)
        self.d["message"] = ""
        self.write()

    def note(self, message: str) -> None:
        self.d["message"] = message
        self.write()

    def finish(self, state: str, **extra) -> None:
        now = time.time()
        self.d["stages"][self.d["stage"]] = round(now - self.d["stage_started"], 1)
        self.d.update(state=state, stage=state, **extra)
        faulthandler.cancel_dump_traceback_later()
        self.write(force=True, arm=False)

    def write(self, force: bool = False, arm: bool = True) -> None:
        now = time.time()
        if not force and now - self._last < 1.0:
            return
        self._last = now
        rss = rss_bytes() / GB
        self.d.update(updated=now, rss_gb=round(rss, 2), peak_rss_gb=round(max(rss, self.d["peak_rss_gb"]), 2))
        write_json_atomic(self.path, self.d)
        if arm:
            # no progress report for `stall` seconds: dump every thread's stack, exit
            faulthandler.dump_traceback_later(self.stall, exit=True, file=sys.stderr)


# --------------------------------------------------------------------------
# System readout
# --------------------------------------------------------------------------
def meminfo() -> Dict[str, int]:
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, v = line.split(":", 1)
            out[k] = int(v.split()[0]) * 1024
    return out


class CpuMeter:
    """Whole-machine CPU use between calls, from /proc/stat."""

    def __init__(self):
        self.prev = self._read()

    @staticmethod
    def _read():
        with open("/proc/stat") as fh:
            v = [int(x) for x in fh.readline().split()[1:]]
        idle = v[3] + v[4]
        return sum(v), idle

    def percent(self) -> float:
        cur = self._read()
        dt, di = cur[0] - self.prev[0], cur[1] - self.prev[1]
        self.prev = cur
        return 100.0 * (1 - di / dt) if dt > 0 else 0.0


def gpu_readout(timeout: float = 8.0) -> Optional[Dict[str, float]]:
    """Utilization, memory, temperature and power from nvidia-smi; None if it does not answer."""
    q = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    vals = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]

    def num(x):
        try:
            return float(x)
        except ValueError:
            return float("nan")

    u, mu, mt, t, p = (num(x) for x in vals[:5])
    return {"util": u, "mem_used_gb": mu / 1024, "mem_total_gb": mt / 1024, "temp": t, "power": p}


def group_pids(pgid: int) -> List[int]:
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as fh:
                rest = fh.read().rsplit(")", 1)[1].split()
            if int(rest[2]) == pgid:
                out.append(int(d))
        except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
            continue
    return out


def group_usage(pgid: int) -> Dict:
    """Proportional memory (PSS: shared pages split between sharers) and CPU ticks per process of a group."""
    pss, ticks, n = 0, {}, 0
    for pid in group_pids(pgid):
        try:
            with open(f"/proc/{pid}/smaps_rollup") as fh:
                for line in fh:
                    if line.startswith("Pss:"):
                        pss += int(line.split()[1]) * 1024
                        break
            with open(f"/proc/{pid}/stat") as fh:
                rest = fh.read().rsplit(")", 1)[1].split()
            ticks[pid] = int(rest[11]) + int(rest[12])
            n += 1
        except (FileNotFoundError, ProcessLookupError, IndexError, ValueError, PermissionError):
            continue
    return {"pss_gb": pss / GB, "ticks": ticks, "procs": n}


def kill_group(pgid: int, grace: float = 8.0) -> None:
    """SIGTERM the whole process group, then SIGKILL whatever is left."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        t0 = time.time()
        while time.time() - t0 < grace:
            if not group_pids(pgid):
                return
            time.sleep(0.3)


def fmt_dur(sec: float | None) -> str:
    if sec is None or sec != sec or sec < 0:
        return "-"
    sec = int(round(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"
