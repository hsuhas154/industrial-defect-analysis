"""Run the DAGM shared-content check for many classes at once, safely and visibly.

What it does:

1. Plans one job per DAGM class that has no cached result yet, plus a
   ``verify`` job for classes that are already cached (by default Classes 1
   and 2): those recompute the class with the new code, write nothing to the
   cache and must reproduce the cached result exactly.
2. Starts up to ``--jobs`` worker processes (``phase4_dagm_worker``), each in
   its own process group. Each splits its E1 step over ``--helpers`` forked
   processes. Big classes (2,300 images) start first.
3. Memory: before a memory-heavy step a worker reserves RAM or GPU memory in a
   shared ledger whose budgets leave ``--ram-reserve-gb`` and
   ``--gpu-reserve-gb`` free. If available RAM still falls below
   ``--ram-emergency-gb`` (or swap use grows by more than 1 GB), the newest
   worker is stopped and queued again, and one fewer job runs from then on.
4. Hangs: a worker that reports no progress for ``--stall`` seconds dumps its
   stack to its log and exits; the job is retried in a fresh process (up to
   ``--retries`` times). Three stalls within 20 minutes stop the whole run,
   because that points at the GPU driver rather than at one job.
5. Shows a status block every ``--every`` seconds (also written to
   ``status.txt`` in the run folder; ``--status`` prints it from another
   terminal) and one line per event.
6. At the end: every cache file that existed before the run must be unchanged
   (SHA-256), every verify job must report ``identical``. If a verify job
   differs, the run stops and the caches it wrote are moved to a quarantine
   folder, so the report cannot use them.

Ctrl+C stops everything cleanly: all worker groups are terminated, finished
classes stay cached, unfinished ones leave nothing behind.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from idefect.data.phase4_dagm_worker import BASE_RAM
from idefect.data.phase4_runtime import (GB, CpuMeter, Ledger, fmt_dur, gpu_readout, group_usage,
                                         kill_group, meminfo, read_json)

REPO = Path(__file__).resolve().parents[3]


@dataclass
class Job:
    cls: int
    mode: str                     # run | verify | smoke
    n_images: int
    attempt: int = 1
    requeues: int = 0
    state: str = "queued"         # queued | running | done | failed
    proc: Optional[subprocess.Popen] = None
    started: float = 0.0
    first_started: float = 0.0
    finished: float = 0.0
    kill_reason: str = ""
    result: Dict = field(default_factory=dict)
    note: str = ""
    last_stage: str = ""
    ticks: Dict = field(default_factory=dict)
    ticks_at: float = 0.0
    cpu_pct: float = 0.0
    pss_gb: float = 0.0

    @property
    def name(self) -> str:
        return {"run": "class", "verify": "verify", "smoke": "smoke"}[self.mode] + str(self.cls)


class Orchestrator:
    def __init__(self, a, cfg):
        self.a, self.cfg = a, cfg
        self.t0 = time.time()
        self.cpu = CpuMeter()
        self.gpu = gpu_readout()
        self.gpu_at = time.time()
        self.stalls: List[float] = []
        self.stop_reason = ""
        self.interrupted = False
        self.last_print = 0.0
        base = REPO / "experiments" / "logs" / ("phase4_dagm_smoke" if a.smoke else "phase4_dagm")
        self.run_dir = base / time.strftime("%Y%m%d-%H%M%S")
        (self.run_dir / "status").mkdir(parents=True, exist_ok=True)
        latest = base / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(self.run_dir.name)
        self.log_path = self.run_dir / "orchestrator.log"

    # ---------------------------------------------------------------- output
    def say(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as fh:
            fh.write(line + "\n")

    def block(self, text: str) -> None:
        print(text, flush=True)
        with open(self.log_path, "a") as fh:
            fh.write(text + "\n")
        (self.run_dir / "status.txt").write_text(text + "\n")

    # -------------------------------------------------------------- planning
    def plan(self) -> List[Job]:
        from idefect.data.archives import read_sha256_list
        from idefect.data.dagm import load_dagm
        from idefect.data.phase4_dagm_worker import CLASS_SIZES, TRANSFORMS
        from idefect.data.phase4_leakage import _cache_file

        dc, dup = self.cfg.dagm, self.cfg.data.duplicates
        if self.a.smoke:
            return [Job(c, "smoke", self.a.smoke) for c in self.a.classes]
        expected = read_sha256_list(dc.manifest)
        root = Path(dc.root)
        runs, verifies = [], []
        self.cache_paths = {}
        for c in sorted(set(self.a.classes) | set(self.a.verify)):
            sub = load_dagm(dc.root, [c])
            dig = [expected[Path(s.image).relative_to(root).as_posix()] for s in sub]
            path = _cache_file(dup.cache_dir, f"dagm_class{c}", dig, dc.screen, dup, TRANSFORMS)
            self.cache_paths[c] = path
            if path.is_file():
                if c in self.a.verify:
                    verifies.append(Job(c, "verify", CLASS_SIZES[c]))
                else:
                    self.say(f"class{c}: already cached ({path.name}), skipped")
            elif c in self.a.classes:
                runs.append(Job(c, "run", CLASS_SIZES[c]))
            else:
                self.say(f"verify{c}: no cached result to compare with, skipped")
        runs.sort(key=lambda j: (-j.n_images, j.cls))
        # big classes first; the verify jobs early so the proof arrives early
        order: List[Job] = []
        for k, j in enumerate(runs):
            order.append(j)
            if k < len(verifies):
                order.append(verifies[k])
        order += verifies[len(runs):]
        return order

    # ------------------------------------------------------------- processes
    def launch(self, job: Job) -> None:
        a = self.a
        cmd = [sys.executable, "-u", "-m", "idefect.data.phase4_dagm_worker", "--cls", str(job.cls),
               "--mode", job.mode, "--run-dir", str(self.run_dir), "--helpers", str(self.helpers),
               "--stall", str(a.stall), "--attempt", str(job.attempt), "--config", a.config]
        if job.mode == "smoke":
            cmd += ["--limit", str(a.smoke)]
        env = os.environ.copy()
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[k] = "1"
        env["PYTHONPATH"] = str(REPO / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        logf = open(self.run_dir / f"{job.name}.log", "a")
        logf.write(f"===== attempt {job.attempt} started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        logf.flush()
        status = self.status_path(job)
        if status.exists():
            status.unlink()
        job.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True, cwd=REPO, env=env)
        logf.close()
        job.state, job.started, job.kill_reason, job.last_stage = "running", time.time(), "", ""
        job.first_started = job.first_started or job.started
        job.ticks, job.ticks_at, job.cpu_pct = {}, 0.0, 0.0
        self.say(f"{job.name}: started (attempt {job.attempt}, pid {job.proc.pid}, "
                 f"{job.n_images:,} images, {self.helpers} E1 helpers)")

    def status_path(self, job: Job) -> Path:
        return self.run_dir / "status" / f"{job.mode}_class{job.cls}.json"

    def log_tail(self, job: Job, n: int = 25) -> str:
        try:
            lines = (self.run_dir / f"{job.name}.log").read_text(errors="replace").splitlines()
        except FileNotFoundError:
            return ""
        return "\n".join("    | " + x for x in lines[-n:])

    def handle_exit(self, job: Job, rc: int) -> List[Job]:
        """Returns the jobs to put back at the front of the queue."""
        st = read_json(self.status_path(job)) or {}
        job.finished = time.time()
        took = fmt_dur(job.finished - job.started)
        if job.kill_reason == "memory":
            job.requeues += 1
            job.state = "queued"
            self.say(f"{job.name}: stopped to free memory after {took}; queued again")
            return [job]
        if rc == 0:
            job.state, job.result = "done", st.get("summary", {})
            r = job.result
            if job.mode == "verify":
                job.note = "identical to the cached result" if r.get("identical") else "DIFFERS"
            else:
                job.note = (f"{r.get('exact')} exact, {r.get('same_or_review')} same or review, "
                            f"max fine distinct {r.get('max_fine_distinct')}, "
                            f"planted min {r.get('planted_fine_min')}")
            self.say(f"{job.name}: DONE in {took}: {job.note}")
            return []
        if rc in (3, 4):
            job.state, job.result = "failed", st.get("summary", {})
            job.note = "recomputed result DIFFERS from the cache" if rc == 3 else "planted copies missed"
            self.say(f"{job.name}: FAILED: {job.note}\n{self.log_tail(job)}")
            if rc == 3:
                self.stop_reason = (f"verify{job.cls} differs from the cached result: "
                                    + "; ".join(job.result.get("differences", [])[:5]))
            return []
        tail = self.log_tail(job)
        stalled = job.kill_reason == "stale" or "Timeout (" in tail
        why = "stalled (no progress)" if stalled else f"exited with code {rc}"
        self.say(f"{job.name}: {why} in stage {st.get('stage', '?')} after {took}. Log tail:\n{tail}")
        if stalled:
            now = time.time()
            self.stalls = [t for t in self.stalls if now - t < 1200] + [now]
            if len(self.stalls) >= 3:
                self.stop_reason = ("three stalls within 20 minutes: the GPU driver is probably stuck. "
                                    "Close this, run `wsl --shutdown` in Windows PowerShell, reopen "
                                    "Ubuntu and start again (finished classes stay cached).")
        if job.attempt <= self.a.retries and not self.stop_reason:
            job.attempt += 1
            job.state = "queued"
            self.say(f"{job.name}: will retry in a fresh process (attempt {job.attempt})")
            return [job]
        job.state, job.note = "failed", why
        return []

    # ---------------------------------------------------------------- status
    def render(self, jobs: List[Job]) -> str:
        now = time.time()
        mi = meminfo()
        total, avail = mi["MemTotal"] / GB, mi["MemAvailable"] / GB
        swap = (mi["SwapTotal"] - mi["SwapFree"]) / GB
        g = self.gpu
        gtxt = (f"GPU {g['util']:.0f}%  {g['mem_used_gb']:.1f}/{g['mem_total_gb']:.1f} GB  "
                f"{g['temp']:.0f}C  {g['power']:.0f}W" if g else "GPU: nvidia-smi DOES NOT ANSWER")
        cnt = {s: sum(j.state == s for j in jobs) for s in ("done", "running", "queued", "failed")}
        led = Ledger(self.run_dir).snapshot()
        held = {r: sum(v["bytes"] for k, v in led["held"].items() if k.endswith("/" + r)) / GB
                for r in ("ram", "gpu")}
        waits = [f"{v['label']} {v['bytes'] / GB:.1f} GB {k.split('/')[1].upper()} "
                 f"({fmt_dur(now - v['since'])})" for k, v in led["waiting"].items()]
        out = [f"==== DAGM check {time.strftime('%H:%M:%S')}  elapsed {fmt_dur(now - self.t0)}  "
               f"done {cnt['done']}/{len(jobs)}  running {cnt['running']}  queued {cnt['queued']}  "
               f"failed {cnt['failed']}  (max jobs {self.jobs_max}) ====",
               f"CPU {self.cpu.percent():.0f}% of {os.cpu_count()} threads  RAM used {total - avail:.1f} of "
               f"{total:.1f} GB (available {avail:.1f}, kept free {self.a.ram_reserve_gb})  "
               f"swap {swap:.1f} GB  {gtxt}",
               f"reserved: RAM {held['ram']:.1f}/{led['budget'].get('ram', 0) / GB:.1f} GB  "
               f"GPU {held['gpu']:.1f}/{led['budget'].get('gpu', 0) / GB:.1f} GB"
               + (f"  waiting: {'; '.join(waits)}" if waits else ""),
               f" {'job':<9} {'state':<8} {'try':>3}  {'stage':<9} {'progress':<26} {'in stage':>8} "
               f"{'ETA':>7} {'mem':>6} {'CPU':>6}"]
        for j in jobs:
            if j.state == "running":
                st = read_json(self.status_path(j)) or {}
                done, tot = st.get("done", 0), st.get("total", 0)
                el = now - st.get("stage_started", now)
                pr = f"{done:,}/{tot:,} {100 * done / tot:4.1f}%" if tot else (st.get("message") or "")[:26]
                rf = st.get("rate_from")
                eta = None
                if rf and tot and done > rf[1] and now > rf[0]:
                    eta = (now - rf[0]) * (tot - done) / (done - rf[1])
                out.append(f" {j.name:<9} {j.state:<8} {j.attempt:>3}  {st.get('stage', 'start'):<9} "
                           f"{pr:<26} {fmt_dur(el):>8} {fmt_dur(eta):>7} {j.pss_gb:5.1f}G "
                           f"{j.cpu_pct:5.0f}%")
                if st.get("message") and tot:
                    out.append(f" {'':<9} {st['message']}")
            elif j.state in ("done", "failed"):
                out.append(f" {j.name:<9} {j.state:<8} {j.attempt:>3}  {j.note}  "
                           f"({fmt_dur(j.finished - j.first_started)})")
            else:
                out.append(f" {j.name:<9} {j.state:<8} {j.attempt:>3}")
        return "\n".join(out)

    def sample_usage(self, jobs: List[Job]) -> None:
        hz = os.sysconf("SC_CLK_TCK")
        for j in jobs:
            if j.state != "running" or j.proc is None:
                continue
            u = group_usage(j.proc.pid)
            now = time.time()
            if j.ticks_at and now > j.ticks_at:
                # per process, so helpers that exited in between do not count negative;
                # a process started in between counts with all its time
                used = sum(max(0, t - j.ticks.get(pid, 0)) for pid, t in u["ticks"].items())
                j.cpu_pct = 100.0 * used / hz / (now - j.ticks_at)
            j.ticks, j.ticks_at, j.pss_gb = u["ticks"], now, u["pss_gb"]

    # ------------------------------------------------------------------ loop
    def run(self) -> int:
        a = self.a
        jobs = self.plan()
        if not jobs:
            self.say("Nothing to do: every requested class is cached and no verify job was asked for.")
            return 0
        queue = list(jobs)
        running: List[Job] = []
        self.jobs_max = a.jobs
        swap0 = None

        def on_signal(sig, frame):
            self.interrupted = True

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, on_signal)
        self.say("queue: " + ", ".join(j.name for j in queue))
        while queue or running:
            if self.interrupted or self.stop_reason:
                break
            now = time.time()
            for j in list(running):
                rc = j.proc.poll()
                if rc is None:
                    st = read_json(self.status_path(j))
                    upd = st["updated"] if st else j.started
                    if now - upd > a.stall + 120 and j.kill_reason != "stale":
                        self.say(f"{j.name}: no heartbeat for {fmt_dur(now - upd)}; stopping it")
                        j.kill_reason = "stale"
                        kill_group(j.proc.pid)
                    elif st and st.get("stage") != j.last_stage:
                        if j.last_stage and j.last_stage not in ("start", "load"):
                            took = st.get("stages", {}).get(j.last_stage)
                            self.say(f"{j.name}: {j.last_stage} finished"
                                     + (f" in {fmt_dur(took)}" if took is not None else "")
                                     + f", now {st.get('stage')}")
                        j.last_stage = st.get("stage")
                    continue
                running.remove(j)
                kill_group(j.proc.pid, grace=2.0)      # any helper left behind
                queue[:0] = self.handle_exit(j, rc)
            mi = meminfo()
            avail = mi["MemAvailable"] / GB
            swap = (mi["SwapTotal"] - mi["SwapFree"]) / GB
            swap0 = swap if swap0 is None else min(swap0, swap)
            live = [x for x in running if x.kill_reason != "memory"]
            if len(live) > 0 and (avail < a.ram_emergency_gb or swap - swap0 > 1.0):
                victim = max(live, key=lambda x: x.started)
                self.say(f"MEMORY GUARD: available RAM {avail:.1f} GB, swap {swap:.1f} GB. Stopping "
                         f"{victim.name} (newest) and running at most {max(1, len(live) - 1)} jobs from now on")
                victim.kill_reason = "memory"
                kill_group(victim.proc.pid)
                self.jobs_max = max(1, len(live) - 1)
            while (queue and len(running) < self.jobs_max
                   and avail - a.ram_reserve_gb >= (BASE_RAM + 1.2 * queue[0].n_images * 512 * 512) / GB):
                j = queue.pop(0)
                self.launch(j)
                running.append(j)
                avail = meminfo()["MemAvailable"] / GB
            if now - self.gpu_at >= 10:
                self.gpu, self.gpu_at = gpu_readout(), now
                if self.gpu is None:
                    self.say("WARNING: nvidia-smi did not answer within 8 s")
            if now - self.last_print >= a.every:
                self.sample_usage(jobs)
                self.block(self.render(jobs))
                self.last_print = now
            time.sleep(2.0)

        if running:
            why = "Ctrl+C" if self.interrupted else self.stop_reason
            self.say(f"STOPPING all jobs ({why})")
            for j in running:
                kill_group(j.proc.pid)
                j.proc.wait()
                j.state, j.note, j.finished = "failed", "stopped", time.time()
        self.block(self.render(jobs))
        return self.finish(jobs)

    def finish(self, jobs: List[Job]) -> int:
        ok = True
        changed = [p for p, d in self.before.items() if not Path(p).is_file() or _sha256(Path(p)) != d]
        if changed:
            ok = False
            self.say("CACHE INTEGRITY: these files changed during the run: " + ", ".join(changed))
        elif self.before:
            self.say(f"cache integrity: all {len(self.before)} cache files that existed before the run are "
                     "unchanged (SHA-256)")
        bad_verify = [j for j in jobs if j.mode == "verify" and not j.result.get("identical")]
        if any(j.mode == "verify" for j in jobs) and not bad_verify:
            self.say("verify: the new code reproduced every cached class exactly")
        if bad_verify and not self.a.smoke:
            ok = False
            written = [str(self.cache_paths[j.cls]) for j in jobs
                       if j.mode == "run" and j.state == "done" and self.cache_paths[j.cls].is_file()]
            if any(j.result.get("identical") is False for j in bad_verify) and written:
                q = REPO / "experiments" / "cache" / f"quarantine_{self.run_dir.name}"
                q.mkdir(parents=True, exist_ok=True)
                for w in written:
                    shutil.move(w, q / Path(w).name)
                self.say(f"QUARANTINE: a verify job differs, so the {len(written)} caches written in this "
                         f"run were moved to {q}")
        failed = [j.name for j in jobs if j.state != "done"]
        if failed:
            ok = False
            self.say("not finished: " + ", ".join(failed) + " (run the same command again to do only these)")
        summary = {"run_dir": str(self.run_dir), "ok": ok, "stop_reason": self.stop_reason,
                   "interrupted": self.interrupted, "elapsed_s": round(time.time() - self.t0, 1),
                   "jobs": [{"job": j.name, "state": j.state, "attempts": j.attempt, "requeues": j.requeues,
                             "note": j.note, "result": j.result} for j in jobs]}
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        self.say(f"summary: {self.run_dir / 'summary.json'}")
        if ok and not self.a.smoke:
            self.say("ALL DONE. Next: python scripts/phase4_dataset_report.py --only dagm --no-figures")
        return 0 if ok else 1


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _parse_classes(s: str) -> List[int]:
    out = sorted({int(x) for x in s.replace(" ", "").split(",") if x}) if s else []
    if any(c < 1 or c > 10 for c in out):
        raise argparse.ArgumentTypeError("DAGM classes are 1 to 10")
    return out


def main(argv=None) -> int:
    cpu = os.cpu_count() or 4
    p = argparse.ArgumentParser(description="Parallel DAGM shared-content check with memory guards, "
                                            "hang watchdog and live status.")
    p.add_argument("--config", default="configs/neu_baseline.yaml")
    p.add_argument("--classes", type=_parse_classes, default=list(range(1, 11)),
                   help="classes to compute if not cached yet (default: all 10)")
    p.add_argument("--verify", type=_parse_classes, default=[1, 2],
                   help="cached classes to recompute and compare (default: 1,2; empty string: none)")
    p.add_argument("--jobs", type=int, default=4, help="classes in flight at once (default 4)")
    p.add_argument("--helpers", type=int, default=0,
                   help="forked E1 processes per class (default: half the CPU threads less 2, max 12)")
    p.add_argument("--ram-reserve-gb", type=float, default=2.5, help="RAM never reserved by jobs")
    p.add_argument("--ram-emergency-gb", type=float, default=1.2,
                   help="below this much available RAM the newest job is stopped")
    p.add_argument("--gpu-reserve-gb", type=float, default=0.7, help="GPU memory never reserved by jobs")
    p.add_argument("--stall", type=float, default=300.0, help="seconds without progress before a job is "
                                                              "treated as hung")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--every", type=float, default=60.0, help="seconds between status blocks")
    p.add_argument("--smoke", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--status", action="store_true", help="print the latest status block and exit")
    a = p.parse_args(argv)
    os.chdir(REPO)
    if a.status:
        for base in ("phase4_dagm", "phase4_dagm_smoke"):
            f = REPO / "experiments" / "logs" / base / "latest" / "status.txt"
            if f.is_file():
                age = time.time() - f.stat().st_mtime
                print(f"{f} (written {fmt_dur(age)} ago)\n" + f.read_text())
        return 0

    from idefect.config import load_config

    cfg = load_config(a.config)
    lock_dir = REPO / "experiments" / "logs"
    lock = open(lock_dir / ".phase4_dagm.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another DAGM check is running (lock held). Stop it first.")
        return 1
    o = Orchestrator(a, cfg)
    g = o.gpu
    if g is None:
        o.say("STOP: nvidia-smi does not answer. Run `wsl --shutdown` in Windows PowerShell, reopen Ubuntu, "
              "try again.")
        return 1
    o.helpers = a.helpers or max(2, min(12, cpu // 2 - 2))
    mi = meminfo()
    ram_budget = mi["MemAvailable"] - a.ram_reserve_gb * GB
    gpu_budget = (g["mem_total_gb"] - g["mem_used_gb"] - a.gpu_reserve_gb) * GB
    Ledger.create(o.run_dir, {"ram": int(ram_budget), "gpu": int(gpu_budget)})
    cache_dir = REPO / cfg.data.duplicates.cache_dir
    o.before = {str(f): _sha256(f) for f in sorted(cache_dir.glob("*.json"))} if cache_dir.is_dir() else {}
    o.say(f"run folder: {o.run_dir}")
    o.say(f"machine: {cpu} CPU threads, RAM available {mi['MemAvailable'] / GB:.1f} of "
          f"{mi['MemTotal'] / GB:.1f} GB, GPU {g['mem_total_gb'] - g['mem_used_gb']:.1f} of "
          f"{g['mem_total_gb']:.1f} GB free, {g['temp']:.0f}C")
    o.say(f"plan: up to {a.jobs} classes at once, {o.helpers} E1 helpers each; reservable RAM "
          f"{ram_budget / GB:.1f} GB, GPU {gpu_budget / GB:.1f} GB; hang watchdog {fmt_dur(a.stall)}; "
          f"status every {fmt_dur(a.every)}")
    if g["mem_used_gb"] > 1.0:
        o.say(f"note: {g['mem_used_gb']:.1f} GB of GPU memory is already in use by something else")
    o.say(f"{len(o.before)} cache files fingerprinted; they must be unchanged at the end")
    try:
        return o.run()
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
