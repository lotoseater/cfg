"""Durable server lifecycle: stop verified occupier, smoke, formal, then restore."""

import argparse
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from src.io import digest, write_json
from src.paths import BASE, PROJECT_ROOT, bounded, environment

OCCUPANCY_SCRIPT = Path("/mnt/data/LONG/hongxu/take_gpu.py")
ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "LD_LIBRARY_PATH",
    "OMP_NUM_THREADS",
    "PYTHONPATH",
)


def active_pids():
    text = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    )
    return {int(line.strip()) for line in text.splitlines() if line.strip()}


def snapshot(pid):
    # Only the specifically authorized process is inspected; never enumerate other /proc entries.
    proc = Path(f"/proc/{pid}")
    cwd = (proc / "cwd").resolve()
    exe = (proc / "exe").resolve()
    argv = (proc / "cmdline").read_bytes().decode().strip("\0").split("\0")
    if len(argv) != 2 or (cwd / argv[1]).resolve() != OCCUPANCY_SCRIPT:
        raise RuntimeError("PID is not the exact authorized take_gpu.py")
    if cwd != OCCUPANCY_SCRIPT.parent or exe != cwd / "miniconda3/envs/DFlow/bin/python3.11":
        raise RuntimeError("Occupancy launch context mismatch")
    env = dict(
        item.split("=", 1)
        for item in (proc / "environ").read_bytes().decode().split("\0")
        if "=" in item
    )
    return dict(
        pid=pid,
        cwd=str(cwd),
        exe=str(exe),
        argv=argv,
        start_ticks=(proc / "stat").read_text().rsplit(")", 1)[1].split()[19],
        env={key: env[key] for key in ENV_KEYS if key in env},
        script_sha256=digest(OCCUPANCY_SCRIPT),
    )


def state(root, status, **kwargs):
    temporary = root / "state.json.writing"
    temporary.write_text(
        json.dumps(dict(status=status, updated=time.time(), **kwargs), indent=2) + "\n"
    )
    temporary.replace(root / "state.json")
    print(status, json.dumps(kwargs), flush=True)


def original_alive(meta):
    try:
        return snapshot(meta["pid"])["start_ticks"] == meta["start_ticks"]
    except (FileNotFoundError, ProcessLookupError, RuntimeError):
        return False


def restore(root, meta):
    state(root, "WAITING_GPU_RELEASE")
    while active_pids() or original_alive(meta):
        time.sleep(15)
    if digest(OCCUPANCY_SCRIPT) != meta["script_sha256"]:
        raise RuntimeError("Occupancy script changed; cannot restore unchanged")
    session = "cfg_occupancy_restored_" + root.name
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise RuntimeError("Restore session already exists")
    unset = [item for key in ENV_KEYS if key not in meta["env"] for item in ("-u", key)]
    command = "exec env " + shlex.join(
        unset
        + [f"{k}={v}" for k, v in meta["env"].items()]
        + ["PYTHONUNBUFFERED=1", meta["exe"], str(OCCUPANCY_SCRIPT)]
    )
    command += " >> " + shlex.quote(str(root / "restored_occupancy.log")) + " 2>&1"
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "-c", meta["cwd"], command], check=True
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        pids = active_pids()
        if len(pids) == 1:
            pid = next(iter(pids))
            now = snapshot(pid)
            if all(now[key] == meta[key] for key in ("cwd", "exe", "argv", "env", "script_sha256")):
                write_json(
                    root / "restoration.json",
                    dict(
                        status="RESTORED",
                        pid=pid,
                        session=session,
                        snapshot=now,
                        finished=time.time(),
                    ),
                )
                return pid
        time.sleep(2)
    raise RuntimeError("Restored occupier has not appeared on GPUs within 120s")


def run_child(root, mode, gpus, smoke=None):
    command = [sys.executable, "-m", "scripts.run", mode, "--gpus", gpus]
    if smoke is not None:
        command += ["--smoke", str(smoke)]
    elif mode == "smoke":
        command += ["--smoke-images", "128", "--smoke-batch-size", "128"]
    log_path = root / (mode + "_controller.log")
    with log_path.open("x") as log:
        child = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment(0),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state(root, mode.upper() + "_RUNNING", controller_pid=child.pid, log=str(log_path))
        try:
            code = child.wait()
        except BaseException:
            # Do not restore over generation processes even when the guard is interrupted.
            child.wait()
            raise
    if code != 0:
        raise RuntimeError(f"{mode} controller failed; see {log_path}")
    line = next(
        (line for line in log_path.read_text().splitlines() if line.startswith("RUN_ROOT ")), None
    )
    if line is None:
        raise RuntimeError("Missing run root in log")
    result = bounded(line.removeprefix("RUN_ROOT "))
    expected = "SMOKE_COMPLETE" if mode == "smoke" else "COMPLETE"
    if (result / "status.txt").read_text().strip() != expected:
        raise RuntimeError("Run did not finish completely")
    return result


def main(args):
    root = bounded(args.control_root)
    root.mkdir(exist_ok=False, parents=True)
    with (BASE / "HDD/cfg/occupancy_control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lifecycle(args, root)


def lifecycle(args, root):
    meta = snapshot(args.pid)
    if active_pids() != {args.pid}:
        raise RuntimeError("Other GPU tasks are present; do not stop anything")
    write_json(root / "occupancy_snapshot.json", meta)
    if snapshot(args.pid) != meta:
        raise RuntimeError("PID identity changed before stop")
    stopped = False
    failure = None
    smoke = formal = None
    try:
        os.kill(args.pid, signal.SIGINT)
        stopped = True
        state(root, "OCCUPANCY_SIGINT", pid=args.pid)
        deadline = time.monotonic() + 60
        while (active_pids() or original_alive(meta)) and time.monotonic() < deadline:
            time.sleep(1)
        if active_pids() or original_alive(meta):
            raise RuntimeError("Occupier or GPU tasks remain after SIGINT")
        state(root, "GPUS_FREE")
        with (root / "historical_replay.log").open("x") as log:
            replay = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "scripts.audit_baselines",
                    "--output",
                    str(root / "historical_replay.json"),
                ],
                cwd=PROJECT_ROOT,
                env=environment(0),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = replay.wait()
            except BaseException:
                replay.wait()
                raise
            if code:
                raise RuntimeError("Historical Heun replay failed")
        smoke = run_child(root, "smoke", args.gpus)
        state(
            root,
            "SMOKE_READY",
            smoke_root=str(smoke),
            next_action="Waiting for proceed.json after visual inspection",
        )
        while not (root / "proceed.json").is_file():
            if (root / "cancel.json").is_file():
                raise RuntimeError("Formal launch cancelled; restoring occupancy")
            time.sleep(5)
        proceed = json.loads((root / "proceed.json").read_text())
        if proceed.get("smoke_root") != str(smoke):
            raise RuntimeError("Proceed request identifies a different smoke run")
        formal = run_child(root, "formal", args.gpus, smoke)
        write_json(
            root / "completed.json",
            dict(
                smoke_root=str(smoke),
                formal_root=str(formal),
                all_generation_and_metrics_complete=True,
            ),
        )
    except BaseException as error:
        failure = str(error)
        state(root, "RUN_FAILED", error=failure, smoke_root=str(smoke), formal_root=str(formal))
    finally:
        if stopped:
            try:
                pid = restore(root, meta)
                state(
                    root,
                    "FAILED_RESTORED" if failure else "COMPLETE_RESTORED",
                    restored_pid=pid,
                    error=failure,
                    smoke_root=str(smoke),
                    formal_root=str(formal),
                )
            except BaseException as error:
                state(root, "RESTORE_FAILED", error=str(error), original_failure=failure)
                raise
    if failure:
        raise RuntimeError(failure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    main(parser.parse_args())
