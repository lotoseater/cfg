"""Create immutable run plans and schedule groups on explicit GPU lanes."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

from .baselines import existing_baselines
from .config import configured_groups, load_config
from .io import digest, write_json
from .paths import BASE, PROJECT_ROOT, bounded, environment
from .provenance import assert_upstream, code_hashes
from .validation import validate


def launch(args):
    config = load_config(args.config)
    assert_upstream(config)
    for key in ("checkpoint", "fid_stats", "inception_weights"):
        if digest(BASE / config[key]) != config[key + "_sha256"]:
            raise RuntimeError(f"Asset hash mismatch: {key}")
    if config.get("noise") and digest(BASE / config["noise"]) != config["noise_sha256"]:
        raise RuntimeError("Shared noise bank hash mismatch")
    baselines = existing_baselines(config, verify_images=args.mode == "smoke")
    if args.mode == "formal":
        if not args.smoke:
            raise ValueError("Formal generation requires --smoke PATH to a completed smoke run")
        smoke = bounded(args.smoke)
        if (smoke / "status.txt").read_text().strip() != "SMOKE_COMPLETE":
            raise RuntimeError("Smoke run not complete")
        old = json.loads((smoke / "plan.json").read_text())
        if old["config"] != config or old["code_sha256"] != code_hashes():
            raise RuntimeError("Config/code changed since smoke test; rerun smoke")
        if old["generation_batch_size"] != config["batch_size"]:
            raise RuntimeError("Smoke must use the configured full batch size")
        smoke_names = {group["name"] for group in old["groups"]}
        required = {group["name"] for group in configured_groups(config)}
        if not required.issubset(smoke_names):
            raise RuntimeError("Smoke run did not cover every configured group")
    gpu_ids = [int(x) for x in args.gpus.split(",")]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)) or min(gpu_ids) < 0:
        raise ValueError("Invalid GPU list")
    if (
        args.smoke_images <= 0
        or args.smoke_images > config["num_images"]
        or args.smoke_batch_size <= 0
    ):
        raise ValueError("Invalid smoke count/batch")
    parent = BASE / ("HDD/cfg/smoketest" if args.mode == "smoke" else "HDD/cfg/runs")
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y%m%d_%H%M%S_%f")
    root = parent / (args.mode + "_" + timestamp)
    root.mkdir(exist_ok=False)
    count = args.smoke_images if args.mode == "smoke" else config["num_images"]
    # Smoke samples cover the whole label space; the 10k manifest is balanced.
    ids = (
        [i * (config["num_images"] - 1) // max(count - 1, 1) for i in range(count)]
        if args.mode == "smoke"
        else list(range(count))
    )
    groups = configured_groups(config)
    plan = dict(
        mode=args.mode,
        config=config,
        groups=groups,
        num_images=count,
        generation_batch_size=args.smoke_batch_size
        if args.mode == "smoke"
        else config["batch_size"],
        code_sha256=code_hashes(),
        gpus=gpu_ids,
        smoke_reference=str(args.smoke),
        created_hong_kong=timestamp,
    )
    write_json(root / "plan.json", plan)
    write_json(root / "reused_baselines.json", baselines)
    write_json(
        root / "samples.json",
        [
            dict(
                sample_id=i,
                class_id=i // (config["num_images"] // 1000),
                seed=config["eval_seed"] if config.get("noise") else config["eval_seed"] + i,
                noise_index=i * config.get("noise_index_stride", 1),
            )
            for i in ids
        ],
    )
    (root / "status.txt").write_text("RUNNING\n")
    print("RUN_ROOT", root, flush=True)
    lanes = [groups[i :: len(gpu_ids)] for i in range(len(gpu_ids))]

    def lane(gpu, selected):
        for group in selected:
            name = group["name"]
            with (root / (name + ".log")).open("x") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.generate",
                        "--root",
                        str(root),
                        "--group",
                        name,
                    ],
                    cwd=PROJECT_ROOT,
                    env=environment(gpu),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode:
                raise RuntimeError(f"{name} generation failed; see {root / (name + '.log')}")
            if args.mode == "formal":
                with (root / (name + "_metrics.log")).open("x") as log:
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "scripts.evaluate",
                            "--root",
                            str(root),
                            "--group",
                            name,
                        ],
                        cwd=PROJECT_ROOT,
                        env=environment(gpu),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                if result.returncode:
                    (root / name / "status.txt").write_text("METRICS_FAILED\n")
                    raise RuntimeError(f"{name} evaluation failed")
            print("GROUP_COMPLETE", name, "GPU", gpu, flush=True)

    try:
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
            futures = [pool.submit(lane, gpu, selected) for gpu, selected in zip(gpu_ids, lanes)]
            # Wait for every already launched lane even if another lane fails.
            failures = []
            for future in futures:
                try:
                    future.result()
                except Exception as error:
                    failures.append(str(error))
            if failures:
                raise RuntimeError("; ".join(failures))
        summary = validate(root)
        write_json(root / "summary.json", summary)
        if args.mode == "formal":
            lines = [
                "| Method | CFG | Steps | NFE | FID-10k | images/s | Sampling seconds |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for row in summary["external_baselines"]:
                fid = row["metrics"]["metrics"]["frechet_inception_distance"]
                lines.append(
                    f"| Heun (existing) | {row['cfg']:g} | 50 | 99 | {fid:.6f} | historical | historical |"
                )
            for row in summary["results"]:
                group, generation, metrics = (
                    row["group"],
                    row["generation"],
                    row["metrics"]["metrics"],
                )
                lines.append(
                    f"| {group['name']} | {group['cfg']:g} | {generation['steps']} | {generation['nfe_per_sample']} | "
                    f"{metrics['frechet_inception_distance']:.6f} | {generation['images_per_sampling_second']:.3f} | {generation['sampling_seconds']:.3f} |"
                )
            (root / "results.md").write_text("\n".join(lines) + "\n")
        (root / "status.txt").write_text(
            "SMOKE_COMPLETE\n" if args.mode == "smoke" else "COMPLETE\n"
        )
        print("RUN_COMPLETE", root, flush=True)
    except BaseException:
        (root / "status.txt").write_text("FAILED\n")
        raise
