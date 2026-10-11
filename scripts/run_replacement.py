"""Generate paired reference-replacement arms after explicit CUDA preflight."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from src.io import write_json
from src.paths import BASE, PROJECT_ROOT, bounded, environment
from src.provenance import code_hashes
from src.replacement.protocol import (
    DEFAULT_CONFIG,
    generator_config,
    groups,
    load_protocol,
    samples,
    validate_pairing,
    verify_assets,
)


def cuda_preflight(gpus):
    # This subprocess is the same Python and library environment as generation workers.
    program = "import torch; torch.cuda.init(); a=torch.arange(256,device='cuda',dtype=torch.float32).reshape(16,16); b=a@a.T; assert torch.cuda.is_bf16_supported(); c=a.to(torch.bfloat16); d=c@c.T; torch.cuda.synchronize(); assert torch.isfinite(b).all() and torch.isfinite(d).all(); print(torch.cuda.get_device_name())"
    for gpu in gpus:
        subprocess.run([sys.executable, "-c", program], env=environment(gpu), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "formal"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--pilot", type=Path, help="Completed, evaluated pilot required for formal generation"
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print protocol and counts, without CUDA or file writes",
    )
    args = parser.parse_args()
    study = load_protocol(args.config)
    selected = groups(study)
    records = samples(study, args.stage)
    if args.plan_only:
        print(
            json.dumps(
                dict(
                    stage=args.stage,
                    groups=selected,
                    images_per_group=len(records),
                    total_images=len(records) * len(selected),
                    protocol=study,
                ),
                indent=2,
            )
        )
        return
    if args.stage == "formal":
        if args.pilot is None:
            parser.error("Formal generation requires --pilot PATH")
        pilot = bounded(args.pilot)
        old = json.loads((pilot / "plan.json").read_text())
        if (
            old["mode"] != "pilot"
            or old["protocol"] != study
            or old["code_sha256"] != code_hashes()
        ):
            parser.error("Pilot protocol/source differs; rerun the pilot")
        if (pilot / "classwise/status.txt").read_text().strip() != "COMPLETE":
            parser.error("Pilot classwise evaluation is incomplete")
    devices = [int(value) for value in args.gpus.split(",")]
    if not devices or len(devices) != len(set(devices)) or min(devices) < 0:
        parser.error("Invalid GPU list")
    cuda_preflight(devices)
    verify_assets(study)
    config = generator_config(study)
    parent = BASE / ("HDD/cfg/smoketest" if args.stage == "pilot" else "HDD/cfg/runs")
    root = parent / datetime.now().strftime(f"replacement_{args.stage}_%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True, exist_ok=False)
    plan = dict(
        mode=args.stage,
        config=config,
        groups=selected,
        num_images=len(records),
        generation_batch_size=study["batch_size"],
        code_sha256=code_hashes(),
        gpus=devices,
        protocol=study,
    )
    write_json(root / "plan.json", plan)
    write_json(root / "samples.json", records)
    (root / "status.txt").write_text("GENERATING\n")
    print("RUN_ROOT", root, flush=True)

    def lane(gpu, assigned):
        for group in assigned:
            with (root / f"{group['name']}.log").open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.generate",
                        "--root",
                        str(root),
                        "--group",
                        group["name"],
                    ],
                    cwd=PROJECT_ROOT,
                    env=environment(gpu),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print("GROUP_COMPLETE", group["name"], gpu, flush=True)

    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [
                pool.submit(lane, gpu, selected[index :: len(devices)])
                for index, gpu in enumerate(devices)
            ]
            for future in futures:
                future.result()
        write_json(root / "pairing.json", validate_pairing(root))
        (root / "status.txt").write_text("GENERATED\n")
        print("GENERATION_COMPLETE", root, flush=True)
    except BaseException:
        (root / "status.txt").write_text("FAILED\n")
        raise


if __name__ == "__main__":
    main()
