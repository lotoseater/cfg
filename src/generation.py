"""Generate a single group with exclusive outputs and audited model evaluations."""

import hashlib
import json
import os
import sys
import time

from src.config import expected_nfe, group_config
from src.io import digest, write_json
from src.models import build_denoiser
from src.paths import BASE, PROJECT_ROOT, bounded
from src.provenance import assert_upstream, code_hashes, git_revision


def generate(root, name):
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    import numpy as np
    import torch
    from PIL import Image

    from src.samplers import Sampler, initial_noise

    root = bounded(root)
    plan = json.loads((root / "plan.json").read_text())
    group = next(row for row in plan["groups"] if row["name"] == name)
    config = group_config(plan["config"], group)
    folder = root / name
    folder.mkdir(exist_ok=False)
    try:
        (folder / "status.txt").write_text("PREPARING\n")
        assert_upstream(config)
        current_hashes = code_hashes()
        revision = git_revision()
        if current_hashes != plan["code_sha256"]:
            raise RuntimeError("Project source changed after plan creation")
        for key in ("checkpoint",):
            if digest(BASE / config[key]) != config[key + "_sha256"]:
                raise RuntimeError(f"Asset hash mismatch: {key}")
        torch.set_num_threads(4)
        torch.manual_seed(config["eval_seed"])
        model = build_denoiser(config, group["cfg"])
        checkpoint = torch.load(BASE / config["checkpoint"], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint[config["ema"]], strict=True)
        del checkpoint
        model = model.cuda().eval().requires_grad_(False)
        if config.get("velocity_diagnostics"):
            from src.diagnostics import VelocityGeometry

            geometry = VelocityGeometry(group["method"])
        else:
            geometry = None
        sampler = Sampler(model.net, config, group["cfg"], diagnostics=geometry)
        network_calls = []

        def audit(module, inputs, output):
            network_calls.append(
                dict(
                    grad_enabled=torch.is_grad_enabled(),
                    output_requires_grad=output.requires_grad,
                    network_batch=inputs[0].shape[0],
                )
            )

        hook = model.net.register_forward_hook(audit)
        records = json.loads((root / "samples.json").read_text())
        images = folder / "images"
        images.mkdir()
        write_json(
            folder / "config.json",
            dict(
                common=config,
                group=group,
                mode=plan["mode"],
                batch_size=plan["generation_batch_size"],
                num_images=len(records),
            ),
        )
        write_json(
            folder / "manifest.json",
            dict(
                pid=os.getpid(),
                command=sys.argv,
                checkpoint_sha256=config["checkpoint_sha256"],
                ema=config["ema"],
                samples_sha256=digest(root / "samples.json"),
                code_sha256=current_hashes,
                upstream_commit=config["upstream_commit"],
                project_git_commit=revision,
                project_version="Exact source recorded by code_sha256; Git revision recorded when available",
                upstream_module=str(PROJECT_ROOT / "src/upstream/denoiser.py"),
                torch=torch.__version__,
                gpu=torch.cuda.get_device_name(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                started=time.time(),
                noise_policy=config["noise_policy"],
            ),
        )
        torch.cuda.reset_peak_memory_stats()
        total_begin = time.perf_counter()
        sampling_seconds = 0.0
        raw_nfe = 0
        batch_size = plan["generation_batch_size"]
        finite_stats = dict(min=1e30, max=-1e30, saturated_pixels=0, pixels=0)
        noise_hashes = []
        (folder / "status.txt").write_text("GENERATING\n")
        with (folder / "batches.jsonl").open("x") as batches:
            for start in range(0, len(records), batch_size):
                block = records[start : start + batch_size]
                ids = [r["sample_id"] for r in block]
                labels = torch.tensor([r["class_id"] for r in block], device="cuda")
                cpu_noise = initial_noise(ids, config)
                expected_seeds = [
                    config["eval_seed"] if config.get("noise") else config["eval_seed"] + i
                    for i in ids
                ]
                if [r["seed"] for r in block] != expected_seeds:
                    raise RuntimeError("Manifest seed mismatch")
                noise_hashes.extend(
                    dict(sample_id=i, sha256=hashlib.sha256(n.numpy().tobytes()).hexdigest())
                    for i, n in zip(ids, cpu_noise)
                )
                noise = cpu_noise.cuda()
                torch.cuda.synchronize()
                begin = time.perf_counter()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    output = sampler.generate(noise, labels, group["method"])
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - begin
                sampling_seconds += elapsed
                geometry_record = None
                if geometry is not None:
                    geometry_record = geometry.save_batch(
                        folder / "velocity_geometry", start, ids, [r["class_id"] for r in block]
                    )
                expected = expected_nfe(group["method"], config["steps"])
                if len(sampler.trace) != expected or any(r["t"] >= 1 for r in sampler.trace):
                    raise AssertionError("NFE or endpoint audit failed")
                raw_nfe += len(sampler.trace)
                if not torch.isfinite(output).all() or output.shape != noise.shape:
                    raise FloatingPointError("Invalid generated tensor")
                finite_stats["min"] = min(finite_stats["min"], float(output.min()))
                finite_stats["max"] = max(finite_stats["max"], float(output.max()))
                finite_stats["saturated_pixels"] += int((output.abs() >= 1).sum())
                finite_stats["pixels"] += output.numel()
                if plan["mode"] == "smoke" and start == 0:
                    np.save(folder / "raw_first2.npy", output[:2].float().cpu().numpy())
                    write_json(folder / "model_call_trace.json", sampler.trace)
                arrays = np.round(
                    np.clip(((output.float() + 1) / 2).cpu().numpy() * 255, 0, 255)
                ).astype(np.uint8)
                for sample_id, array in zip(ids, arrays):
                    Image.fromarray(array.transpose(1, 2, 0)).save(images / f"{sample_id:06d}.png")
                record = dict(
                    start=start,
                    end=start + len(ids),
                    nfe_per_sample=expected,
                    sampling_seconds=elapsed,
                    batch_size=len(ids),
                    velocity_geometry=geometry_record,
                )
                batches.write(json.dumps(record) + "\n")
                batches.flush()
                print(name, "BATCH", json.dumps(record), flush=True)
                del output, noise, cpu_noise
        hook.remove()
        if geometry is not None:
            from src.diagnostics import summarize_geometry

            geometry_summary = summarize_geometry(
                folder / "velocity_geometry", [r["sample_id"] for r in records]
            )
        else:
            geometry_summary = None
        if any(row["grad_enabled"] or row["output_requires_grad"] for row in network_calls):
            raise AssertionError("Unexpected network gradients")
        if len(network_calls) != raw_nfe:
            raise AssertionError("Network hook count differs from NFE")
        paths = sorted(images.glob("*.png"))
        if [p.name for p in paths] != [f"{r['sample_id']:06d}.png" for r in records]:
            raise AssertionError("Image IDs/count mismatch")
        for path in paths:
            with Image.open(path) as image:
                image.verify()
        write_json(folder / "noise_hashes.json", noise_hashes)
        write_json(
            folder / "network_audit.json",
            dict(
                forward_calls=network_calls,
                nfe_definition="One time/state evaluation, with CFG branches batched in one forward",
            ),
        )
        write_json(
            folder / "generation.json",
            dict(
                num_images=len(records),
                steps=config["steps"],
                nfe_per_sample=expected,
                batch_network_calls=raw_nfe,
                sampling_seconds=sampling_seconds,
                generation_seconds=time.perf_counter() - total_begin,
                images_per_sampling_second=len(records) / sampling_seconds,
                max_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                max_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                velocity_geometry=geometry_summary,
                timing_includes_velocity_diagnostics=geometry is not None,
                tensor_statistics=finite_stats,
                saturation_fraction=finite_stats["saturated_pixels"] / finite_stats["pixels"],
                network_grad_enabled_calls=0,
                network_output_requires_grad_calls=0,
                validated_png_count=len(paths),
            ),
        )
        (folder / "status.txt").write_text(
            "SMOKE_COMPLETE\n" if plan["mode"] == "smoke" else "GENERATED\n"
        )
        print(name, "GENERATION_COMPLETE", flush=True)
    except BaseException:
        (folder / "status.txt").write_text("GENERATION_FAILED\n")
        raise
