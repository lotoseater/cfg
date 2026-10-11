"""Replay only the first full batch and compare pixels to old Heun PNGs."""

import argparse
import os
from pathlib import Path

from src.baselines import existing_baselines
from src.config import load_config
from src.io import digest, write_json
from src.models import build_denoiser
from src.paths import BASE, bounded
from src.provenance import assert_upstream


def main(output):
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    import numpy as np
    import torch
    from PIL import Image

    from src.samplers import Sampler, initial_noise

    c = load_config()
    assert_upstream(c)
    baselines = existing_baselines(c)
    if digest(BASE / c["checkpoint"]) != c["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint mismatch")
    torch.set_num_threads(4)
    model = build_denoiser(c, 3.0)
    ckpt = torch.load(BASE / c["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt[c["ema"]], strict=True)
    del ckpt
    model = model.cuda().eval().requires_grad_(False)
    ids = list(range(c["batch_size"]))
    noise = initial_noise(ids, c).cuda()
    labels = torch.tensor([i // 10 for i in ids], device="cuda")
    results = []
    for baseline in baselines:
        sampler = Sampler(model.net, c, baseline["cfg"])
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            generated = sampler.generate(noise, labels, "heun")
        arrays = np.round(
            np.clip(((generated.float() + 1) / 2).cpu().numpy() * 255, 0, 255)
        ).astype(np.uint8)
        exact = 0
        different_pixels = 0
        max_delta = 0
        for sample_id, array in zip(ids, arrays):
            with Image.open(Path(baseline["source_images"]) / f"{sample_id:05d}.png") as image:
                old = np.array(image)
            new = array.transpose(1, 2, 0)
            exact += int(np.array_equal(old, new))
            different_pixels += int(np.count_nonzero(old != new))
            max_delta = max(
                max_delta, int(np.abs(old.astype(np.int16) - new.astype(np.int16)).max())
            )
        row = dict(
            name=baseline["name"],
            checked_samples=len(ids),
            exact_png_pixels_images=exact,
            different_channel_values=different_pixels,
            max_uint8_difference=max_delta,
            batch_size=c["batch_size"],
            checkpoint_sha256=c["checkpoint_sha256"],
            noise_sha256=c["noise_sha256"],
        )
        results.append(row)
        print(row, flush=True)
        if exact != len(ids):
            write_json(bounded(output), dict(status="MISMATCH", results=results))
            raise AssertionError("Historical PNG replay mismatch")
    write_json(bounded(output), dict(status="EXACT", results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args().output)
