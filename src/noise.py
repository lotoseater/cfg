"""Shared-bank or per-ID CPU noise, independent of batch/shard partitioning."""

import torch


def initial_noise(sample_ids, config):
    """CPU per-ID generators make input independent of batching and sharding."""
    if config.get("noise"):
        import numpy as np

        from ..paths import BASE, bounded

        bank = np.load(bounded(BASE / config["noise"]), mmap_mode="r", allow_pickle=False)
        if (
            bank.shape != (config["noise_num_images"], 3, config["img_size"], config["img_size"])
            or bank.dtype != np.float32
        ):
            raise ValueError("Noise bank shape/dtype mismatch")
        indices = [int(i) * config.get("noise_index_stride", 1) for i in sample_ids]
        return torch.from_numpy(np.array(bank[indices], copy=True))
    return torch.stack(
        [
            torch.randn(
                (3, config["img_size"], config["img_size"]),
                generator=torch.Generator(device="cpu").manual_seed(config["eval_seed"] + int(i)),
                dtype=torch.float32,
            )
            for i in sample_ids
        ]
    )
