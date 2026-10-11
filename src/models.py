"""Construct the frozen JiT model with one shared inference configuration."""

import sys
from types import SimpleNamespace

from .paths import UPSTREAM_DIR


def build_denoiser(config, cfg_scale):
    sys.path.insert(0, str(UPSTREAM_DIR))
    from denoiser import Denoiser

    args = SimpleNamespace(
        model=config["model"],
        img_size=config["img_size"],
        class_num=config["class_num"],
        attn_dropout=0.0,
        proj_dropout=0.0,
        label_drop_prob=0.1,
        P_mean=-0.8,
        P_std=0.8,
        t_eps=config["t_eps"],
        noise_scale=config["noise_scale"],
        ema_decay1=0.9999,
        ema_decay2=0.9996,
        sampling_method="heun",
        num_sampling_steps=config["steps"],
        cfg=cfg_scale,
        interval_min=config["interval_min"],
        interval_max=config["interval_max"],
    )
    return Denoiser(args)
