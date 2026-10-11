"""Validate the shared experiment protocol and resolve per-group overrides."""

import json
import re
from pathlib import Path

from .paths import BASE, CONFIG_DIR, bounded


def load_config(path=None):
    config = json.loads(Path(path or CONFIG_DIR / "baseline50.json").read_text())
    if config["model"] != "JiT-B/16" or config["img_size"] != 256:
        raise ValueError("This experiment requires JiT-B/16 at 256x256")
    if config["steps"] not in (25, 50, 100) or config["num_images"] not in (10000, 50000):
        raise ValueError("Protocol requires 25/50/100 steps and 10k or 50k images")
    if config["class_num"] != 1000 or config["num_images"] % config["class_num"]:
        raise ValueError("Invalid balanced ImageNet protocol")
    if config["batch_size"] <= 0:
        raise ValueError("Batch size must be positive")
    for key in ("checkpoint", "fid_stats", "inception_weights"):
        bounded(BASE / config[key])
    if config.get("noise"):
        bounded(BASE / config["noise"])
    groups = configured_groups(config)
    if not groups or len({g["name"] for g in groups}) != len(groups):
        raise ValueError("Group names must be unique and nonempty")
    for group in groups:
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", group["name"]):
            raise ValueError("Invalid group folder name")
        if group["method"] not in (
            "euler",
            "heun",
            "heun_flip",
            "ab2",
            "cab2",
            "sg",
            "sg_prev",
        ) or group["cfg"] not in (1.0, 3.0):
            raise ValueError("Invalid method/CFG")
        overrides = group.get("overrides", {})
        if set(overrides) - {"steps", "cab_theta"}:
            raise ValueError("Only steps and cab_theta may vary between groups")
        effective = group_config(config, group)
        if effective["steps"] not in (25, 50, 100) or not 0 <= effective["cab_theta"] <= 1.5:
            raise ValueError("Invalid group steps/theta")
    if config.get("velocity_diagnostics") and config["cfg1_conditional_only"]:
        raise ValueError("Velocity diagnostics require both conditional/unconditional branches")
    return config


def configured_groups(config):
    if "groups" in config:
        return config["groups"]
    return [
        dict(name=f"{method}_cfg{cfg:g}", method=method, cfg=cfg)
        for method in config["methods"]
        for cfg in config["cfg_values"]
    ]


def group_config(config, group):
    return dict(config, **group.get("overrides", {}))


def expected_nfe(method, steps):
    return (
        2 * steps - 1 if method in ("heun", "heun_flip") else 2 * steps if method == "sg" else steps
    )
