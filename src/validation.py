"""Validate generated IDs, images, pairing and diagnostic coverage."""

import json

from .baselines import existing_baselines


def validate(root):
    from PIL import Image, ImageDraw

    plan = json.loads((root / "plan.json").read_text())
    rows = json.loads((root / "samples.json").read_text())
    hashes = None
    results = []
    counts = {}
    for row in rows:
        counts[row["class_id"]] = counts.get(row["class_id"], 0) + 1
    if plan["mode"] == "formal" and (
        len(counts) != 1000 or set(counts.values()) != {plan["config"]["num_images"] // 1000}
    ):
        raise AssertionError("Class counts are not balanced")
    for group in plan["groups"]:
        folder = root / group["name"]
        noise = json.loads((folder / "noise_hashes.json").read_text())
        if hashes is None:
            hashes = noise
        elif hashes != noise:
            raise AssertionError("Initial noise differs between groups")
        paths = sorted((folder / "images").glob("*.png"))
        if [p.name for p in paths] != [f"{r['sample_id']:06d}.png" for r in rows]:
            raise AssertionError(f"Image set mismatch: {group['name']}")
        for path in paths:
            with Image.open(path) as image:
                if image.size != (256, 256) or image.mode != "RGB":
                    raise AssertionError("Image size/mode mismatch")
                image.verify()
        generation = json.loads((folder / "generation.json").read_text())
        if plan["config"].get("velocity_diagnostics"):
            geometry = json.loads((folder / "velocity_geometry/summary.json").read_text())
            if geometry["num_images"] != len(rows) or geometry["steps"] != generation["steps"]:
                raise AssertionError("Velocity geometry coverage mismatch")
        metrics = (
            json.loads((folder / "metrics.json").read_text())
            if (folder / "metrics.json").exists()
            else None
        )
        results.append(dict(group=group, generation=generation, metrics=metrics))
    if plan["mode"] == "smoke":
        chosen = [
            rows[i]["sample_id"]
            for i in sorted({0, len(rows) // 3, 2 * len(rows) // 3, len(rows) - 1})
        ]
        width, height = 180 + 128 * len(chosen), 30 + 128 * len(plan["groups"])
        sheet = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(sheet)
        for col, sample_id in enumerate(chosen):
            draw.text((180 + 128 * col, 8), f"ID {sample_id:06d}", fill="black")
        for row, group in enumerate(plan["groups"]):
            draw.text((8, 30 + 128 * row + 50), group["name"], fill="black")
            for col, sample_id in enumerate(chosen):
                with Image.open(root / group["name"] / "images" / f"{sample_id:06d}.png") as image:
                    sheet.paste(image.resize((128, 128)), (180 + 128 * col, 30 + 128 * row))
        sheet.save(root / "preview.png")
    return dict(
        mode=plan["mode"],
        image_count_per_group=len(rows),
        identical_initial_noise=True,
        class_counts=counts,
        results=results,
        external_baselines=existing_baselines(plan["config"]),
    )
