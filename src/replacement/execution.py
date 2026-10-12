"""Partition intact generation batches across GPUs and merge audited worker outputs."""

import json
import os
import shutil

from src.io import write_json


def partition_samples(records, batch_size, workers):
    if batch_size <= 0 or workers <= 0:
        raise ValueError("Positive batch size and worker count required")
    partitions = [[] for _ in range(workers)]
    for index, start in enumerate(range(0, len(records), batch_size)):
        partitions[index % workers].extend(records[start : start + batch_size])
    return [part for part in partitions if part]


def merge_workers(root, shards, plan, records):
    """Keep shard logs/manifests and assemble the normal six-arm evaluation layout."""
    for group in plan["groups"]:
        name = group["name"]
        folder = root / name
        images = folder / "images"
        images.mkdir(parents=True, exist_ok=False)
        hashes, manifests, calls, statistics = [], [], [], []
        starts, ends = [], []
        for shard in shards:
            source = shard / name
            if (source / "status.txt").read_text().strip() != "GENERATED":
                raise ValueError("Incomplete generation shard: " + str(source))
            shard_records = json.loads((shard / "samples.json").read_text())
            expected = [f"{row['sample_id']:06d}.png" for row in shard_records]
            actual = sorted(path.name for path in (source / "images").glob("*.png"))
            if actual != expected:
                raise ValueError("Shard sample coverage mismatch: " + str(source))
            shard_hashes = json.loads((source / "noise_hashes.json").read_text())
            if [row["sample_id"] for row in shard_hashes] != [
                row["sample_id"] for row in shard_records
            ]:
                raise ValueError("Shard noise manifest mismatch")
            config = json.loads((source / "config.json").read_text())
            if (
                config["group"] != group
                or config["batch_size"] != plan["generation_batch_size"]
                or config["num_images"] != len(shard_records)
                or config["common"] != dict(plan["config"], **group.get("overrides", {}))
            ):
                raise ValueError("Shard configuration mismatch")
            for filename in expected:
                destination = images / filename
                if destination.exists():
                    raise ValueError("Duplicate sample across shards: " + filename)
                try:
                    os.link(source / "images" / filename, destination)
                except OSError:
                    shutil.copy2(source / "images" / filename, destination)
            manifest = json.loads((source / "manifest.json").read_text())
            if manifest["code_sha256"] != plan["code_sha256"]:
                raise ValueError("Shard source changed")
            manifests.append(manifest)
            starts.append(manifest["started"])
            ends.append((source / "generation.json").stat().st_mtime)
            hashes.extend(shard_hashes)
            calls.extend(
                dict(row, shard=shard.name)
                for row in json.loads((source / "network_audit.json").read_text())["forward_calls"]
            )
            statistics.append(json.loads((source / "generation.json").read_text()))
        hashes.sort(key=lambda row: row["sample_id"])
        if [row["sample_id"] for row in hashes] != [row["sample_id"] for row in records]:
            raise ValueError("Merged sample coverage mismatch")
        write_json(folder / "noise_hashes.json", hashes)
        write_json(
            folder / "config.json",
            dict(
                common=dict(plan["config"], **group.get("overrides", {})),
                group=group,
                mode=plan["mode"],
                batch_size=plan["generation_batch_size"],
                num_images=len(records),
            ),
        )
        write_json(
            folder / "manifest.json",
            dict(workers=manifests, shard_directories=[str(p) for p in shards]),
        )
        write_json(
            folder / "network_audit.json",
            dict(
                forward_calls=calls,
                nfe_definition=(
                    "One time/state evaluation, with CFG branches batched in one forward"
                ),
            ),
        )
        write_json(
            folder / "generation.json",
            dict(
                num_images=len(records),
                steps=plan["config"]["steps"],
                nfe_per_sample=statistics[0]["nfe_per_sample"],
                batch_network_calls=sum(row["batch_network_calls"] for row in statistics),
                sampling_seconds_sum=sum(row["sampling_seconds"] for row in statistics),
                parallel_wall_span_seconds=max(ends) - min(starts),
                max_allocated_gib=max(row["max_allocated_gib"] for row in statistics),
                max_reserved_gib=max(row["max_reserved_gib"] for row in statistics),
                validated_png_count=sum(row["validated_png_count"] for row in statistics),
                workers=statistics,
            ),
        )
        (folder / "status.txt").write_text("GENERATED\n")
