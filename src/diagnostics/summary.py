"""Aggregate saved per-image metrics without model calls."""

import json
from pathlib import Path

import numpy as np


def summarize_geometry(folder, expected_ids=None):
    folder = Path(folder)
    schema = json.loads((folder / "schema.json").read_text())
    values, ids = [], []
    for path in sorted(folder.glob("batch_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            ids.extend(data["sample_ids"].tolist())
            values.append(data["values"])
    if expected_ids is not None and ids != list(expected_ids):
        raise AssertionError("Geometry sample IDs differ from generated images")
    all_values = np.concatenate(values, axis=1)
    if np.isinf(all_values).any():
        raise FloatingPointError("Infinite geometry statistics")
    rows = []
    for step, block in enumerate(all_values):
        stats = {}
        for col, key in enumerate(schema["columns"]):
            column = block[:, col].astype(np.float64)
            finite = column[np.isfinite(column)]
            stats[key] = dict(
                valid_count=len(finite),
                undefined_count=len(column) - len(finite),
                mean=float(finite.mean()) if len(finite) else None,
                std=float(finite.std()) if len(finite) else None,
                p10=float(np.quantile(finite, 0.1)) if len(finite) else None,
                p50=float(np.quantile(finite, 0.5)) if len(finite) else None,
                p90=float(np.quantile(finite, 0.9)) if len(finite) else None,
            )
        rows.append(dict(step=step, time=stats["time"]["mean"], metrics=stats))
    result = dict(
        num_images=len(ids),
        steps=len(rows),
        method=schema["method"],
        extra_model_evaluations=0,
        rows=rows,
    )
    with (folder / "summary.json").open("x") as handle:
        json.dump(result, handle, allow_nan=False)
        handle.write("\n")
    return dict(
        num_images=len(ids),
        steps=len(rows),
        metric_columns=len(schema["columns"]),
        extra_model_evaluations=0,
        summary="velocity_geometry/summary.json",
    )
