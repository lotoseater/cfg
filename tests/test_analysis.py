"""Check saved-run analysis reports reflect input data and undefined comparisons."""

import json
import tempfile
import unittest
from pathlib import Path

from src.analysis import create_report, load_curves
from src.paths import PROJECT_ROOT


class AnalysisTests(unittest.TestCase):
    def test_report_reads_run_summaries_and_uses_their_values(self):
        values = {
            "current.cond_vs_uncond.angle_deg": 12.5,
            "current.cond_vs_uncond.norm_ratio_a_over_b": 1.2,
            "current.cond.rms": 0.6,
            "current.uncond.rms": 0.5,
            "correction_vs_base.norm_ratio_a_over_b": 0.02,
            "combined_vs_base.angle_deg": 2.0,
            "combined_vs_base.norm_ratio_a_over_b": 1.01,
            "history.cfg.angle_deg": 3.0,
            "shifted.current_vs_aux.cfg.angle_deg": 4.0,
            "correction.parallel_over_base": 0.01,
            "correction.perpendicular_over_base": 0.02,
            "correction_vs_cfg_direction.angle_deg": 123.0,
            "cab.theta_vs_base.norm_ratio_a_over_b": None,
        }
        groups = [
            dict(name=f"{method}_cfg{cfg}", method=method, cfg=cfg)
            for method in ("heun_flip", "sg", "cab2", "ab2")
            for cfg in (3, 1)
        ]
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            (run / "plan.json").write_text(json.dumps(dict(groups=groups)))
            for group in groups:
                folder = run / group["name"] / "velocity_geometry"
                folder.mkdir(parents=True)
                rows = [
                    dict(
                        step=step,
                        time=time,
                        metrics={key: dict(mean=value) for key, value in values.items()},
                    )
                    for step, time in enumerate((0.0, 0.2, 0.5, 0.9, 0.98))
                ]
                (folder / "summary.json").write_text(json.dumps(dict(num_images=4, rows=rows)))
            output = root / "analysis"
            aggregates = create_report(run, output)
            report = (output / "report.md").read_text()
            self.assertIn(
                "| AB2 | 5 | 1 | 12.500 | 1.200000 | 2.00% | 1.0100 | 2.000 | 123.00 |", report
            )
            self.assertIn("每组样本数：4", report)
            self.assertNotIn("2026-10-11", report)
            self.assertEqual(aggregates[0]["means"]["correction_vs_cfg_direction.angle_deg"], 123.0)
            self.assertIsNone(aggregates[0]["means"]["cab.theta_vs_base.norm_ratio_a_over_b"])
            self.assertEqual(load_curves(output / "velocity_curves.json"), load_curves(run))
            self.assertEqual(len(list(output.glob("*.png"))), 3)
            self.assertEqual(len(list(output.glob("*.svg"))), 3)
