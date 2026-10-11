"""Plot immutable per-step geometry summaries; no server/model access."""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = dict(ab2="#2471a3", cab2="#8951a5", sg="#d45a21", heun_flip="#4e6468")
LABELS = dict(
    ab2="AB2",
    cab2="CAB2",
    sg="SG",
    heun_flip="Heun-flip",
    euler="Euler",
    heun="Heun",
    sg_prev="History-SG",
)
COLORS.update(euler="#777777", heun="#328f66", sg_prev="#b76394")
STYLES = dict(ab2="-", cab2="--", sg="-", heun_flip=":", euler="-", heun="--", sg_prev=":")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
    }
)


def curve(group, key, stat="mean"):
    rows = group["rows"]
    return np.asarray([r["time"] for r in rows]), np.asarray(
        [
            r["stats"].get(key, {}).get(stat)
            if r["stats"].get(key, {}).get(stat) is not None
            else np.nan
            for r in rows
        ],
        dtype=float,
    )


def middle_mean(group, key):
    t, v = curve(group, key)
    selected = v[(t >= 0.2 - 1e-6) & (t <= 0.9 + 1e-6)]
    return float(np.nanmean(selected)) if np.isfinite(selected).any() else None


def decorate(ax, cfg, title, ylabel, ylim=None):
    ax.set_title(f"{title} | CFG={cfg}")
    ax.set_xlabel("t (noise 0 -> data 1)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    if ylim:
        ax.set_ylim(*ylim)
    ax.axvspan(0.95, 1, color="#f3c5c5", alpha=0.4, zorder=0)
    ax.axvline(0.1, color="#777777", linestyle="--", linewidth=0.75, alpha=0.55)
    ax.grid(alpha=0.18)


def save(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_overview(data, output):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10.4))
    panels = [
        (
            "current.cond_vs_uncond.angle_deg",
            "Conditional vs unconditional",
            "Angle (degrees)",
            None,
            1,
        ),
        (
            "correction_vs_base.norm_ratio_a_over_b",
            "Method correction relative to base",
            "100 * ||d|| / ||v_CFG|| (%)",
            None,
            100,
        ),
        ("combined_vs_base.angle_deg", "Combined velocity vs base", "Angle (degrees)", None, 1),
    ]
    for col, cfg in enumerate((1, 3)):
        for row, (key, title, ylabel, ylim, scale) in enumerate(panels):
            ax = axes[row, col]
            for g in data["groups"]:
                if g["cfg"] != cfg:
                    continue
                t, v = curve(g, key)
                ax.plot(
                    t,
                    v * scale,
                    STYLES[g["method"]],
                    color=COLORS[g["method"]],
                    lw=1.8,
                    label=LABELS[g["method"]],
                )
            decorate(ax, cfg, title, ylabel, ylim)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.958), ncol=2, frameon=False
    )
    fig.suptitle("Velocity geometry", fontsize=17, y=0.995)
    fig.text(
        0.5,
        0.015,
        "Per-image means on each method's own trajectory. d = combined velocity - raw CFG velocity.\n"
        "Red shade: t >= 0.95. Dashed vertical line: t = 0.1. Verify clamp and CFG interval in the run config.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.9))
    save(fig, output, "velocity_geometry_overview")


def plot_branches(data, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8))
    for col, cfg in enumerate((1, 3)):
        for g in data["groups"]:
            if g["cfg"] != cfg:
                continue
            t, c = curve(g, "current.cond.rms")
            _, u = curve(g, "current.uncond.rms")
            axes[0, col].plot(
                t,
                c,
                STYLES[g["method"]],
                color=COLORS[g["method"]],
                lw=1.8,
                label=LABELS[g["method"]],
            )
            axes[0, col].plot(t, u, ":", color=COLORS[g["method"]], lw=0.8, alpha=0.75)
            t, ratio = curve(g, "current.cond_vs_uncond.norm_ratio_a_over_b")
            axes[1, col].plot(t, ratio, STYLES[g["method"]], color=COLORS[g["method"]], lw=1.8)
        decorate(axes[0, col], cfg, "Basic branch magnitudes", "RMS velocity")
        decorate(
            axes[1, col],
            cfg,
            "Conditional / unconditional norm ratio",
            "||v_cond|| / ||v_uncond||",
        )
        axes[1, col].axhline(1, color="black", lw=0.7, alpha=0.5)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.956),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Basic conditional / unconditional velocity magnitudes", fontsize=16, y=0.995)
    fig.text(
        0.5,
        0.014,
        "Conditional lines: method style; unconditional: thin dotted lines in the same color (almost overlap).\n"
        "RMS = L2 / sqrt(number of elements). Ratios are averaged from per-image ratios.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.89))
    save(fig, output, "basic_branch_magnitudes")


def plot_reference_pairs(data, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))
    for col, cfg in enumerate((1, 3)):
        for g in data["groups"]:
            if g["cfg"] != cfg:
                continue
            if g["method"] != "heun_flip":
                key = (
                    "shifted.current_vs_aux.cfg.angle_deg"
                    if g["method"] == "sg"
                    else "history.cfg.angle_deg"
                )
                t, v = curve(g, key)
                detail = (
                    "same state, shifted time" if g["method"] == "sg" else "previous state and time"
                )
                axes[0, col].plot(
                    t,
                    v,
                    STYLES[g["method"]],
                    color=COLORS[g["method"]],
                    lw=1.7,
                    label=LABELS[g["method"]].split(" / ")[0] + " (" + detail + ")",
                )
        decorate(axes[0, col], cfg, "Raw velocity pair angle", "Angle (degrees)")
        groups = [g for g in data["groups"] if g["cfg"] == cfg]
        pos = np.arange(len(groups))
        w = 0.34
        par = [100 * finite_or_nan(middle_mean(g, "correction.parallel_over_base")) for g in groups]
        perp = [
            100 * finite_or_nan(middle_mean(g, "correction.perpendicular_over_base"))
            for g in groups
        ]
        ax = axes[1, col]
        ax.bar(pos - w / 2, par, w, color="#364c61", label="Parallel (signed)")
        ax.bar(pos + w / 2, perp, w, color="#d4984d", label="Perpendicular magnitude")
        ax.set_xticks(pos, [LABELS[g["method"]] for g in groups])
        ax.set_title(f"Correction decomposition, 0.2 <= t <= 0.9 | CFG={cfg}")
        ax.set_ylabel("Relative to ||v_CFG|| (%)")
        ax.grid(axis="y", alpha=0.18)
        ax.axhline(0, color="black", lw=0.7)
        for x, a, b in zip(pos, par, perp):
            for location, value in ((x - w / 2, a), (x + w / 2, b)):
                if np.isfinite(value):
                    ax.text(location, value + 0.1, f"{value:.2f}", ha="center", fontsize=8)
        ax.legend(frameon=False, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Reference velocity pairs and correction directions", fontsize=16, y=0.99)
    fig.text(
        0.5,
        0.015,
        "Different time offsets and state arguments: pair angles are descriptive, not an isolated shift/coefficient ablation.\n"
        "Bottom panels: first average per image, then average uniformly over each method's actual grid in [0.2, 0.9].",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save(fig, output, "velocity_pairs_and_decomposition")


def interior_aggregates(data):
    metrics = [
        "current.cond_vs_uncond.angle_deg",
        "current.cond_vs_uncond.norm_ratio_a_over_b",
        "combined_vs_base.angle_deg",
        "combined_vs_base.norm_ratio_a_over_b",
        "correction_vs_base.norm_ratio_a_over_b",
        "correction.parallel_over_base",
        "correction.perpendicular_over_base",
        "correction_vs_cfg_direction.angle_deg",
        "cfg_direction_vs_base.angle_deg",
        "cfg_correction_vs_base.norm_ratio_a_over_b",
        "cab.theta_vs_base.norm_ratio_a_over_b",
    ]
    return [
        dict(
            name=g["name"],
            cfg=g["cfg"],
            method=g["method"],
            interior_time_range=[0.2, 0.9],
            means={k: middle_mean(g, k) for k in metrics},
        )
        for g in data["groups"]
    ]


def finite_or_nan(value):
    return np.nan if value is None else value


def display(value, digits=2, scale=1, suffix=""):
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * scale:.{digits}f}{suffix}"


def write_report(data, aggregates, output):
    lines = [
        "# 速度几何分析",
        "",
        f"源目录：`{data['run_root']}`。每组样本数：{data.get('num_images_per_group', '未记录')}。",
        "大小使用 CHW 整体 L2 或 RMS；两者的比值与角度一致。下表先对每个时刻的逐样本量取均值，再对 0.2≤t≤0.9 内的实际时间点等权平均。",
        "各方法沿自己的轨迹统计，网格可能不同。平均夹角接近 90° 并不代表每个样本都正交，应结合保存的分布与分位数判断。零向量的夹角未定义，表中以 — 表示。",
        "算法修正定义为组合速度减去当前原始 CFG 速度；CFG 方向定义为条件速度减去无条件速度。",
        "",
        "![完整几何曲线](velocity_geometry_overview.png)",
        "",
        "![条件/无条件大小](basic_branch_magnitudes.png)",
        "",
        "![参照速度及修正分解](velocity_pairs_and_decomposition.png)",
        "",
        "| 方法 | 步数 | CFG | c/u夹角° | c/u大小比 | 修正/原速度 | 组合/原速度 | 组合转角° | 修正与CFG方向夹角° |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, aggregate in zip(data["groups"], aggregates):
        means = aggregate["means"]
        values = [
            LABELS[group["method"]],
            str(len(group["rows"])),
            f"{group['cfg']:g}",
            display(means["current.cond_vs_uncond.angle_deg"], 3),
            display(means["current.cond_vs_uncond.norm_ratio_a_over_b"], 6),
            display(means["correction_vs_base.norm_ratio_a_over_b"], scale=100, suffix="%"),
            display(means["combined_vs_base.norm_ratio_a_over_b"], 4),
            display(means["combined_vs_base.angle_deg"], 3),
            display(means["correction_vs_cfg_direction.angle_deg"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines += [
        "",
        "末端比较选取各组最接近 t=0.98 的实际时间点，并列出该时间。Heun/Heun-flip 的最后一步使用 Euler，因此该步算法修正为零；修正与其他向量的夹角未定义。",
        "",
        "| 方法 | CFG | 实际t | 修正/原速度 | 组合转角° |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in data["groups"]:
        row = min(group["rows"], key=lambda r: abs(r["time"] - 0.98))
        stats = row["stats"]
        correction = stats.get("correction_vs_base.norm_ratio_a_over_b", {}).get("mean")
        angle = stats.get("combined_vs_base.angle_deg", {}).get("mean")
        lines.append(
            f"| {LABELS[group['method']]} | {group['cfg']:g} | {row['time']:.4f} | "
            f"{display(correction, scale=100, suffix='%')} | {display(angle)} |"
        )
    lines += [
        "",
        "末端分母截断、CFG 生效区间和 SG 时间偏移以该次运行配置为准；这些设置可能改变分支比较。速度几何本身不证明生成质量变化的因果关系。",
        "完整逐样本统计在各组 velocity_geometry/batch_*.npz，字段与汇总在 schema.json 和 summary.json。本报告只读取已保存的统计。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def load_curves(path):
    """Accept either a saved compact curve file or a generated run directory."""
    if path.is_file():
        return json.loads(path.read_text())
    plan = json.loads((path / "plan.json").read_text())
    groups = []
    counts = set()
    for group in plan["groups"]:
        summary = json.loads((path / group["name"] / "velocity_geometry/summary.json").read_text())
        counts.add(summary["num_images"])
        groups.append(
            dict(
                group,
                rows=[
                    dict(step=row["step"], time=row["time"], stats=row["metrics"])
                    for row in summary["rows"]
                ],
            )
        )
    if len(counts) != 1:
        raise ValueError("Velocity analysis requires equal nonzero group coverage")
    return dict(run_root=str(path), num_images_per_group=counts.pop(), groups=groups)


def create_report(input_path, output_path):
    data = load_curves(input_path)
    if not data["groups"] or any(not group["rows"] for group in data["groups"]):
        raise ValueError("Velocity analysis requires nonempty per-step summaries")
    if any(group["cfg"] not in (1, 3) or group["method"] not in LABELS for group in data["groups"]):
        raise ValueError("Report supports the configured CFG=1/3 sampling methods")
    output_path.mkdir(parents=True, exist_ok=True)
    plot_overview(data, output_path)
    plot_branches(data, output_path)
    plot_reference_pairs(data, output_path)
    aggregates = interior_aggregates(data)
    for name, value in [("velocity_curves.json", data), ("interior_summary.json", aggregates)]:
        (output_path / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    write_report(data, aggregates, output_path)
    return aggregates
