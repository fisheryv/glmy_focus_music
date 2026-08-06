"""Build figures for the ACE-Step topology-guidance design report.

The plots intentionally separate measured fingerprint evidence from proposed
engineering stages.  They do not simulate or imply generation outcomes.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import json
import os
from pathlib import Path

_ROOT_FOR_MPL = Path(__file__).resolve().parents[1]
_MPL_CONFIG = _ROOT_FOR_MPL / "tmp" / "matplotlib"
_MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "metadata" / "focus_path_homology_fingerprint_v2_summary.json"
OUTPUT_DIR = ROOT / "runs" / "ace_topology_guidance_design" / "figures"

COLORS = {
    "blue": "#2563EB",
    "cyan": "#0891B2",
    "green": "#15803D",
    "orange": "#C2410C",
    "red": "#B91C1C",
    "purple": "#7E22CE",
    "gray": "#475569",
    "light": "#E2E8F0",
    "pale_blue": "#DBEAFE",
    "pale_green": "#DCFCE7",
    "pale_orange": "#FFEDD5",
    "pale_red": "#FEE2E2",
    "pale_purple": "#F3E8FF",
}


def _setup() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT_DIR / f"{name}.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 10,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.4,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#0F172A",
        linespacing=1.35,
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color=COLORS["gray"],
            connectionstyle="arc3,rad=0",
        )
    )


def build_qualification_figure(summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    validation = summary["validation_180"]
    performance = [validation["balanced_accuracy"], validation["roc_auc"]]
    bars = axes[0].bar(
        [0, 1],
        performance,
        color=[COLORS["blue"], COLORS["green"]],
        width=0.56,
    )
    axes[0].set_xticks([0, 1], ["Balanced accuracy", "ROC-AUC"])
    axes[0].set_ylim(0.5, 1.03)
    axes[0].set_ylabel("validation/180 s")
    axes[0].set_title("L+P 纯 Path Homology 指纹", fontsize=12, pad=12)
    axes[0].grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, performance, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    increments = [
        summary["phase_increment"]["delta_pseudo_f"],
        summary["structure_increment"]["delta_pseudo_f"],
    ]
    bars = axes[1].bar(
        [0, 1],
        increments,
        color=[COLORS["green"], COLORS["red"]],
        width=0.56,
    )
    axes[1].axhline(0.0, color=COLORS["gray"], linewidth=0.9)
    axes[1].set_xticks([0, 1], ["加入相位 P", "加入结构 S"])
    axes[1].set_ylabel("Δpseudo-F")
    axes[1].set_title("融合增量决定主指纹组成", fontsize=12, pad=12)
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].set_ylim(-6.6, 8.5)
    axes[1].text(
        0,
        7.15,
        f"{increments[0]:+.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].text(
        1,
        -4.70,
        f"{increments[1]:+.3f}",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].text(0, 7.85, "p=0.001, FDR=0.002", ha="center", color=COLORS["green"])
    axes[1].text(1, -6.05, "p=1.000", ha="center", color=COLORS["red"])

    fig.suptitle("纯 Path Homology v2 的现有实证边界", fontsize=15, y=1.03)
    fig.text(
        0.5,
        -0.02,
        "相位进入主指纹；结构保留为宏观辅助层。分类性能不等于注意力因果效果。",
        ha="center",
        fontsize=10,
        color=COLORS["gray"],
    )
    fig.tight_layout()
    _save(fig, "qualification_evidence")


def build_architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(
        ax,
        (0.02, 0.61),
        0.16,
        0.20,
        "精确 PH 教师\nL+P / 51 维\n离线、不可导",
        COLORS["pale_blue"],
        COLORS["blue"],
    )
    _box(
        ax,
        (0.22, 0.61),
        0.17,
        0.20,
        "轨迹数据集\n解码 x0_hat 快照\n精确端点标签",
        COLORS["pale_purple"],
        COLORS["purple"],
    )
    _box(
        ax,
        (0.43, 0.61),
        0.17,
        0.20,
        "可微 LTSN\ng_phi(x0_hat,t)\nFocus logit + PH 坐标",
        COLORS["pale_green"],
        COLORS["green"],
    )
    _box(
        ax,
        (0.64, 0.61),
        0.16,
        0.20,
        "ACE-Step Turbo\n8 步 Euler\n中段弱校正",
        COLORS["pale_orange"],
        COLORS["orange"],
    )
    _box(
        ax,
        (0.84, 0.61),
        0.14,
        0.20,
        "解码音频\nexact 复核\n质量与盲听",
        COLORS["pale_blue"],
        COLORS["blue"],
    )

    for x1, x2 in [(0.18, 0.22), (0.39, 0.43), (0.60, 0.64), (0.80, 0.84)]:
        _arrow(ax, (x1, 0.71), (x2, 0.71))

    _box(
        ax,
        (0.15, 0.20),
        0.26,
        0.18,
        "硬门 1\nExact reranking\n先证明生成空间可辨识",
        COLORS["pale_red"],
        COLORS["red"],
    )
    _box(
        ax,
        (0.46, 0.20),
        0.21,
        0.18,
        "硬门 2\n代理相关、排序、校准\n均达到冻结阈值",
        COLORS["pale_red"],
        COLORS["red"],
    )
    _box(
        ax,
        (0.72, 0.20),
        0.21,
        0.18,
        "硬门 3\nexact gain 成立\n质量满足非劣界",
        COLORS["pale_red"],
        COLORS["red"],
    )
    _arrow(ax, (0.28, 0.61), (0.28, 0.38))
    _arrow(ax, (0.535, 0.61), (0.565, 0.38))
    _arrow(ax, (0.91, 0.61), (0.825, 0.38))

    ax.text(
        0.5,
        0.06,
        "v2 当前进入 exact scoring / shadow / experimental reranking；采样引导仍须逐门验证。",
        ha="center",
        fontsize=11,
        color=COLORS["gray"],
    )
    ax.set_title("ACE-Step 拓扑引导的证据门控架构", fontsize=16, pad=14)
    fig.tight_layout()
    _save(fig, "guidance_architecture")


def build_schedule_figure() -> None:
    timesteps = np.array([1.0, 0.954545, 0.9, 0.833333, 0.75, 0.642857, 0.5, 0.3])
    weights = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0])
    x = np.arange(1, 9)

    fig, ax1 = plt.subplots(figsize=(11.5, 4.8))
    ax2 = ax1.twinx()
    ax1.plot(x, timesteps, "o-", color=COLORS["blue"], linewidth=2.2, label="timestep")
    ax2.bar(x, weights, color=COLORS["orange"], alpha=0.36, width=0.62, label="引导窗口")
    ax1.set_xticks(x)
    ax1.set_xlabel("Turbo 推理步（8 步，shift=3.0）")
    ax1.set_ylabel("噪声时间 t", color=COLORS["blue"])
    ax2.set_ylabel("相对引导权重", color=COLORS["orange"])
    ax1.set_ylim(0.2, 1.05)
    ax2.set_ylim(0, 1.25)
    ax1.grid(axis="both", alpha=0.2)
    for step, t in zip(x, timesteps, strict=True):
        ax1.text(step, t + 0.027, f"{t:.3f}", ha="center", fontsize=9, color=COLORS["blue"])
    for step, weight in zip(x, weights, strict=True):
        if weight > 0:
            ax2.text(
                step,
                weight + 0.045,
                f"{weight:.1f}",
                ha="center",
                fontsize=9,
                color=COLORS["orange"],
            )
    ax1.axvspan(3.5, 6.5, color=COLORS["pale_orange"], alpha=0.33, zorder=-2)
    ax1.text(
        4.95,
        0.28,
        "建议仅在第 4–6 步启用\nRMS clip：0.25% / 0.5% / 1.0% 候选档",
        ha="center",
        va="bottom",
        fontsize=10,
        color=COLORS["gray"],
    )
    ax1.set_title("建议的中段三角形弱引导窗口（工程起点，尚未验证）", fontsize=14, pad=12)
    fig.tight_layout()
    _save(fig, "sampler_guidance_schedule")


def build_stage_gates_figure() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        (
            0.04,
            0.69,
            "0  Shadow",
            "记录轨迹与 exact 标签\n不改变生成",
            COLORS["pale_blue"],
            COLORS["blue"],
        ),
        (
            0.23,
            0.69,
            "1  Exact 重排",
            "同 prompt 候选池\n证明可辨识性",
            COLORS["pale_purple"],
            COLORS["purple"],
        ),
        (
            0.42,
            0.69,
            "2  LTSN",
            "未见轨迹上通过\n相关/排序/校准",
            COLORS["pale_green"],
            COLORS["green"],
        ),
        (
            0.61,
            0.69,
            "3  采样开发",
            "冻结步骤窗口\n只选择一次强度",
            COLORS["pale_orange"],
            COLORS["orange"],
        ),
        (
            0.80,
            0.69,
            "4  最终确认",
            "配对 A/B + exact 复核\n质量非劣",
            COLORS["pale_red"],
            COLORS["red"],
        ),
    ]
    for x0, y0, title, body, face, edge in stages:
        _box(ax, (x0, y0), 0.16, 0.20, f"{title}\n{body}", face, edge, fontsize=9.5)
    for x1, x2 in [(0.20, 0.23), (0.39, 0.42), (0.58, 0.61), (0.77, 0.80)]:
        _arrow(ax, (x1, 0.79), (x2, 0.79))

    gate_texts = [
        (0.31, "≥10% exact gain + 非劣"),
        (0.50, "代理分数与 exact 一致"),
        (0.69, "步骤/强度冻结"),
        (0.88, "盲听、质量、失败率全通过"),
    ]
    for x0, text in gate_texts:
        ax.plot([x0, x0], [0.60, 0.65], color=COLORS["red"], linewidth=3)
        ax.text(x0, 0.56, text, ha="center", va="top", fontsize=9, color=COLORS["red"])

    ax.add_patch(
        FancyArrowPatch(
            (0.88, 0.48),
            (0.13, 0.24),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            color=COLORS["gray"],
            connectionstyle="arc3,rad=0.18",
        )
    )
    _box(
        ax,
        (0.04, 0.12),
        0.30,
        0.15,
        "任一门槛失败：停止升级\n回退 shadow / exact reranking\n不得用失败数据继续调主指标",
        COLORS["pale_red"],
        COLORS["red"],
        fontsize=10,
    )
    ax.text(
        0.64,
        0.19,
        "已有 validation 与 holdout 不得用于\n继续调采样强度或代理损失。",
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["gray"],
    )
    ax.set_title("从冻结指纹到采样引导的升级门槛", fontsize=16, pad=16)
    fig.tight_layout()
    _save(fig, "stage_gates")


def main() -> None:
    _setup()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    build_qualification_figure(summary)
    build_architecture_figure()
    build_schedule_figure()
    build_stage_gates_figure()


if __name__ == "__main__":
    main()
