"""Render paper-ready LTCH-P3 architecture and LTSN comparison figures."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG = ROOT / "tmp" / "matplotlib"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUTPUT = ROOT / "papers" / "fig"
INK = "#172033"
MUTED = "#526075"
BLUE = (0.18, 0.43, 0.82)
GREEN = (0.10, 0.55, 0.37)
ORANGE = (0.89, 0.45, 0.12)
PURPLE = (0.48, 0.30, 0.76)
RED = (0.76, 0.20, 0.25)


def _pale(color: tuple[float, float, float], amount: float = 0.84) -> tuple[float, ...]:
    return tuple(channel * (1.0 - amount) + amount for channel in color)


def _setup() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "SimHei",
                "Arial",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "figure.dpi": 160,
            "savefig.dpi": 240,
        }
    )


def _box(
    ax: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    color: tuple[float, float, float],
    *,
    fontsize: float = 9.0,
    linewidth: float = 1.45,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.08",
            linewidth=linewidth,
            facecolor=_pale(color),
            edgecolor=color,
            zorder=2,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.28,
        zorder=3,
    )


def _arrow(
    ax: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str | tuple[float, float, float] = MUTED,
    dashed: bool = False,
    rad: float = 0.0,
    linewidth: float = 1.4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=linewidth,
            linestyle="--" if dashed else "-",
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=5,
        )
    )


def _save(figure: Any, stem: str) -> None:
    for suffix in ("png", "svg"):
        figure.savefig(
            OUTPUT / f"{stem}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def build_ltch_p3_architecture() -> None:
    figure, ax = plt.subplots(figsize=(19.5, 10.8))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10.6)
    ax.axis("off")

    ax.text(
        10,
        10.16,
        "LTCH-P3：三维局部 Pitch 拓扑的轻量潜空间控制头",
        ha="center",
        va="center",
        fontsize=18,
        color=INK,
    )
    ax.text(
        10,
        9.78,
        "全局序列到全局控制：输入为每个采样步自己的预测干净潜变量，输出不是逐帧曲线",
        ha="center",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )

    trunk_y = 6.9
    trunk_h = 1.65
    stages = [
        (
            0.25,
            2.20,
            "预测干净潜变量\n$\\hat{x}_{0|t}=x_t-t v_t$\n"
            "$[B,T,64]$\n180 s：$T\\approx4500$",
            BLUE,
        ),
        (
            2.78,
            2.18,
            "逐帧输入映射\nLayerNorm(64)\nLinear $64\\to128$\n"
            "$[B,T,128]$\n8,448 params",
            BLUE,
        ),
        (
            5.29,
            2.17,
            "时间下采样\nDepthwise Conv1d\n$k=9,s=4,p=4$\n"
            "$[B,\\lceil T/4\\rceil,128]$\n1,280 params",
            GREEN,
        ),
        (
            7.80,
            2.18,
            "条件调制 + 位置\n$h\\odot(1+\\gamma)+\\beta$\n"
            "固定正弦位置编码\n$[B,T',128]$\n33,024 params",
            ORANGE,
        ),
        (
            10.31,
            2.65,
            "全局 Transformer Encoder ×3\nPre-Norm；$d=128$；4 heads\n"
            "FFN $128\\to512\\to128$；GELU\nDropout 0.1；padding mask\n"
            "$[B,T',128]$；594,816 params",
            PURPLE,
        ),
        (
            13.29,
            2.38,
            "掩码全局统计汇聚\nAttention pooling [128]\n"
            "Mean [128] + Std [128]\n$[B,384]$\n含 output LN：385 params",
            GREEN,
        ),
        (
            16.00,
            3.55,
            "共享融合 MLP\nConcat：pool 384 + condition 128\n$[B,512]$\n"
            "Linear $512\\to256$ + SiLU + Dropout\n"
            "Linear $256\\to128$ + SiLU\n$[B,128]$；164,224 params",
            RED,
        ),
    ]
    for x, width, text, color in stages:
        _box(ax, x, trunk_y, width, trunk_h, text, color, fontsize=8.55)
    for left, right in zip(stages[:-1], stages[1:], strict=True):
        _arrow(
            ax,
            (left[0] + left[1], trunk_y + trunk_h / 2),
            (right[0], trunk_y + trunk_h / 2),
        )

    _box(
        ax,
        2.05,
        3.78,
        3.05,
        1.52,
        "采样条件\n连续噪声时间 $t$\n离散 step id / 8",
        ORANGE,
        fontsize=9.1,
    )
    _box(
        ax,
        5.63,
        3.78,
        3.30,
        1.52,
        "FourierStepEmbedding\n32 个对数频率；sin/cos\n"
        "输入 128 → MLP $128\\to256\\to128$\n"
        "$e_t\\in\\mathbb{R}^{128}$；65,920 params",
        ORANGE,
        fontsize=8.65,
    )
    _arrow(ax, (5.10, 4.54), (5.63, 4.54), color=ORANGE)
    _arrow(
        ax,
        (8.15, 5.30),
        (8.88, trunk_y),
        color=ORANGE,
        dashed=True,
        rad=-0.08,
    )
    _arrow(
        ax,
        (8.93, 4.54),
        (17.73, trunk_y),
        color=ORANGE,
        dashed=True,
        rad=-0.20,
    )

    head_y = 1.18
    _box(
        ax,
        5.05,
        head_y,
        2.70,
        1.45,
        "坐标均值头\nLinear $128\\to3$\n$\\hat\\mu_z\\in\\mathbb{R}^3$\n387 params",
        GREEN,
        fontsize=9.0,
    )
    _box(
        ax,
        8.20,
        head_y,
        2.70,
        1.45,
        "异方差头\nLinear $128\\to3$\n$\\log\\hat\\sigma_z^2\\in[-8,4]^3$\n387 params",
        ORANGE,
        fontsize=9.0,
    )
    _box(
        ax,
        11.35,
        head_y,
        2.35,
        1.45,
        "OOD 安全头\nLinear $128\\to1$\n$u_{OOD}$\n129 params",
        PURPLE,
        fontsize=9.0,
    )
    _box(
        ax,
        15.05,
        head_y,
        4.05,
        1.45,
        "冻结 Focus 读出（无独立可训练头）\n"
        "$\\hat S_F=w^T\\hat\\mu_z+b$\n"
        "$w=(-5.882,\\;4.216,\\;-0.514)$；$b=0.743$",
        RED,
        fontsize=8.85,
    )
    shared = (17.77, trunk_y)
    for target in ((6.40, head_y + 1.45), (9.55, head_y + 1.45), (12.53, head_y + 1.45)):
        _arrow(ax, shared, target, rad=(target[0] - shared[0]) * 0.018)
    ax.plot(
        [6.40, 6.40, 17.08],
        [head_y, 0.86, 0.86],
        color=RED,
        linewidth=1.4,
        zorder=4,
    )
    _arrow(ax, (17.08, 0.86), (17.08, head_y), color=RED)

    ax.text(
        0.28,
        0.35,
        "总计：869,000 个可训练参数。实线为数据流；橙色虚线为时间条件。"
        "Attention/Mean/Std 均忽略 padding；Focus 只由三维均值确定。",
        fontsize=10.0,
        color=MUTED,
        ha="left",
    )
    figure.tight_layout()
    _save(figure, "ltch_p3_network_architecture")


def _vertical_pipeline(
    ax: Any,
    *,
    center: float,
    title: str,
    subtitle: str,
    stages: Sequence[tuple[str, tuple[float, float, float]]],
    footer: str,
) -> None:
    width = 6.0
    x = center - width / 2
    ax.text(center, 12.85, title, ha="center", va="center", fontsize=16, color=INK)
    ax.text(center, 12.45, subtitle, ha="center", va="center", fontsize=9.8, color=MUTED)
    height = 1.00
    gap = 0.24
    top = 11.05
    for index, (label, color) in enumerate(stages):
        y = top - index * (height + gap)
        _box(ax, x, y, width, height, label, color, fontsize=8.85)
        if index:
            previous_y = top - (index - 1) * (height + gap)
            _arrow(ax, (center, previous_y), (center, y + height), linewidth=1.2)
    final_y = top - (len(stages) - 1) * (height + gap)
    ax.text(
        center,
        final_y - 0.48,
        footer,
        ha="center",
        va="center",
        fontsize=10.2,
        color=INK,
    )


def build_ltsn_comparison() -> None:
    figure, ax = plt.subplots(figsize=(16.8, 12.4))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 13.6)
    ax.axis("off")

    old_stages = [
        ("$\\hat{x}_{0|t}$  [B,T,64] + time/step condition 128", BLUE),
        ("Stem：Conv1d 64→128，k=9，stride=5", BLUE),
        ("Pitch-local TCN ×4：192 channels；dilation 1/2/4/8", GREEN),
        ("Long-range phase TCN ×4：256 channels；stride=4；dilation 1/4/16/64", ORANGE),
        ("Conditioned Transformer ×2：d=256，8 heads，FFN=1024", PURPLE),
        ("双分支 pool：local 576 + global 768 + condition 128 = 1472", GREEN),
        ("Fusion 1472→512→256；mean 18 + logvar 18 + OOD 1", RED),
    ]
    new_stages = [
        ("$\\hat{x}_{0|t}$  [B,T,64] + time/step condition 128", BLUE),
        ("LayerNorm + Linear 64→128", BLUE),
        ("Depthwise Conv1d：128 channels，k=9，stride=4", GREEN),
        ("单次 FiLM + 固定正弦位置编码", ORANGE),
        ("标准 Transformer Encoder ×3：d=128，4 heads，FFN=512", PURPLE),
        ("单分支 pool：attention 128 + mean 128 + std 128 + condition 128 = 512", GREEN),
        ("Fusion 512→256→128；mean 3 + logvar 3 + OOD 1", RED),
    ]
    _vertical_pipeline(
        ax,
        center=4.05,
        title="旧 LTSN / PathHomologySurrogate",
        subtitle="Pitch 16-D + 双 phase 2-D；多尺度、双分支时序建模",
        stages=old_stages,
        footer="4,132,903 params｜共享表征 256-D｜输出 18-D",
    )
    _vertical_pipeline(
        ax,
        center=12.95,
        title="LTCH-P3 / LatentTopologyControlHead",
        subtitle="仅保留局部 Pitch-3；单尺度、单分支全局建模",
        stages=new_stages,
        footer="869,000 params｜共享表征 128-D｜输出 3-D",
    )

    ax.add_patch(
        FancyBboxPatch(
            (7.55, 3.20),
            1.90,
            6.60,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=_pale(RED, 0.90),
            edgecolor=RED,
            linewidth=1.4,
            zorder=1,
        )
    )
    changes = [
        "移除\nphase 目标",
        "移除\n8 个 TCN 块",
        "双 pool\n→ 单 pool",
        "1472-D\n→ 512-D",
        "18-D\n→ 3-D",
    ]
    for index, label in enumerate(changes):
        ax.text(
            8.50,
            9.15 - index * 1.25,
            label,
            ha="center",
            va="center",
            fontsize=9.2,
            color=RED,
            linespacing=1.18,
        )

    ax.text(
        8.5,
        1.05,
        "参数减少 78.97%（约 4.76× 更小）",
        ha="center",
        va="center",
        fontsize=13,
        color=RED,
    )
    ax.text(
        8.5,
        0.53,
        "代价：不再显式建模跨周期 phase 长程结构；收益：目标与容量更匹配，梯度路径更短。",
        ha="center",
        va="center",
        fontsize=9.8,
        color=MUTED,
    )
    figure.tight_layout()
    _save(figure, "ltch_p3_vs_ltsn_architecture")


def main() -> None:
    _setup()
    build_ltch_p3_architecture()
    build_ltsn_comparison()


if __name__ == "__main__":
    main()
