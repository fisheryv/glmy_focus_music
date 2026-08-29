"""Render architecture and training figures for the LTSN design report."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG = ROOT / "tmp" / "matplotlib"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUTPUT = ROOT / "runs" / "ltsn_design" / "figures"
COLORS = {
    "blue": "#2563EB",
    "green": "#15803D",
    "orange": "#C2410C",
    "purple": "#7E22CE",
    "red": "#B91C1C",
    "gray": "#475569",
    "pale_blue": "#DBEAFE",
    "pale_green": "#DCFCE7",
    "pale_orange": "#FFEDD5",
    "pale_purple": "#F3E8FF",
    "pale_red": "#FEE2E2",
    "pale_gray": "#F1F5F9",
}


def _setup() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "savefig.dpi": 180,
        }
    )


def _save(figure: Any, stem: str) -> None:
    for suffix in ("png", "svg"):
        figure.savefig(
            OUTPUT / f"{stem}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def _box(
    ax: Any,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    face: str,
    edge: str,
    fontsize: float = 9.5,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.022",
            linewidth=1.5,
            facecolor=face,
            edgecolor=edge,
        )
    )
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


def _arrow(
    ax: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str | None = None,
    dashed: bool = False,
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            linestyle="--" if dashed else "-",
            color=color or COLORS["gray"],
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def build_architecture() -> None:
    figure, ax = plt.subplots(figsize=(17.2, 9.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Per-step predicted-clean latent.
    _box(
        ax,
        (0.015, 0.77),
        0.095,
        0.085,
        "噪声潜变量  x(t)\n[B,T,64]",
        COLORS["pale_blue"],
        COLORS["blue"],
        fontsize=9.0,
    )
    _box(
        ax,
        (0.015, 0.64),
        0.095,
        0.085,
        "速度预测  v(t)\n[B,T,64]",
        COLORS["pale_blue"],
        COLORS["blue"],
        fontsize=9.0,
    )
    _box(
        ax,
        (0.135, 0.665),
        0.145,
        0.165,
        "预测干净潜变量\n"
        "xhat(0) = x(t) - t v(t)\n"
        "[B,T,64]\nT≈4500（180 s）",
        COLORS["pale_blue"],
        COLORS["blue"],
    )
    _arrow(ax, (0.11, 0.812), (0.135, 0.775))
    _arrow(ax, (0.11, 0.682), (0.135, 0.715))

    # Main temporal encoder.
    trunk = [
        (
            0.305,
            0.665,
            0.135,
            0.165,
            "Stem\nLayerNorm\nConv1d(64→128,k=9,s=5)\n"
            "GroupNorm + SiLU + Dropout\n[B,T/5,128]；约 0.36 s",
            "pale_blue",
            "blue",
        ),
        (
            0.465,
            0.665,
            0.145,
            0.165,
            "Pitch-local TCN\n1×1 投影至 192 通道\n"
            "4×深度可分离残差块\nk=3；dilation=1,2,4,8\n[B,T/5,192]",
            "pale_green",
            "green",
        ),
        (
            0.635,
            0.665,
            0.15,
            0.165,
            "Long-range phase TCN\nConv1d(192→256,k=7,s=4)\n"
            "4×残差块\ndilation=1,4,16,64\n[B,T/20,256]",
            "pale_orange",
            "orange",
        ),
        (
            0.81,
            0.665,
            0.175,
            0.165,
            "全局编码器\n2×Pre-Norm Transformer\n"
            "d=256；8 heads；FFN=1024\n[B,T/20,256]\n远距离位置连接（非 Structure 分支）",
            "pale_purple",
            "purple",
        ),
    ]
    for x, y, w, h, label, face, edge in trunk:
        _box(ax, (x, y), w, h, label, COLORS[face], COLORS[edge], fontsize=8.9)

    _arrow(ax, (0.28, 0.747), (0.305, 0.747))
    for left, right in zip(trunk[:-1], trunk[1:], strict=True):
        _arrow(
            ax,
            (left[0] + left[2], left[1] + left[3] / 2),
            (right[0], right[1] + right[3] / 2),
        )

    # Timestep embedding and FiLM/AdaLN modulation.
    _box(
        ax,
        (0.055, 0.375),
        0.22,
        0.16,
        "时间条件\n连续时间 t + 离散 step id\n"
        "Fourier 特征 → 两层 MLP\n时间嵌入 e(t)：128 维",
        COLORS["pale_gray"],
        COLORS["gray"],
    )
    ax.text(
        0.285,
        0.555,
        "FiLM / AdaLN 调制",
        fontsize=9.2,
        color=COLORS["gray"],
        ha="center",
    )
    for target_x, rad in ((0.535, 0.14), (0.71, 0.05), (0.895, -0.05)):
        _arrow(
            ax,
            (0.275, 0.49),
            (target_x, 0.665),
            color=COLORS["gray"],
            dashed=True,
            rad=rad,
        )

    # Local/global pooling and feature fusion.
    _box(
        ax,
        (0.39, 0.38),
        0.17,
        0.145,
        "局部序列汇聚\nMasked Attention + Mean + Std\n"
        "3 × 192 = 576 维",
        COLORS["pale_green"],
        COLORS["green"],
    )
    _box(
        ax,
        (0.59, 0.38),
        0.17,
        0.145,
        "全局序列汇聚\nMasked Attention + Mean + Std\n"
        "3 × 256 = 768 维",
        COLORS["pale_purple"],
        COLORS["purple"],
    )
    _box(
        ax,
        (0.795, 0.38),
        0.19,
        0.145,
        "融合层\n576 + 768 + 时间嵌入 128 = 1472 维\n"
        "MLP 1472→512→256\n共享表征 [B,256]",
        COLORS["pale_red"],
        COLORS["red"],
    )
    _arrow(ax, (0.535, 0.665), (0.475, 0.525), rad=0.12)
    _arrow(ax, (0.898, 0.665), (0.675, 0.525), rad=0.16)
    _arrow(ax, (0.56, 0.452), (0.795, 0.47), rad=-0.05)
    _arrow(ax, (0.76, 0.452), (0.795, 0.435), rad=0.05)
    _arrow(
        ax,
        (0.275, 0.43),
        (0.795, 0.405),
        color=COLORS["gray"],
        dashed=True,
        rad=0.02,
    )

    # Three prediction heads and frozen Focus readout.
    _box(
        ax,
        (0.39, 0.11),
        0.145,
        0.14,
        "坐标均值头\n均值向量：18 维\n"
        "Pitch 16 + Acoustic phase 1\n+ Chroma phase 1",
        COLORS["pale_green"],
        COLORS["green"],
        fontsize=9.0,
    )
    _box(
        ax,
        (0.56, 0.11),
        0.135,
        0.14,
        "不确定度头\nlog-variance：18 维\n异方差不确定度",
        COLORS["pale_orange"],
        COLORS["orange"],
        fontsize=9.0,
    )
    _box(
        ax,
        (0.72, 0.11),
        0.105,
        0.14,
        "分布外检测\nOOD logit\n1 维",
        COLORS["pale_gray"],
        COLORS["gray"],
        fontsize=9.0,
    )
    _box(
        ax,
        (0.855, 0.11),
        0.13,
        0.14,
        "冻结线性读出\nFocus logit = w 转置 × 均值 + b",
        COLORS["pale_red"],
        COLORS["red"],
        fontsize=9.2,
    )
    for target_x in (0.462, 0.627, 0.772):
        _arrow(ax, (0.89, 0.38), (target_x, 0.25), rad=(target_x - 0.7) * 0.25)
    ax.plot(
        [0.462, 0.462, 0.92],
        [0.25, 0.29, 0.29],
        color=COLORS["gray"],
        linewidth=1.4,
    )
    _arrow(ax, (0.92, 0.29), (0.92, 0.25))
    ax.text(
        0.69,
        0.30,
        "仅使用坐标均值",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color=COLORS["gray"],
    )

    ax.text(
        0.02,
        0.035,
        "实线：特征数据流；虚线：时间条件调制。全局注意力只连接远距离 Acoustic/Chroma phase 位置，"
        "不构成额外的 Structure 分支。",
        fontsize=10.0,
        color=COLORS["gray"],
    )
    ax.set_title(
        "潜空间拓扑代理网络（LTSN）：从逐步预测干净潜变量到冻结 18-D 拓扑指纹",
        fontsize=16,
        pad=15,
    )
    figure.tight_layout()
    _save(figure, "ltsn_network_architecture")


def build_architecture_reference_style() -> None:
    """Render a paper diagram using the classic pastel Transformer style."""

    figure, ax = plt.subplots(figsize=(13.2, 14.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14.2)
    ax.axis("off")

    ink = "#111111"
    blue = "#CFE2F3"
    green = "#D9EAD3"
    yellow = "#FFF2CC"
    peach = "#FCE5CD"
    white = "#FFFFFF"

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        label: str,
        fill: str,
        *,
        fontsize: float = 9.3,
        linewidth: float = 1.25,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.018,rounding_size=0.08",
                linewidth=linewidth,
                facecolor=fill,
                edgecolor=ink,
                zorder=2,
            )
        )
        ax.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=ink,
            linespacing=1.28,
            zorder=3,
        )

    def frame(
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=1.45,
                facecolor=white,
                edgecolor=ink,
                zorder=0,
            )
        )
        ax.text(
            x + width / 2,
            y + height - 0.22,
            title,
            ha="center",
            va="center",
            fontsize=11.3,
            color=ink,
            fontweight="normal",
            zorder=4,
        )

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        dashed: bool = False,
        rad: float = 0.0,
        linewidth: float = 1.25,
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=linewidth,
                linestyle="--" if dashed else "-",
                color=ink,
                connectionstyle=f"arc3,rad={rad}",
                zorder=5,
            )
        )

    def residual_pair(
        x: float,
        y: float,
        width: float,
        blue_label: str,
        green_label: str,
    ) -> None:
        box(x, y, width, 0.69, blue_label, blue, fontsize=8.8)
        box(x, y + 0.84, width, 0.46, green_label, green, fontsize=8.7)
        arrow((x + width / 2, y + 0.69), (x + width / 2, y + 0.84))
        arrow(
            (x + 0.15, y + 0.06),
            (x + 0.15, y + 1.08),
            rad=-0.38,
            linewidth=1.05,
        )

    ax.text(
        7,
        13.82,
        "潜空间拓扑代理网络（LTSN）",
        ha="center",
        va="center",
        fontsize=18,
        color=ink,
    )
    ax.text(
        7,
        13.46,
        "逐步预测干净潜变量 → 冻结 18-D 拓扑指纹",
        ha="center",
        va="center",
        fontsize=10.8,
        color=ink,
    )

    # Two towers reproduce the reference image's stacked-network composition.
    frame(0.55, 2.35, 6.15, 10.45, "多尺度时序编码器")
    frame(7.25, 2.35, 6.15, 10.45, "多尺度汇聚与输出")

    # Input construction and the first convolutional reduction.
    box(
        1.25,
        0.55,
        4.75,
        0.82,
        "每个采样步：x(t), v(t)\n"
        "预测干净潜变量  xhat(0) = x(t) - t v(t)\n[B,T,64]；T≈4500（180 s）",
        yellow,
        fontsize=9.2,
    )
    arrow((3.625, 1.37), (3.625, 2.62))
    box(
        1.28,
        2.62,
        4.7,
        0.83,
        "Stem\nLayerNorm；Conv1d(64→128,k=9,s=5)\n"
        "GroupNorm + SiLU + Dropout 0.05\n[B,T/5,128]",
        blue,
        fontsize=8.8,
    )

    # Pitch-local TCN: a residual block stack, represented like the reference.
    arrow((3.625, 3.45), (3.625, 3.73))
    ax.text(3.625, 4.98, "Pitch-local TCN ×4", ha="center", fontsize=10.2)
    residual_pair(
        1.42,
        3.73,
        4.42,
        "1×1 投影至 192 通道\nDepthwise-separable Conv1d\nk=3；dilation=1,2,4,8",
        "Residual + FiLM/AdaLN\n输出 [B,T/5,192]",
    )

    # Long-range TCN.
    arrow((3.625, 5.03), (3.625, 5.34))
    ax.text(3.625, 6.60, "Long-range phase TCN ×4", ha="center", fontsize=10.2)
    residual_pair(
        1.42,
        5.34,
        4.42,
        "Conv1d(192→256,k=7,s=4)\nResidual Conv blocks\ndilation=1,4,16,64",
        "Residual + FiLM/AdaLN\n输出 [B,T/20,256]",
    )

    # Global Transformer, with the reference image's blue/green residual stack.
    arrow((3.625, 6.64), (3.625, 7.02))
    ax.text(3.625, 9.48, "全局编码器 ×2", ha="center", fontsize=10.2)
    box(
        1.18,
        7.02,
        4.9,
        2.22,
        "",
        white,
        linewidth=1.1,
    )
    residual_pair(
        1.52,
        7.27,
        4.22,
        "8-head Self-Attention + FFN\nd=256；FFN=1024",
        "Pre-Norm + Residual\n[B,T/20,256]",
    )
    ax.text(
        3.625,
        9.72,
        "连接远距离 Acoustic/Chroma phase 位置\n不构成额外 Structure 分支",
        ha="center",
        va="center",
        fontsize=8.9,
        color=ink,
    )
    arrow((3.625, 9.24), (3.625, 9.50))
    box(
        1.40,
        10.22,
        4.45,
        0.65,
        "全局时序表征  [B,T/20,256]",
        green,
        fontsize=9.0,
    )
    arrow((3.625, 9.88), (3.625, 10.22))

    # Time embedding mirrors the positional-encoding cue in the reference.
    box(
        7.82,
        0.55,
        4.85,
        0.82,
        "时间条件\n连续时间 t + 离散 step id → Fourier 特征 → 两层 MLP\n"
        "时间嵌入：128 维",
        yellow,
        fontsize=9.0,
    )
    ax.text(7.45, 1.02, "≈", fontsize=22, ha="center", va="center", color=ink)
    ax.text(7.45, 0.72, "time", fontsize=8.0, ha="center", color=ink)

    # Separate local and global pooling branches.
    box(
        7.68,
        3.18,
        2.45,
        1.02,
        "局部序列汇聚\nMasked Attention\n+ Mean + Std\n576 维",
        blue,
        fontsize=8.8,
    )
    box(
        10.52,
        3.18,
        2.45,
        1.02,
        "全局序列汇聚\nMasked Attention\n+ Mean + Std\n768 维",
        blue,
        fontsize=8.8,
    )
    arrow((5.84, 4.73), (7.68, 3.72), rad=0.12)
    ax.text(6.78, 4.30, "local tap", fontsize=7.9, ha="center", color=ink)
    arrow((5.85, 10.55), (10.52, 4.02), rad=-0.12)
    ax.text(8.68, 7.72, "global tap", fontsize=7.9, ha="center", color=ink)

    # Concatenation and fusion MLP.
    arrow((8.90, 4.20), (9.55, 4.67), rad=-0.10)
    arrow((11.74, 4.20), (11.10, 4.67), rad=0.10)
    box(
        8.35,
        4.67,
        3.95,
        0.58,
        "Concat：576 + 768 + time 128 = 1472 维",
        green,
        fontsize=8.9,
    )
    arrow((10.325, 5.25), (10.325, 5.58))
    box(
        8.35,
        5.58,
        3.95,
        0.82,
        "融合 MLP\n1472 → 512 → 256\n共享表征 [B,256]",
        blue,
        fontsize=9.1,
    )
    arrow((12.10, 1.37), (11.72, 4.67), dashed=True, rad=0.10)
    for target_y in (4.40, 5.98, 6.54):
        arrow((8.05, 1.37), (6.02, target_y), dashed=True, rad=0.12)

    # Three parallel heads.
    arrow((10.325, 6.40), (10.325, 6.76))
    box(7.60, 6.76, 1.75, 1.10, "坐标均值\n18 维\n16 Pitch + 2 phase", peach, fontsize=8.4)
    box(9.45, 6.76, 1.75, 1.10, "Log-variance\n18 维\n异方差不确定度", peach, fontsize=8.4)
    box(11.30, 6.76, 1.45, 1.10, "OOD logit\n1 维", peach, fontsize=8.5)
    arrow((10.325, 6.40), (8.48, 6.76), rad=0.14)
    arrow((10.325, 6.40), (10.325, 6.76))
    arrow((10.325, 6.40), (12.02, 6.76), rad=-0.14)

    # Frozen readout consumes only the coordinate mean.
    arrow((8.48, 7.86), (10.325, 8.40), rad=-0.12)
    box(
        8.35,
        8.40,
        3.95,
        0.78,
        "冻结线性读出（仅坐标均值）\nFocus logit = w 转置 × 均值 + b",
        yellow,
        fontsize=9.1,
    )
    arrow((10.325, 9.18), (10.325, 9.56))
    box(
        8.72,
        9.56,
        3.20,
        0.62,
        "18-D 拓扑代理输出",
        green,
        fontsize=10.0,
    )
    arrow((10.325, 10.18), (10.325, 10.63))

    # Legend and output arrow, matching the sparse reference treatment.
    arrow((10.325, 10.63), (10.325, 11.20))
    ax.text(10.325, 11.42, "Focus logit / uncertainty / OOD", ha="center", fontsize=9.3)
    ax.text(
        7,
        0.18,
        "浅蓝：时序计算模块    浅绿：残差/归一化/汇聚    浅黄：条件输入与冻结读出    虚线：FiLM/AdaLN 时间调制",
        ha="center",
        va="center",
        fontsize=9.0,
        color=ink,
    )

    figure.tight_layout()
    _save(figure, "ltsn_network_architecture_reference_style")


def build_training_pipeline() -> None:
    figure, ax = plt.subplots(figsize=(15.5, 7.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        (
            0.015,
            "门 0\nExact reranking\n先证明生成空间可辨识",
            "pale_red",
            "red",
        ),
        (
            0.178,
            "采集无引导轨迹\n多 prompt × 多 seed\n保存 step 4/5/6/8",
            "pale_blue",
            "blue",
        ),
        (
            0.341,
            "逐快照教师标注\n解码各自 x0_hat\nExact PH→18 维坐标",
            "pale_purple",
            "purple",
        ),
        (
            0.504,
            "按 prompt 分组切分\ntrain / dev / calibration\nqualification 完全隔离",
            "pale_green",
            "green",
        ),
        (
            0.667,
            "训练与校准\n多任务损失 + ensemble\n冻结权重/阈值/hash",
            "pale_orange",
            "orange",
        ),
        (
            0.83,
            "资格检验\n相关/排序/覆盖/OOD\n代理优化后 exact 复核",
            "pale_red",
            "red",
        ),
    ]
    stage_width = 0.145
    for x, label, face, edge in stages:
        _box(
            ax,
            (x, 0.66),
            stage_width,
            0.20,
            label,
            COLORS[face],
            COLORS[edge],
            fontsize=9.2,
        )
    for left, right in zip(stages[:-1], stages[1:], strict=True):
        _arrow(ax, (left[0] + stage_width, 0.76), (right[0], 0.76))

    _box(
        ax,
        (0.18, 0.31),
        0.20,
        0.19,
        "训练样本\nDiscovery 真实音频 VAE 重建\n"
        "+ 新 development prompts 生成轨迹\n+ 10–15% 安全/OOD 负例",
        COLORS["pale_blue"],
        COLORS["blue"],
    )
    _box(
        ax,
        (0.41, 0.31),
        0.20,
        0.19,
        "标签原则\n每个 step 的 x0_hat 单独解码\n"
        "不得复制最终音频标签\n失败标签进入 OOD，不静默填补",
        COLORS["pale_purple"],
        COLORS["purple"],
    )
    _box(
        ax,
        (0.64, 0.31),
        0.20,
        0.19,
        "冻结门槛\nlogit ρ≥0.70\n坐标中位 ρ≥0.50\n块距离 ρ≥0.50\n90% PI 覆盖 0.85–0.95",
        COLORS["pale_green"],
        COLORS["green"],
    )
    _arrow(ax, (0.26, 0.66), (0.28, 0.50))
    _arrow(ax, (0.43, 0.66), (0.51, 0.50))
    _arrow(ax, (0.94, 0.66), (0.74, 0.50))

    _box(
        ax,
        (0.33, 0.05),
        0.34,
        0.15,
        "任一资格门失败：停止升级\n保留 shadow / exact reranking\n"
        "不得用 qualification 结果反复改损失",
        COLORS["pale_red"],
        COLORS["red"],
        fontsize=10,
    )
    ax.text(
        0.5,
        0.94,
        "训练目标是逼近 exact Path Homology 教师，不是直接证明注意力、功能性或生成质量。",
        ha="center",
        fontsize=11,
        color=COLORS["gray"],
    )
    ax.set_title("LTSN 数据构建、训练、校准与冻结流程", fontsize=16, pad=16)
    figure.tight_layout()
    _save(figure, "ltsn_training_pipeline")


def main() -> None:
    _setup()
    build_architecture()
    build_training_pipeline()


if __name__ == "__main__":
    main()
