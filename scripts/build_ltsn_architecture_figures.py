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
    figure, ax = plt.subplots(figsize=(16, 7.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (
            0.015,
            0.63,
            0.12,
            0.20,
            "输入 x0_hat\n[B,T,64]\nT≈4500 @ 180 s",
            "pale_blue",
            "blue",
        ),
        (
            0.16,
            0.63,
            0.13,
            0.20,
            "LayerNorm + Stem\nConv1d 64→128\nk=9, stride=5\nT→T/5≈900",
            "pale_blue",
            "blue",
        ),
        (
            0.315,
            0.63,
            0.14,
            0.20,
            "局部 TCN\n通道 192，4 blocks\ndilation 1/2/4/8\n时间粒度≈0.2 s",
            "pale_green",
            "green",
        ),
        (
            0.48,
            0.63,
            0.14,
            0.20,
            "中尺度 TCN\nstride=4，通道 256\nT→T/20≈225\ndilation 1/4/16/64",
            "pale_orange",
            "orange",
        ),
        (
            0.645,
            0.63,
            0.13,
            0.20,
            "全局编码器\nTransformer ×2\n8 heads / FFN 1024\n全长 225 tokens",
            "pale_purple",
            "purple",
        ),
        (
            0.80,
            0.63,
            0.18,
            0.20,
            "多尺度汇聚与融合\nattention + mean + std\n576 + 768 + time 128\nMLP 1472→512→256",
            "pale_red",
            "red",
        ),
    ]
    for x, y, w, h, label, face, edge in boxes:
        _box(ax, (x, y), w, h, label, COLORS[face], COLORS[edge])
    for left, right in zip(boxes[:-1], boxes[1:], strict=True):
        _arrow(
            ax,
            (left[0] + left[2], left[1] + left[3] / 2),
            (right[0], right[1] + right[3] / 2),
        )

    _box(
        ax,
        (0.28, 0.28),
        0.17,
        0.17,
        "噪声时间 t / step id\nFourier embedding 64\nMLP 64→128\nFiLM / AdaLN 条件",
        COLORS["pale_gray"],
        COLORS["gray"],
    )
    for target_x in (0.385, 0.55, 0.71):
        _arrow(
            ax,
            (0.365, 0.45),
            (target_x, 0.63),
            color=COLORS["gray"],
            dashed=True,
        )

    _box(
        ax,
        (0.53, 0.20),
        0.14,
        0.19,
        "坐标均值头 μ\nPitch 16\nRhythm 16\nModulation 17\nPhase 1+1 = 51",
        COLORS["pale_green"],
        COLORS["green"],
    )
    _box(
        ax,
        (0.70, 0.20),
        0.12,
        0.19,
        "不确定度头 log σ²\n51 维 aleatoric\n+ 3-seed ensemble\nepistemic",
        COLORS["pale_orange"],
        COLORS["orange"],
    )
    _box(
        ax,
        (0.85, 0.20),
        0.13,
        0.19,
        "冻结确定性读出\nS_hat = w^T μ + b\nfocus_band_loss\nOOD/高不确定→no-op",
        COLORS["pale_red"],
        COLORS["red"],
    )
    _arrow(ax, (0.89, 0.63), (0.60, 0.39))
    _arrow(ax, (0.90, 0.63), (0.76, 0.39))
    _arrow(ax, (0.67, 0.31), (0.85, 0.31), rad=-0.28)
    _arrow(ax, (0.82, 0.295), (0.85, 0.295))

    ax.text(
        0.02,
        0.08,
        "主网络约 3–4M 参数；不输入 prompt、类别标签或 Structure PH；"
        "避免文本捷径和未验证宏观目标进入梯度。",
        fontsize=10.5,
        color=COLORS["gray"],
    )
    ax.set_title(
        "LTSN：从 ACE-Step 预测干净潜变量到纯 Path Homology v2 指纹",
        fontsize=16,
        pad=14,
    )
    figure.tight_layout()
    _save(figure, "ltsn_network_architecture")


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
            "逐快照教师标注\n解码各自 x0_hat\nExact PH→51 坐标",
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
