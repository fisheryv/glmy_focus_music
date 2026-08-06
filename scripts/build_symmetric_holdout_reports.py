# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
METADATA = ROOT / "metadata"
FINAL_FIGURES = ROOT / "runs" / "symmetric_holdout_final" / "figures"


def _json(name: str) -> dict[str, Any]:
    return json.loads((METADATA / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _render_final_figures(permanova: pd.DataFrame, directional: pd.DataFrame) -> None:
    FINAL_FIGURES.mkdir(parents=True, exist_ok=True)
    order = ["pitch", "rhythm", "modulation", "structure", "local", "hierarchical"]
    labels = ["Pitch", "Rhythm", "Modulation", "Structure", "Local fusion", "Hierarchical"]
    x = np.arange(len(order))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10.0, 5.2))
    for offset, scale, label, color in (
        (-width / 2, 180.0, "180 s primary", "#4472C4"),
        (width / 2, 300.0, "300 s sensitivity", "#ED7D31"),
    ):
        rows = permanova[permanova["scale_seconds"] == scale].set_index("feature_set").loc[order]
        bars = axis.bar(x + offset, rows["pseudo_f"], width, label=label, color=color)
        for bar, value in zip(bars, rows["pseudo_f"], strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.12,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Permutation pseudo-F")
    axis.set_title("Frozen holdout endpoints: all p ≤ 0.005")
    axis.set_ylim(0, permanova["pseudo_f"].max() * 1.22)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(
            FINAL_FIGURES / f"holdout_frozen_endpoints.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)

    primary = directional[directional["scale_seconds"] == 180.0]
    summary = (
        primary.groupby("view")["replicated_q_0_10"]
        .agg(["sum", "count"])
        .reindex(["pitch", "rhythm", "modulation", "structure"])
    )
    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    colors = ["#4472C4", "#4472C4", "#4472C4", "#A5A5A5"]
    bars = axis.bar(["Pitch", "Rhythm", "Modulation", "Structure"], summary["sum"], color=colors)
    for bar, row in zip(bars, summary.itertuples(), strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.18,
            f"{int(row.sum)}/{int(row.count)}",
            ha="center",
            va="bottom",
        )
    axis.set_ylabel("Replicated directional metrics (BH q ≤ 0.10)")
    axis.set_title("Validation-selected metrics replicated in 180 s holdout")
    axis.set_ylim(0, summary["count"].max() * 1.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(
            FINAL_FIGURES / f"holdout_directional_replication.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)


def _view_report(
    *,
    title: str,
    view: str,
    summary: dict[str, Any],
    validation_discoveries: int,
    stable_300: int,
    h1_counts: dict[str, Any],
    holdout_row: pd.Series,
    directional: pd.DataFrame,
    figure_dir: str,
    figure_prefix: str,
    method: str,
    model_hash: str,
    extra_note: str,
) -> str:
    locked = directional[directional["view"] == view]
    primary = locked[locked["scale_seconds"] == 180.0]
    failed = primary[~primary["replicated_q_0_10"]]
    failed_text = (
        "无。"
        if failed.empty
        else "、".join(
            f"`{row.metric}`（预期 {row.expected_focus_direction}，观察为 "
            f"{row.observed_focus_direction}，q={_fmt(float(row.p_fdr_bh))}）"
            for row in failed.itertuples()
        )
    )
    h1_text = "；".join(
        f"{group} {payload.get('primary_nonzero', payload.get('nonzero', 0))}/{payload.get('n', payload.get('total', 0))}"
        for group, payload in h1_counts.items()
    )
    return f"""# {title}

生成日期：2026-08-02。切分版本：`symmetric_holdout_v2`。

> 证据边界：本报告替代旧切分结果。validation 为方案冻结阶段；holdout 为哈希门控后的单次操作性最终确认。由于 Classical holdout 在旧切分中曾属于 discovery，它不是 pristine 外部确认集。

## 结论

- 状态表示：{method}
- 模型仅使用两组各 195 首 discovery/180 s 拟合；模型 SHA-256：`{model_hash}`。
- validation/180 s 的 20 个预设指标中，{validation_discoveries} 个通过 BH-FDR q≤0.10；其中 {stable_300} 个在 validation/300 s 同方向且再次通过 FDR。
- holdout/180 s 整体表示的 pseudo-F={_fmt(float(holdout_row.pseudo_f))}，p={_fmt(float(holdout_row.p_value))}，次级家族 q={_fmt(float(holdout_row.p_fdr_bh))}。
- validation 选定并锁定 {len(primary)} 个方向性指标，holdout 中 {int(primary['direction_matched'].sum())} 个方向一致，{int(primary['replicated_q_0_10'].sum())} 个在四视角联合 BH-FDR 后复现。
- 未复现指标：{failed_text}
- validation/180 s 主阈值 H1 非零计数：{h1_text}。这些稀疏 H1 结果没有进入最终锁定方向性家族，不能据此声称稳定的 Focus 特异 H1。
- {extra_note}

## 方法与检验

状态序列构造成 top-6 非自环有向转移图；主过滤阈值固定为 0.50–0.95，敏感阈值扩展至 0.05–0.95。GLMY Path Homology 输出图指标、H0 与 H1 描述子。validation 主检验固定为 180 s，300 s 只作时长敏感性。holdout 前，输入、模型、指标、方向、融合权重、阈值与 FDR 家族均写入 `metadata/holdout_gate.json`。

## 图形

![{view} group summary](../runs/{figure_dir}/{figure_prefix}_group_summary.png)

[SVG](../runs/{figure_dir}/{figure_prefix}_group_summary.svg)

![{view} Betti curves](../runs/{figure_dir}/{figure_prefix}_betti_curves.png)

[SVG](../runs/{figure_dir}/{figure_prefix}_betti_curves.svg)

## 关键产物

- `metadata/{view if view not in {'pitch', 'modulation'} else ('pitch_v2' if view == 'pitch' else 'modulation_tertile')}_topology_segments.csv`
- `metadata/{view if view not in {'pitch', 'modulation'} else ('pitch_v2' if view == 'pitch' else 'modulation_tertile')}_statistical_tests.csv`
- `metadata/holdout_confirmation_directional_metrics.csv`
- `metadata/holdout_confirmation_permanova.csv`
"""


def main() -> int:
    holdout = _json("holdout_confirmation_summary.json")
    fusion = _json("multiview_fusion_summary.json")
    state_model = json.loads(
        (ROOT / "features" / "models" / "state_model.json").read_text(encoding="utf-8")
    )
    pitch_model = json.loads(
        (ROOT / "features" / "models" / "pitch_v2_codebook.json").read_text(
            encoding="utf-8"
        )
    )
    modulation_model = json.loads(
        (ROOT / "features" / "models" / "modulation_tertile_model.json").read_text(
            encoding="utf-8"
        )
    )
    pitch = _json("pitch_v2_summary.json")
    rhythm = _json("rhythm_analysis_summary.json")
    modulation = _json("modulation_tertile_summary.json")
    structure = _json("structure_analysis_summary.json")
    permanova = pd.read_csv(METADATA / "holdout_confirmation_permanova.csv")
    validation_permanova = pd.read_csv(METADATA / "multiview_fusion_permanova.csv")
    incremental = pd.read_csv(METADATA / "holdout_confirmation_incremental.csv")
    directional = pd.read_csv(METADATA / "holdout_confirmation_directional_metrics.csv")
    _render_final_figures(permanova, directional)
    primary_180 = permanova[permanova["scale_seconds"] == 180.0].set_index("feature_set")

    reports = {
        "path-homology-pitch-v2-analysis.md": _view_report(
            title="Path Homology 音高视角（Tonnetz codebook）最终分析",
            view="pitch",
            summary=pitch,
            validation_discoveries=pitch["primary_fdr_discoveries"],
            stable_300=pitch["replicated_same_direction"],
            h1_counts=pitch["validation_180_h1_counts"],
            holdout_row=primary_180.loc["pitch"],
            directional=directional,
            figure_dir="pitch_v2_path_homology_open",
            figure_prefix="pitch_v2",
            method="beat-synchronous chroma → Tonnetz → 16 状态平衡码本",
            model_hash=pitch_model["codebook_sha256"],
            extra_note="音高视角在 validation 与 holdout 中均是六个冻结表示里 pseudo-F 最高者。",
        ),
        "path-homology-rhythm-analysis.md": _view_report(
            title="Path Homology 节奏视角最终分析",
            view="rhythm",
            summary=rhythm,
            validation_discoveries=rhythm["primary_fdr_discoveries"],
            stable_300=rhythm["replicated_same_direction"],
            h1_counts=rhythm["validation_180_h1_counts"],
            holdout_row=primary_180.loc["rhythm"],
            directional=directional,
            figure_dir="rhythm_path_homology_open",
            figure_prefix="rhythm",
            method="8 维节奏窗口 → 标准化 → 10 状态 MiniBatch K-means",
            model_hash=state_model["model_sha256"],
            extra_note="所有 14 个锁定方向性指标都在 holdout/180 s 复现。",
        ),
        "path-homology-modulation-analysis.md": _view_report(
            title="Path Homology 调制视角最终分析",
            view="modulation",
            summary=modulation,
            validation_discoveries=modulation["primary_fdr_discoveries"],
            stable_300=modulation["replicated_same_direction"],
            h1_counts=modulation["validation_180_h1_counts"],
            holdout_row=primary_180.loc["modulation"],
            directional=directional,
            figure_dir="modulation_tertile_path_homology_open",
            figure_prefix="modulation",
            method="谱调制能量 → discovery 平衡三分位 → Low/Medium/High 三状态",
            model_hash=modulation_model["model_sha256"],
            extra_note="三状态图在 validation 与 holdout 均没有 H1；证据来自图组织与 H0，而不是环。",
        ),
        "path-homology-structure-analysis.md": _view_report(
            title="Path Homology 结构视角最终分析",
            view="structure",
            summary=structure,
            validation_discoveries=structure["primary_fdr_discoveries_q_0_10"],
            stable_300=structure["replicated_same_direction"],
            h1_counts=structure["validation_180_h1_counts"],
            holdout_row=primary_180.loc["structure"],
            directional=directional,
            figure_dir="structure_path_homology_open",
            figure_prefix="structure",
            method="声学 SSM → Foote novelty 边界 → 段级 16 状态原型",
            model_hash=state_model["model_sha256"],
            extra_note=(
                "validation/180 s 扩展阈值中没有有限 H1 区间，因此未选择机制示例；"
                "结构是四视角中复现最弱的一项。"
            ),
        ),
    }
    for name, text in reports.items():
        (DOCS / name).write_text(text, encoding="utf-8")

    holdout_rows = "\n".join(
        f"| {feature_set} | {_fmt(float(row.pseudo_f))} | {_fmt(float(row.p_value))} | {_fmt(float(row.p_fdr_bh))} |"
        for feature_set, row in primary_180.loc[
            ["pitch", "rhythm", "modulation", "structure", "local", "hierarchical"]
        ].iterrows()
    )
    directional_summary = (
        directional[directional["scale_seconds"] == 180.0]
        .groupby("view")
        .agg(
            locked=("metric", "size"),
            direction_matched=("direction_matched", "sum"),
            replicated=("replicated_q_0_10", "sum"),
        )
        .reindex(["pitch", "rhythm", "modulation", "structure"])
    )
    directional_rows = "\n".join(
        f"| {view} | {int(row.locked)} | {int(row.direction_matched)} | {int(row.replicated)} |"
        for view, row in directional_summary.iterrows()
    )
    validation_rows = "\n".join(
        f"| {name} | {discoveries}/20 | {stable} | {_fmt(float(fusion_value))} |"
        for name, discoveries, stable, fusion_value in (
            ("pitch", pitch["primary_fdr_discoveries"], pitch["replicated_same_direction"], validation_permanova[(validation_permanova.scale_seconds == 180) & (validation_permanova.feature_set == "pitch")].iloc[0].pseudo_f),
            ("rhythm", rhythm["primary_fdr_discoveries"], rhythm["replicated_same_direction"], validation_permanova[(validation_permanova.scale_seconds == 180) & (validation_permanova.feature_set == "rhythm")].iloc[0].pseudo_f),
            ("modulation", modulation["primary_fdr_discoveries"], modulation["replicated_same_direction"], validation_permanova[(validation_permanova.scale_seconds == 180) & (validation_permanova.feature_set == "modulation")].iloc[0].pseudo_f),
            ("structure", structure["primary_fdr_discoveries_q_0_10"], structure["replicated_same_direction"], validation_permanova[(validation_permanova.scale_seconds == 180) & (validation_permanova.feature_set == "structure")].iloc[0].pseudo_f),
        )
    )
    primary_increment = incremental[incremental["scale_seconds"] == 180.0].set_index("comparison")
    gate_sha = _sha256(METADATA / "holdout_gate.json")
    execution_sha = _sha256(METADATA / "holdout_confirmation_execution.json")
    final = f"""# 四视角 Path Homology 对称 holdout 最终研究报告

生成日期：2026-08-02。切分版本：`symmetric_holdout_v2`。

## 摘要

本研究按“局部三视角（音高、节奏、调制）先融合，再与宏观结构视角整合”的方案，重新完成了 600 首音乐的全流程 Path Homology。两组均严格切分为 195 discovery、60 validation、45 holdout。所有状态模型只用 discovery/180 s 拟合；validation 用于方案冻结；holdout 在 SHA-256 门控后只开启一次，且没有据其结果改参数、指标、方向、阈值、FDR 家族或融合权重。

最终结果是：局部三视角融合在 holdout/180 s 上确认组间分离（pseudo-F={_fmt(holdout['primary_180']['pseudo_f'])}，p={_fmt(holdout['primary_180']['p_value'])}），但它没有优于单独音高视角；加入结构也没有增量。44 个 validation 锁定方向性指标中，43 个方向一致，42 个在四视角联合 BH-FDR q≤0.10 后复现。证据支持“Focus 与 Classical 的状态转移组织不同”，不支持“融合必然增强”、稳定 H1/H2、注意力效果、治疗作用或因果机制。

## 1. 证据地位与限制

- validation/180 s：方案冻结层；融合方案因参考过既往单视角结果，仍属于探索性整合。
- validation/300 s：时长敏感性。
- holdout/180 s：哈希门控后的单次操作性最终确认。
- 重要限制：Classical holdout 在旧切分中曾属于 discovery，因此不能称为 pristine 外部确认集；Classical holdout 还不含钢琴独奏，不能代表完整古典总体。
- 本研究是观察性声学比较，不支持认知、临床、生成质量或因果结论。

## 2. 流程与防泄漏设计

```mermaid
flowchart LR
    A["600 tracks<br/>Focus 300 + Classical 300"] --> B["Symmetric split<br/>195 / 60 / 45 per group"]
    B --> C["Discovery 180 s only<br/>fit all state models"]
    C --> D["Transform all splits<br/>no holdout summary"]
    D --> E["Four-view Path Homology"]
    E --> F["Validation 60 + 60<br/>freeze metrics and weights"]
    F --> G["SHA-256 gate<br/>{gate_sha[:12]}…"]
    G --> H["One-time holdout 45 + 45"]
    H --> I["Final report<br/>no adaptation"]
```

预处理为 22,050 Hz、mono、float32、双遍 EBU R128 至 −15 LUFS、峰值上限 −1 dBFS。1200 个 180/300 s WAV 均通过哈希、响度、峰值和路径审计。新路径使用 `features/audio_symmetric_holdout_v2/`；音频字节与已验证旧预处理完全一致，避免重新编码漂移。

## 3. 四视角表示

1. 音高：beat-synchronous chroma → Tonnetz → 16 状态 discovery 平衡码本。
2. 节奏：8 维节奏窗口 → 标准化 → 10 状态聚类。
3. 调制：谱调制能量 → discovery 平衡三分位 → Low/Medium/High。
4. 结构：声学 SSM → Foote novelty 边界 → 段级 16 状态原型。

每个状态序列构造成 top-6 非自环有向图；主过滤阈值固定 0.50–0.95，扩展敏感阈值 0.05–0.95。指标族固定为 20 个图/H0/H1 描述子。

## 4. validation 结果与冻结决策

| 视角 | 180 s FDR 发现 | 300 s 同方向再现 | validation 融合 pseudo-F |
|---|---:|---:|---:|
{validation_rows}

局部等权融合（音高、节奏、调制各 1/3）在 validation/180 s 的 pseudo-F={_fmt(fusion['primary_180']['local_permanova']['pseudo_f'])}、p={_fmt(fusion['primary_180']['local_permanova']['p_value'])}；音高单视角为 pseudo-F={_fmt(fusion['primary_180']['pitch_permanova']['pseudo_f'])}。局部融合相对音高的增量未得到支持。结构以 0.5 权重加入层级融合后，pseudo-F 降至 {_fmt(fusion['primary_180']['hierarchical_permanova']['pseudo_f'])}；结构增量 Δpseudo-F={_fmt(fusion['primary_180']['structure_increment']['delta_pseudo_f'])}，单侧 p={_fmt(fusion['primary_180']['structure_increment']['p_value_one_sided'])}。

因此冻结：`local` 为 holdout 主终点，`hierarchical` 为次终点；局部权重 1/3–1/3–1/3，层级权重 local 0.5 / structure 0.5，不再改变。

![Validation fusion ablation](../runs/multiview_fusion/figures/multiview_permanova_ablation.png)

[SVG](../runs/multiview_fusion/figures/multiview_permanova_ablation.svg)

## 5. holdout 单次确认

门控 SHA-256：`{gate_sha}`。执行记录 SHA-256：`{execution_sha}`。

| 冻结表示 | 180 s pseudo-F | p | FDR q |
|---|---:|---:|---:|
{holdout_rows}

局部融合相对音高：Δpseudo-F={_fmt(float(primary_increment.loc['local_vs_pitch', 'delta_pseudo_f']))}，单侧 p={_fmt(float(primary_increment.loc['local_vs_pitch', 'p_value_one_sided']))}。加入结构：Δpseudo-F={_fmt(float(primary_increment.loc['add_structure', 'delta_pseudo_f']))}，单侧 p={_fmt(float(primary_increment.loc['add_structure', 'p_value_one_sided']))}。两者均为负，说明“组间可分”不等于“融合带来额外信息”。

![Holdout frozen endpoints](../runs/symmetric_holdout_final/figures/holdout_frozen_endpoints.png)

[SVG](../runs/symmetric_holdout_final/figures/holdout_frozen_endpoints.svg)

### 5.1 validation 选定方向的复现

| 视角 | 锁定指标 | 方向一致 | BH q≤0.10 复现 |
|---|---:|---:|---:|
{directional_rows}

结构视角未复现的是 `edge_density`（方向反转）与 `reciprocity`（方向一致但 q={_fmt(float(directional[(directional.scale_seconds == 180) & (directional.view == 'structure') & (directional.metric == 'reciprocity')].iloc[0].p_fdr_bh))}）。其余三视角全部锁定指标复现。

![Directional replication](../runs/symmetric_holdout_final/figures/holdout_directional_replication.png)

[SVG](../runs/symmetric_holdout_final/figures/holdout_directional_replication.svg)

## 6. 科学结论

### 支持

- Focus 与 Classical 在四种状态转移表示上均存在可重复的组间差异；音高视角最强。
- Focus 更常呈现较少状态/边、更高自转移与定向复现、更低路径熵的局部组织；该方向在音高、节奏和调制中最稳定。
- 局部三视角融合本身是有效的组间表征，并在 300 s 敏感性中保持分离。

### 不支持

- 局部三视角融合优于音高单视角：validation 与 holdout 的增量均不支持。
- 结构对局部融合有正增量：加入结构在 validation 与 holdout 都降低 pseudo-F。
- 稳定、普遍或 Focus 特异的 H1/H2。调制 H1 为零，其他视角 H1 稀疏；最终锁定方向性家族没有 H1 指标。
- 由声学拓扑差异推出专注力提升、临床疗效、生成质量或因果机制。

## 7. 可复现与审计产物

- `metadata/split_assignment_v2.csv`
- `metadata/preprocessed_segments.csv`
- `metadata/feature_segments.csv`
- `features/models/state_model.json`
- `metadata/holdout_gate.json`
- `metadata/holdout_confirmation_execution.json`
- `metadata/holdout_confirmation_permanova.csv`
- `metadata/holdout_confirmation_incremental.csv`
- `metadata/holdout_confirmation_directional_metrics.csv`
- 四份更新后的单视角报告位于 `docs/path-homology-*-analysis.md`。

最终适应性审计：参数、指标、方向、融合权重、阈值和 FDR 家族在 holdout 后均未改变。
"""
    final_path = DOCS / "path-homology-symmetric-holdout-final-report.md"
    final_path.write_text(final, encoding="utf-8")
    fusion_report = f"""# Path Homology 多视角融合最终分析

生成日期：2026-08-02。局部三视角采用音高、节奏、调制等权融合；结构以 0.5 权重与局部块做层级融合。所有块变换仅在 discovery 拟合。

## 冻结结论

- validation/180 s：pitch pseudo-F={_fmt(fusion['primary_180']['pitch_permanova']['pseudo_f'])}，local={_fmt(fusion['primary_180']['local_permanova']['pseudo_f'])}，hierarchical={_fmt(fusion['primary_180']['hierarchical_permanova']['pseudo_f'])}。
- holdout/180 s：pitch={_fmt(float(primary_180.loc['pitch', 'pseudo_f']))}，local={_fmt(float(primary_180.loc['local', 'pseudo_f']))}，hierarchical={_fmt(float(primary_180.loc['hierarchical', 'pseudo_f']))}；三者 p 均为 0.001。
- local 相对 pitch 的 holdout 增量 Δpseudo-F={_fmt(float(primary_increment.loc['local_vs_pitch', 'delta_pseudo_f']))}、单侧 p=1.000。
- 加入 structure 的 holdout 增量 Δpseudo-F={_fmt(float(primary_increment.loc['add_structure', 'delta_pseudo_f']))}、单侧 p=1.000。

结论：局部融合可以区分两组，但没有超过音高；结构也没有提供正增量。组间显著不等于互补信息增加。权重在 holdout 前锁定，之后没有改变。

![Validation ablation](../runs/multiview_fusion/figures/multiview_permanova_ablation.png)

[SVG](../runs/multiview_fusion/figures/multiview_permanova_ablation.svg)

![Holdout endpoints](../runs/symmetric_holdout_final/figures/holdout_frozen_endpoints.png)

[SVG](../runs/symmetric_holdout_final/figures/holdout_frozen_endpoints.svg)

证据边界：融合设计参考过既往单视角结果，因此 validation 融合仍属探索性整合；holdout 是重切分后的操作性最终确认，不是 pristine 外部确认。
"""
    fusion_path = DOCS / "path-homology-multiview-fusion-analysis.md"
    fusion_path.write_text(fusion_report, encoding="utf-8")
    print(final_path.relative_to(ROOT).as_posix())
    for name in reports:
        print((DOCS / name).relative_to(ROOT).as_posix())
    print(fusion_path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
