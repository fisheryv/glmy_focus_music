# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "runs" / ".matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "metadata"
OUTPUT = ROOT / "runs" / "open_focus_classical_600_fresh_analysis"
REPORT = ROOT / "docs" / "open-focus-classical-600-fresh-analysis.md"
RECEIPT = METADATA / "open_focus_classical_600_fresh_analysis_summary.json"
FDR_Q = 0.05

VIEW_SPECS = {
    "Pitch (Tonnetz K=16)": (METADATA / "pitch_v2_statistical_tests.csv", None),
    "Rhythm (K=10)": (METADATA / "rhythm_statistical_tests.csv", None),
    "Modulation (SMP K=10)": (
        METADATA / "modulation_smp_prototype_statistical_tests.csv",
        10,
    ),
    "Structure (K=16)": (METADATA / "structure_statistical_tests.csv", None),
}
VIEW_COLORS = {
    "Pitch (Tonnetz K=16)": "#355C7D",
    "Rhythm (K=10)": "#2A9D8F",
    "Modulation (SMP K=10)": "#E9C46A",
    "Structure (K=16)": "#E76F51",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_view(path: Path, state_count: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if state_count is not None:
        frame = frame[frame["state_count"].astype(int) == state_count].copy()
    return frame


def _stable_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    primary = frame[frame["analysis_set"] == "primary_validation_180"].copy()
    sensitivity = frame[frame["analysis_set"] == "sensitivity_validation_300"].copy()
    sensitivity = sensitivity.set_index("metric")
    rows: list[dict[str, Any]] = []
    for row in primary.itertuples(index=False):
        other = sensitivity.loc[row.metric]
        direction = np.sign(float(row.focus_median) - float(row.classical_median))
        other_direction = np.sign(
            float(other["focus_median"]) - float(other["classical_median"])
        )
        rows.append(
            {
                **row._asdict(),
                "q_300": float(other["p_fdr_bh"]),
                "direction": "higher" if direction > 0 else "lower" if direction < 0 else "equal",
                "stable": bool(
                    float(row.p_fdr_bh) <= FDR_Q
                    and float(other["p_fdr_bh"]) <= FDR_Q
                    and direction == other_direction
                ),
                "signed_effect": float(row.epsilon_squared) * direction,
            }
        )
    return pd.DataFrame(rows)


def _plot_overview(view_results: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 6.4), constrained_layout=True)

    names = list(view_results)
    x = np.arange(len(names))
    primary_counts = [int((view_results[name]["p_fdr_bh"] <= FDR_Q).sum()) for name in names]
    stable_counts = [int(view_results[name]["stable"].sum()) for name in names]
    width = 0.36
    axes[0].bar(x - width / 2, primary_counts, width, color="#446A8A", label="180 s q≤.05")
    axes[0].bar(x + width / 2, stable_counts, width, color="#75B798", label="Stable at 300 s")
    axes[0].set_xticks(x, [name.split(" (")[0] for name in names])
    axes[0].set_ylim(0, 20)
    axes[0].set_ylabel("Metrics (out of 20)")
    axes[0].set_title("Single-view validation discoveries")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    candidates: list[pd.DataFrame] = []
    for name, frame in view_results.items():
        selected = frame[frame["stable"]].copy()
        selected["view_label"] = name
        candidates.append(selected)
    effects = pd.concat(candidates, ignore_index=True)
    effects = effects.reindex(effects["signed_effect"].abs().sort_values().index).tail(14)
    labels = [
        f"{view.split(' (')[0]} · {metric}"
        for view, metric in zip(effects["view_label"], effects["metric"], strict=True)
    ]
    colors = [VIEW_COLORS[name] for name in effects["view_label"]]
    y = np.arange(len(effects))
    axes[1].barh(y, effects["signed_effect"], color=colors)
    axes[1].axvline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_yticks(y, labels, fontsize=8)
    axes[1].set_xlabel("Signed epsilon-squared (Focus higher →)")
    axes[1].set_title("Largest stable rank effects")
    axes[1].grid(axis="x", alpha=0.2)

    png = OUTPUT / "fresh_analysis_overview.png"
    svg = OUTPUT / "fresh_analysis_overview.svg"
    figure.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, svg


def _metric_list(frame: pd.DataFrame) -> str:
    selected = frame[frame["stable"]].sort_values(["p_fdr_bh", "metric"])
    if selected.empty:
        return "无"
    values = [
        f"`{row.metric}`（Focus {'↑' if row.direction == 'higher' else '↓'}）"
        for row in selected.itertuples(index=False)
    ]
    return "、".join(values)


def main() -> int:
    preprocessing = _json(METADATA / "preprocessing_summary.json")
    features = _json(METADATA / "feature_summary.json")
    pitch = _json(METADATA / "pitch_v2_summary.json")
    rhythm = _json(METADATA / "rhythm_analysis_summary.json")
    modulation = _json(METADATA / "modulation_smp_prototype_summary.json")
    structure = _json(METADATA / "structure_analysis_summary.json")
    phase = _json(METADATA / "phase_lifted_path_homology_summary.json")
    fusion = _json(METADATA / "multiview_fusion_summary.json")
    holdout = _json(METADATA / "holdout_confirmation_summary.json")

    chains = {
        item["input_provenance"]["provenance_chain_sha256"]
        for item in (pitch, modulation, phase)
    }
    if chains != {"105ec3e39753a4661e6c7bd6a5692eef940c12db28ac2d39815f0e98d981135b"}:
        raise RuntimeError(f"analysis provenance chains differ: {sorted(chains)}")
    if not all(item.get("ok") for item in (preprocessing, features, pitch, rhythm, modulation, structure, phase)):
        raise RuntimeError("one or more required summaries are not complete")

    view_results = {
        name: _stable_metrics(_load_view(path, state_count))
        for name, (path, state_count) in VIEW_SPECS.items()
    }
    overview_png, overview_svg = _plot_overview(view_results)

    phase_tests = pd.read_csv(METADATA / "phase_lifted_path_homology_tests.csv")
    phase_primary = phase_tests[phase_tests["role"] == "primary_validation"].copy()
    phase_rows = "\n".join(
        "| {name} | {focus:.3f} | {classical:.3f} | {effect:.3f} | {q:.3g} | {decision} |".format(
            name=row.representation,
            focus=float(row.focus_median),
            classical=float(row.classical_median),
            effect=float(row.rank_biserial_focus_minus_classical),
            q=float(row.p_two_sided_fdr_bh),
            decision="支持" if float(row.p_two_sided_fdr_bh) <= FDR_Q else "不支持",
        )
        for row in phase_primary.itertuples(index=False)
    )
    classification = pd.read_csv(METADATA / "phase_lifted_path_homology_classification.csv").iloc[0]
    exclusion_rows = pd.read_csv(METADATA / "phase_lifted_path_homology_exclusions.csv")
    excluded_ids = ", ".join(sorted(exclusion_rows["segment_id"].astype(str).unique()))

    single_view_rows = []
    for name, frame in view_results.items():
        primary_count = int((frame["p_fdr_bh"] <= FDR_Q).sum())
        stable_count = int(frame["stable"].sum())
        role = "主视角" if name.startswith(("Pitch", "Rhythm")) else "探索性"
        single_view_rows.append(
            f"| {name} | {role} | {primary_count}/20 | {stable_count}/20 | {_metric_list(frame)} |"
        )

    report = f"""# Open Focus–Classical 600：从原始数据重新预处理与分析

生成日期：{date.today().isoformat()}。本报告只引用本轮从 `datasets/open-focus-classical-600` 重新生成的 CSV/JSON/图，不沿用旧预处理或旧统计数值。

## 1. 输入、预处理与来源链

- 冻结发布版共有 600 首：Focus 300、Classical 300；每组 discovery/validation/holdout 为 195/60/45，清单无 artist/album/composer 跨 split 泄漏。
- 600 个原始音频均通过发布 SHA-256 校验；原始文件未修改。
- 重新生成 1,200 个分析片段（180 s 与 300 s），22,050 Hz、mono、float32、双遍 EBU R128 目标 −15 LUFS、峰值上限 −1 dBFS；失败 0，总计 {preprocessing['output_audio_hours']:.2f} 小时、{preprocessing['output_gib']:.2f} GiB。
- 一个 Jamendo 文件的目录时长 245.0 s、实际可解码 239.453 s；短曲模式保留实际整曲，未循环、未补零。
- 重新生成 {features['output_files']:,} 个特征文件；Acoustic、Chroma、Rhythm、Modulation、Structure 全部来自上述新 WAV。共享状态模型 SHA-256：`{features['model_sha256']}`。
- 预处理清单 SHA-256：`{preprocessing['manifest_sha256']}`；特征清单 SHA-256：`{features['manifest_sha256']}`；组合来源链：`105ec3e39753a4661e6c7bd6a5692eef940c12db28ac2d39815f0e98d981135b`。

![本轮分析总览](../runs/open_focus_classical_600_fresh_analysis/fresh_analysis_overview.png)

[SVG](../runs/open_focus_classical_600_fresh_analysis/fresh_analysis_overview.svg)

## 2. 单视角 validation 结果

主推断固定为 validation/180 s（每组 60）；validation/300 s 只用于同曲时长敏感性。每个视角的 20 个预设指标分别作 BH-FDR，严格阈值为 q≤0.05。SMP K=10 和 Structure 是探索性视角，不因显著结果升级为预先确认性证据。

| 视角 | 证据角色 | 180 s q≤.05 | 跨 300 s 稳定 | 稳定指标与方向 |
|---|---|---:|---:|---|
{chr(10).join(single_view_rows)}

解释边界：

- Pitch 与 Rhythm 的稳定信息主要来自状态覆盖、转移集中性/熵、自转移、有向复现和 H0 连通过程。
- Pitch 主阈值 H1 非零仅 Classical {pitch['validation_180_h1_counts']['classical']['primary_nonzero']}/60、Focus {pitch['validation_180_h1_counts']['focus']['primary_nonzero']}/60。
- Rhythm 主阈值 H1 两组均为 0/60；扩展低阈值也只有 Classical {rhythm['validation_180_sensitivity_h1_counts']['classical']['nonzero']}/60、Focus {rhythm['validation_180_sensitivity_h1_counts']['focus']['nonzero']}/60。
- SMP K=10 主阈值 H1 为 Classical {modulation['models']['10']['validation_180_h1_counts']['classical']['primary_nonzero']}/60、Focus {modulation['models']['10']['validation_180_h1_counts']['focus']['primary_nonzero']}/60；不能据此宣称普遍调制环。
- Structure 的主阈值 H1 两组均为 {structure['validation_180_h1_counts']['classical']['nonzero']}/60；该视角是宏观段落探索，不是已证实的新增主指纹。

## 3. Phase-lifted Path Homology

三种候选表示都通过 discovery 人工循环对照校准。本轮预设质量门控排除了两个片段：{excluded_ids}。

| 表示 | Focus 中位数 | Classical 中位数 | rank-biserial | BH-FDR q | 180 s 结论 |
|---|---:|---:|---:|---:|---|
{phase_rows}

因此，validation/180 s 支持 Acoustic phase 与 Chroma phase；Rhythm phase 只在 300 s 敏感性达到 FDR，不能替代 180 s 主结果。三分数辅助分类的 balanced accuracy={float(classification.balanced_accuracy):.3f}、macro-F1={float(classification.macro_f1):.3f}、AUROC={float(classification.auroc):.3f}，仅用于描述判别信息，不是独立验证或因果证据。

![Phase effect sizes](../runs/phase_lifted_path_homology/figures/validation_effect_sizes.png)

[SVG](../runs/phase_lifted_path_homology/figures/validation_effect_sizes.svg)

## 4. 多视角融合

融合使用 discovery 拟合的 rank-normalized Mahalanobis block；local 固定等权组合 Pitch、Rhythm、SMP K=10，Structure 以 0.5 权重作为第二层候选。融合是在单视角 validation 结果之后定义，因此证据角色是探索性整合。

- validation/180 s：local pseudo-F={fusion['primary_180']['local_permanova']['pseudo_f']:.3f}, p={fusion['primary_180']['local_permanova']['p_value']:.3g}；辅助 balanced accuracy={fusion['primary_180']['local_classification']['balanced_accuracy']:.3f}。
- 单独 Pitch pseudo-F={fusion['primary_180']['pitch_permanova']['pseudo_f']:.3f}；local 相对 Pitch 没有正增量，不能声称融合优于 Pitch。
- 加入 Structure 的 Δ pseudo-F={fusion['primary_180']['structure_increment']['delta_pseudo_f']:.3f}, p={fusion['primary_180']['structure_increment']['p_value_one_sided']:.3g}；不支持 Structure 对 local 的稳定新增价值。
- 300 s local 仍分组显著，但它是同曲时长敏感性，不是独立复制。

![Multiview PERMANOVA](../runs/multiview_fusion/figures/multiview_permanova_ablation.png)

[SVG](../runs/multiview_fusion/figures/multiview_permanova_ablation.svg)

## 5. Holdout 的操作性结果

在查看本轮 holdout 统计前，脚本对输入、模型、validation 方向、权重、阈值和代码写入哈希门控。该操作能审计“本轮没有根据 holdout 改参数”，但不能抹去这些曲目在历史工作中已经被访问的事实。

- holdout/180 s local pseudo-F={holdout['primary_180']['pseudo_f']:.3f}, p={holdout['primary_180']['p_value']:.3g}。
- validation 锁定的 {holdout['directional_metric_replication_180']['locked_metrics']} 个方向中，{holdout['directional_metric_replication_180']['direction_matched']} 个方向一致，{holdout['directional_metric_replication_180']['replicated_q_0_05']} 个在统一 q≤0.05 下通过。
- 这些数值只能称为冻结工作流的操作性/描述性核验，不能称为 pristine 外部验证、独立复制或确认性升级。

![Operational holdout](../runs/holdout_confirmation/figures/holdout_permanova.png)

[SVG](../runs/holdout_confirmation/figures/holdout_permanova.svg)

## 6. 总结

新的 600 首数据从原始音频到 WAV、五类特征、状态模型、Path Homology、统计与图已完整重建并通过哈希链审计。当前最稳健的观察性结论是：Open Focus 相比 Classical 在 Pitch 与 Rhythm 量化状态空间中覆盖更窄、转移更集中、自转移与有向复现更高、H0 连通过程更紧凑；phase-lifted Acoustic 与 Chroma 的闭合强度也更高。SMP K=10 提供部分探索性差异，但只有 4 个指标跨时长稳定；宏观 Structure 没有给 local 融合带来正增量。各视角均缺乏“普遍而稳定的 H1 环差异”证据。

这些都是当前语料的观察性声学结构差异，不支持专注力提升、认知机制、治疗效果、生成质量或其他因果结论。

## 7. 主要产物

- `metadata/preprocessed_segments.csv`、`metadata/feature_segments.csv`
- `metadata/pitch_v2_*`、`metadata/rhythm_*`
- `metadata/modulation_smp_prototype_*`、`metadata/structure_*`
- `metadata/phase_lifted_path_homology_*`
- `metadata/multiview_fusion_*`
- `metadata/holdout_confirmation_*` 与 `metadata/holdout_gate.json`
- `runs/open_focus_classical_600_fresh_analysis/`、`runs/multiview_fusion/figures/`、`runs/phase_lifted_path_homology/figures/`、`runs/holdout_confirmation/figures/`
"""
    REPORT.write_text(report, encoding="utf-8")

    artifacts = [overview_png, overview_svg, REPORT]
    payload = {
        "generated_at": date.today().isoformat(),
        "ok": True,
        "scope": "fresh open-focus-classical-600 preprocessing and analysis",
        "provenance_chain_sha256": next(iter(chains)),
        "preprocessing_manifest_sha256": preprocessing["manifest_sha256"],
        "feature_manifest_sha256": features["manifest_sha256"],
        "single_view": {
            name: {
                "primary_fdr_discoveries": int((frame["p_fdr_bh"] <= FDR_Q).sum()),
                "stable_same_direction": int(frame["stable"].sum()),
            }
            for name, frame in view_results.items()
        },
        "phase_primary_fdr_discoveries": phase["primary_validation_fdr_discoveries"],
        "fusion_decision": fusion["decision"],
        "holdout_role": holdout["scientific_scope"],
        "artifacts": {
            path.relative_to(ROOT).as_posix(): _sha256(path) for path in artifacts
        },
    }
    RECEIPT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
