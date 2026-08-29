from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG = ROOT / "tmp" / "matplotlib"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402

from topology.multiview_fusion import (  # noqa: E402
    DiscoveryMahalanobisBlock,
    equal_block_fusion,
)
from topology.statistics import TOPOLOGY_METRICS  # noqa: E402

METADATA = ROOT / "metadata"
CONFIG_PATH = ROOT / "configs" / "focus_path_homology_fingerprint_v2.toml"
PROFILE_PATH = METADATA / "focus_path_homology_fingerprint_v2.json"
SCORES_PATH = METADATA / "focus_path_homology_fingerprint_v2_scores.csv"
DIRECTIONS_PATH = METADATA / "focus_path_homology_fingerprint_v2_directions.csv"
SUMMARY_PATH = METADATA / "focus_path_homology_fingerprint_v2_summary.json"
OUTPUT = ROOT / "runs" / "focus_path_homology_fingerprint_v2"
FIGURES = OUTPUT / "figures"

VIEW_FILES = {
    "pitch": METADATA / "pitch_v2_topology_segments.csv",
    "rhythm": METADATA / "rhythm_topology_segments.csv",
    "modulation": METADATA / "modulation_tertile_topology_segments.csv",
    "structure": METADATA / "structure_topology_segments.csv",
}
PHASE_FILE = METADATA / "phase_lifted_path_homology_features.csv"
HOLDOUT_GATE = METADATA / "holdout_gate.json"
PHASE_TESTS = METADATA / "phase_lifted_path_homology_tests.csv"
FUSION_SUMMARY = METADATA / "multiscale_hierarchical_fusion_summary.json"
FUSION_CLASSIFICATION = METADATA / "multiscale_hierarchical_fusion_classification.csv"
FUSION_HOLDOUT = METADATA / "multiscale_hierarchical_fusion_holdout_descriptive.csv"

IDENTITY = ["segment_id", "track_id", "group", "split", "scale_seconds"]
COLORS = {
    "blue": "#2563EB",
    "orange": "#C2410C",
    "green": "#15803D",
    "purple": "#7E22CE",
    "red": "#B91C1C",
    "gray": "#475569",
    "pale_blue": "#DBEAFE",
    "pale_orange": "#FFEDD5",
    "pale_green": "#DCFCE7",
    "pale_purple": "#F3E8FF",
    "pale_red": "#FEE2E2",
}


@dataclass(frozen=True, slots=True)
class SerializedBlock:
    input_features: list[str]
    imputer_median: list[float]
    keep_mask: list[bool]
    retained_mean: list[float]
    whitening: list[list[float]]
    effective_rank: int
    output_dimensions: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialize_block(
    transformer: DiscoveryMahalanobisBlock,
    input_features: list[str],
) -> SerializedBlock:
    if any(
        value is None
        for value in (
            transformer.imputer,
            transformer.keep,
            transformer.mean,
            transformer.whitening,
        )
    ):
        raise RuntimeError("block transformer is not fitted")
    assert transformer.imputer is not None
    assert transformer.keep is not None
    assert transformer.mean is not None
    assert transformer.whitening is not None
    return SerializedBlock(
        input_features=input_features,
        imputer_median=[float(value) for value in transformer.imputer.statistics_],
        keep_mask=[bool(value) for value in transformer.keep],
        retained_mean=[float(value) for value in transformer.mean],
        whitening=transformer.whitening.astype(float).tolist(),
        effective_rank=int(transformer.effective_rank),
        output_dimensions=int(transformer.whitening.shape[1]),
    )


def _load_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    canonical: pd.MultiIndex | None = None
    for view, path in VIEW_FILES.items():
        frame = pd.read_csv(path)
        indexed = frame.set_index(IDENTITY).sort_index()
        if canonical is None:
            canonical = indexed.index
        elif not indexed.index.equals(canonical):
            raise RuntimeError(f"{view} identities do not align")
        if (indexed["status"] == "failed").any():
            raise RuntimeError(f"{view} contains failed rows")
        frames[view] = indexed
    assert canonical is not None

    phase = pd.read_csv(PHASE_FILE)
    pivot = phase.pivot(index=IDENTITY, columns="representation", values="loop_score")
    pivot = pivot.reindex(canonical)
    identity = canonical.to_frame(index=False)
    return identity, frames, pivot


def _mask(identity: pd.DataFrame, split: str, scale: float = 180.0) -> np.ndarray:
    return (identity["split"].astype(str).to_numpy() == split) & np.isclose(
        identity["scale_seconds"].to_numpy(float), scale
    )


def _fit_blocks(
    identity: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    phase: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, SerializedBlock]]:
    discovery = _mask(identity, config["fingerprint"]["reference_split"])
    matrices: dict[str, np.ndarray] = {}
    serialized: dict[str, SerializedBlock] = {}
    for view in (*config["blocks"]["local_views"], *config["blocks"]["auxiliary_views"]):
        raw = frames[view].loc[:, TOPOLOGY_METRICS].to_numpy(float)
        transformer = DiscoveryMahalanobisBlock().fit(raw[discovery])
        matrices[view] = transformer.transform(raw)
        serialized[view] = _serialize_block(transformer, list(TOPOLOGY_METRICS))
    for view in config["blocks"]["phase_views"]:
        raw = phase.loc[:, [view]].to_numpy(float)
        transformer = DiscoveryMahalanobisBlock().fit(raw[discovery])
        matrices[view] = transformer.transform(raw)
        serialized[view] = _serialize_block(transformer, ["loop_score"])
    return matrices, serialized


def _build_directions() -> pd.DataFrame:
    gate = json.loads(HOLDOUT_GATE.read_text(encoding="utf-8"))
    rows = []
    for item in gate["analysis_specification"]["directional_metrics"]:
        rows.append(
            {
                "layer": "local" if item["view"] != "structure" else "macro_auxiliary",
                "view": item["view"],
                "metric": item["metric"],
                "expected_focus_direction": item["expected_focus_direction"],
                "validation_classical_median": item["validation_classical_median"],
                "validation_focus_median": item["validation_focus_median"],
                "validation_p_fdr_bh": item["validation_p_fdr_bh"],
                "evidence_role": "frozen_before_holdout",
            }
        )
    phase_tests = pd.read_csv(PHASE_TESTS)
    selected = phase_tests[
        (phase_tests["split"] == "validation")
        & np.isclose(phase_tests["scale_seconds"], 180.0)
        & phase_tests["representation"].isin(["path_acoustic_phase", "path_chroma_phase"])
    ]
    for row in selected.itertuples(index=False):
        rows.append(
            {
                "layer": "phase",
                "view": row.representation,
                "metric": "loop_score",
                "expected_focus_direction": "greater",
                "validation_classical_median": row.classical_median,
                "validation_focus_median": row.focus_median,
                "validation_p_fdr_bh": row.p_focus_greater_fdr_bh,
                "evidence_role": "exploratory_phase_increment",
            }
        )
    return pd.DataFrame(rows).sort_values(["layer", "view", "metric"]).reset_index(drop=True)


def _save_figure(figure: Any, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _box(
    ax: Any,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    face: str,
    edge: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.5,
        )
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        linespacing=1.35,
        color="#0F172A",
    )


def _arrow(ax: Any, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color=COLORS["gray"],
        )
    )


def _routed_arrow(
    ax: Any,
    points: list[tuple[float, float]],
) -> None:
    for start, end in zip(points[:-2], points[1:-1], strict=True):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=COLORS["gray"],
            linewidth=1.4,
        )
    _arrow(ax, points[-2], points[-1])


def _plot_composition() -> None:
    figure, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(
        ax,
        (0.02, 0.68),
        0.16,
        0.17,
        "Pitch PH\n20 指标 / rank 13",
        COLORS["pale_blue"],
        COLORS["blue"],
    )
    _box(
        ax,
        (0.02, 0.43),
        0.16,
        0.17,
        "Rhythm PH\n20 指标 / rank 13",
        COLORS["pale_blue"],
        COLORS["blue"],
    )
    _box(
        ax,
        (0.02, 0.18),
        0.16,
        0.17,
        "Modulation PH\n20 指标 / rank 14",
        COLORS["pale_blue"],
        COLORS["blue"],
    )
    _box(
        ax,
        (0.25, 0.43),
        0.17,
        0.22,
        "局部块 L\n三视角等权\n49 维",
        COLORS["pale_green"],
        COLORS["green"],
    )
    _box(
        ax,
        (0.47, 0.68),
        0.16,
        0.17,
        "Acoustic phase PH\nloop_score",
        COLORS["pale_purple"],
        COLORS["purple"],
    )
    _box(
        ax,
        (0.47, 0.43),
        0.16,
        0.17,
        "Chroma phase PH\nloop_score",
        COLORS["pale_purple"],
        COLORS["purple"],
    )
    _box(
        ax,
        (0.68, 0.52),
        0.13,
        0.20,
        "相位块 P\n两视角等权\n2 维",
        COLORS["pale_orange"],
        COLORS["orange"],
    )
    _box(
        ax,
        (0.84, 0.43),
        0.14,
        0.26,
        "主指纹 L+P\n51 维\nFocus 判别分数",
        COLORS["pale_red"],
        COLORS["red"],
    )
    _box(
        ax,
        (0.56, 0.10),
        0.18,
        0.17,
        "Structure PH（S）\n宏观辅助层\n不并入主指纹",
        "#F1F5F9",
        COLORS["gray"],
    )
    for y in (0.765, 0.515, 0.265):
        _arrow(ax, (0.18, y), (0.25, 0.54))
    _routed_arrow(
        ax,
        [(0.42, 0.48), (0.45, 0.34), (0.79, 0.34), (0.84, 0.48)],
    )
    _arrow(ax, (0.63, 0.765), (0.68, 0.62))
    _arrow(ax, (0.63, 0.515), (0.68, 0.62))
    _arrow(ax, (0.81, 0.62), (0.84, 0.58))
    ax.text(
        0.33,
        0.05,
        "完全使用 Path Homology；不包含 Vietoris–Rips TDA 端点。",
        ha="center",
        fontsize=11,
        color=COLORS["gray"],
    )
    ax.set_title("focus_path_homology_fingerprint_v2 组成", fontsize=16, pad=12)
    figure.tight_layout()
    _save_figure(figure, "fingerprint_composition")


def _plot_validation(classification: pd.DataFrame, holdout: pd.DataFrame) -> None:
    order = ["L", "P", "LP", "S", "LPS"]
    labels = ["L", "P", "L+P", "S", "L+P+S"]
    validation = classification[np.isclose(classification["scale_seconds"], 180.0)].set_index(
        "feature_set"
    )
    held = holdout[np.isclose(holdout["scale_seconds"], 180.0)].set_index("feature_set")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    y = np.arange(len(order))
    for ax, metric, title in (
        (axes[0], "balanced_accuracy", "Balanced accuracy"),
        (axes[1], "macro_auroc_ovr", "Macro AUROC"),
    ):
        val = np.array([validation.loc[name, metric] for name in order])
        out = np.array([held.loc[name, metric] for name in order])
        ax.scatter(val, y - 0.10, color=COLORS["blue"], marker="o", label="validation/180")
        ax.scatter(out, y + 0.10, color=COLORS["orange"], marker="s", label="opened holdout")
        for index, value in enumerate(val):
            ax.text(value + 0.008, index - 0.10, f"{value:.3f}", va="center", fontsize=8)
        for index, value in enumerate(out):
            ax.text(value + 0.008, index + 0.10, f"{value:.3f}", va="center", fontsize=8)
        ax.axvline(0.5, color=COLORS["gray"], linestyle="--", linewidth=0.8)
        ax.set_xlim(0.48, 1.04)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=0.2)
    figure.suptitle("纯 Path Homology 各层级的判别表现")
    axes[0].legend(frameon=False, loc="upper left")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(figure, "fingerprint_validation")


def _plot_score_distribution(scores: pd.DataFrame) -> None:
    subset = scores[(scores["split"] == "validation") & np.isclose(scores["scale_seconds"], 180.0)]
    groups = ["focus", "classical"]
    values = [subset.loc[subset["group"] == group, "focus_probability"] for group in groups]
    figure, ax = plt.subplots(figsize=(7.8, 4.8))
    boxes = ax.boxplot(values, tick_labels=["Open Focus", "Classical"], patch_artist=True)
    for patch, color in zip(boxes["boxes"], [COLORS["blue"], COLORS["orange"]], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    rng = np.random.default_rng(20260716)
    for index, series in enumerate(values, start=1):
        jitter = rng.normal(index, 0.035, len(series))
        ax.scatter(
            jitter, series, s=15, alpha=0.45, color=[COLORS["blue"], COLORS["orange"]][index - 1]
        )
    ax.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1.0, label="固定阈值 0.5")
    ax.set_ylabel("Discovery-trained Focus probability")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("L+P 主指纹在 validation/180 s 的分数分布")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, "focus_score_distribution")


def _plot_directions(directions: pd.DataFrame) -> None:
    primary = directions[directions["layer"].isin(["local", "phase"])].copy()
    view_order = ["pitch", "rhythm", "modulation", "path_acoustic_phase", "path_chroma_phase"]
    metric_order = [
        "vertex_count",
        "edge_count",
        "edge_density",
        "reciprocity",
        "self_transition_ratio",
        "transition_entropy",
        "path_entropy",
        "directed_recurrence",
        "h0_betti_auc",
        "h0_betti_mean",
        "h0_betti_max",
        "h0_interval_count",
        "h0_observed_persistence",
        "h0_censored_count",
        "loop_score",
    ]
    matrix = np.full((len(view_order), len(metric_order)), np.nan)
    for row in primary.itertuples(index=False):
        i = view_order.index(row.view)
        j = metric_order.index(row.metric)
        matrix[i, j] = 1.0 if row.expected_focus_direction == "greater" else -1.0
    figure, ax = plt.subplots(figsize=(14, 4.4))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.matplotlib.colors.ListedColormap([COLORS["orange"], COLORS["blue"]])
    cmap.set_bad("#F1F5F9")
    ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metric_order)), metric_order, rotation=45, ha="right")
    ax.set_yticks(
        range(len(view_order)),
        ["Pitch", "Rhythm", "Modulation", "Acoustic phase", "Chroma phase"],
    )
    for i in range(len(view_order)):
        for j in range(len(metric_order)):
            if np.isnan(matrix[i, j]):
                continue
            ax.text(
                j,
                i,
                "↑" if matrix[i, j] > 0 else "↓",
                ha="center",
                va="center",
                color="white",
                fontsize=12,
            )
    ax.set_title("主指纹的冻结方向性签名（Focus 相对 Classical）")
    figure.text(
        0.5,
        -0.01,
        "蓝色↑：Focus 更高；橙色↓：Focus 更低；灰色：未锁定",
        ha="center",
        color=COLORS["gray"],
    )
    figure.tight_layout()
    _save_figure(figure, "directional_signature")


def main() -> int:
    with CONFIG_PATH.open("rb") as handle:
        config = tomllib.load(handle)
    identity, frames, phase = _load_inputs()
    matrices, serialized = _fit_blocks(identity, frames, phase, config)
    local = equal_block_fusion([matrices[name] for name in config["blocks"]["local_views"]])
    phase_block = equal_block_fusion([matrices[name] for name in config["blocks"]["phase_views"]])
    lp = equal_block_fusion([local, phase_block])

    discovery = _mask(identity, "discovery")
    labels = identity.loc[discovery, "group"].astype(str).to_numpy()
    classifier = LogisticRegression(
        C=float(config["classifier"]["c"]),
        class_weight="balanced",
        max_iter=int(config["classifier"]["max_iter"]),
        solver="lbfgs",
        random_state=int(config["fingerprint"]["random_seed"]),
    ).fit(lp[discovery], labels)
    if classifier.classes_.tolist() != ["classical", "focus"]:
        raise RuntimeError(f"unexpected class order: {classifier.classes_.tolist()}")
    logits = classifier.decision_function(lp)
    probabilities = classifier.predict_proba(lp)[:, 1]
    focus_discovery = discovery & (identity["group"].astype(str).to_numpy() == "focus")
    target_logit = float(
        np.quantile(logits[focus_discovery], config["classifier"]["focus_target_logit_quantile"])
    )
    band_loss = np.maximum(0.0, target_logit - logits) ** 2
    predictions = np.where(probabilities >= 0.5, "focus", "classical")

    scores = identity.copy()
    scores["local_l2_norm"] = np.linalg.norm(local, axis=1)
    scores["phase_l2_norm"] = np.linalg.norm(phase_block, axis=1)
    scores["focus_logit"] = logits
    scores["focus_probability"] = probabilities
    scores["focus_band_loss"] = band_loss
    scores["predicted_group"] = predictions
    scores.to_csv(SCORES_PATH, index=False, encoding="utf-8", lineterminator="\n")

    directions = _build_directions()
    directions.to_csv(DIRECTIONS_PATH, index=False, encoding="utf-8", lineterminator="\n")

    validation = _mask(identity, "validation")
    holdout = _mask(identity, "holdout")
    validation_labels = (identity.loc[validation, "group"].astype(str) == "focus").astype(int)
    validation_prediction = (probabilities[validation] >= 0.5).astype(int)
    holdout_labels = (identity.loc[holdout, "group"].astype(str) == "focus").astype(int)
    holdout_prediction = (probabilities[holdout] >= 0.5).astype(int)

    validation_metrics = {
        "n": int(validation.sum()),
        "balanced_accuracy": float(
            balanced_accuracy_score(validation_labels, validation_prediction)
        ),
        "roc_auc": float(roc_auc_score(validation_labels, probabilities[validation])),
    }
    holdout_metrics = {
        "n": int(holdout.sum()),
        "balanced_accuracy_descriptive": float(
            balanced_accuracy_score(holdout_labels, holdout_prediction)
        ),
        "roc_auc_descriptive": float(roc_auc_score(holdout_labels, probabilities[holdout])),
    }

    fusion_summary = json.loads(FUSION_SUMMARY.read_text(encoding="utf-8"))
    profile = {
        "schema_version": 2,
        "fingerprint_id": config["fingerprint"]["fingerprint_id"],
        "scope": "pure Path Homology L+P Focus-vs-Classical fingerprint",
        "reference_split": "discovery",
        "reference_scale_seconds": 180.0,
        "reference_sample_count": int(discovery.sum()),
        "reference_focus_count": int(focus_discovery.sum()),
        "contains_tda_features": False,
        "primary_layers": {
            "L": config["blocks"]["local_views"],
            "P": config["blocks"]["phase_views"],
            "LP_dimensions": int(lp.shape[1]),
            "weights": {"L": 0.5, "P": 0.5},
        },
        "auxiliary_layers": {
            "structure": {
                "role": "macro explanation only; excluded from primary fingerprint",
                "reason": "no positive increment over L+P",
            }
        },
        "block_transforms": {name: asdict(block) for name, block in serialized.items()},
        "classifier": {
            "kind": "l2_logistic_regression",
            "classes": classifier.classes_.tolist(),
            "positive_class": "focus",
            "c": float(classifier.C),
            "coefficient": classifier.coef_[0].astype(float).tolist(),
            "intercept": float(classifier.intercept_[0]),
            "decision_threshold_probability": 0.5,
            "focus_target_logit_quantile": float(
                config["classifier"]["focus_target_logit_quantile"]
            ),
            "focus_target_logit": target_logit,
            "control_loss": "max(0, focus_target_logit - focus_logit)^2",
        },
        "directional_signature_counts": {
            layer: int(count) for layer, count in directions.groupby("layer").size().items()
        },
        "validation_180": validation_metrics,
        "opened_holdout_180_descriptive": holdout_metrics,
        "fusion_evidence": fusion_summary["primary_180"],
        "status": {
            "fingerprint": "exploratory_validated_path_homology_fingerprint",
            "allowed": ["exact scoring", "shadow mode", "experimental reranking"],
            "sampling_guidance": "requires separate surrogate and paired generation validation",
            "confirmatory": False,
        },
        "excluded": [
            "all Vietoris-Rips TDA endpoints",
            "structure from primary score",
            "rhythm phase from primary P",
            "H1/H2 as directional targets",
        ],
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in (
                *VIEW_FILES.values(),
                PHASE_FILE,
                HOLDOUT_GATE,
                FUSION_SUMMARY,
                CONFIG_PATH,
            )
        },
    }
    PROFILE_PATH.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    classification = pd.read_csv(FUSION_CLASSIFICATION)
    fusion_holdout = pd.read_csv(FUSION_HOLDOUT)
    _plot_composition()
    _plot_validation(classification, fusion_holdout)
    _plot_score_distribution(scores)
    _plot_directions(directions)

    artifacts = [
        PROFILE_PATH,
        SCORES_PATH,
        DIRECTIONS_PATH,
        *sorted(FIGURES.glob("*")),
    ]
    summary = {
        "generated_at": date.today().isoformat(),
        "fingerprint_id": profile["fingerprint_id"],
        "profile_sha256": _sha256(PROFILE_PATH),
        "contains_tda_features": False,
        "primary_representation": "L+P",
        "dimensions": int(lp.shape[1]),
        "validation_180": validation_metrics,
        "opened_holdout_180_descriptive": holdout_metrics,
        "phase_increment": fusion_summary["primary_180"]["increments"]["LP_minus_L"],
        "structure_increment": fusion_summary["primary_180"]["increments"]["LPS_minus_LP"],
        "directional_signature_counts": profile["directional_signature_counts"],
        "status": profile["status"],
        "artifacts": [path.relative_to(ROOT).as_posix() for path in artifacts],
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
