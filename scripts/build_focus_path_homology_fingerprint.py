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

from topology.multiview_fusion import DiscoveryMahalanobisBlock  # noqa: E402
from topology.statistics import TOPOLOGY_METRICS, _pseudo_f_statistic  # noqa: E402

METADATA = ROOT / "metadata"
CONFIG_PATH = ROOT / "configs" / "focus_path_homology_fingerprint_v2.toml"
PROFILE_PATH = METADATA / "focus_path_homology_fingerprint_v2.json"
SCORES_PATH = METADATA / "focus_path_homology_fingerprint_v2_scores.csv"
DIRECTIONS_PATH = METADATA / "focus_path_homology_fingerprint_v2_directions.csv"
SUMMARY_PATH = METADATA / "focus_path_homology_fingerprint_v2_summary.json"
RELEASE_PATH = METADATA / "focus_path_homology_fingerprint_v2_release.json"
OUTPUT = ROOT / "runs" / "focus_path_homology_fingerprint_v2"
FIGURES = OUTPUT / "figures"

PITCH_FILE = METADATA / "pitch_v2_topology_segments.csv"
PHASE_FILE = METADATA / "phase_lifted_path_homology_features.csv"
PITCH_TESTS = METADATA / "pitch_v2_statistical_tests.csv"
PHASE_TESTS = METADATA / "phase_lifted_path_homology_tests.csv"
FROZEN_EVIDENCE = METADATA / "pitch_phase_hierarchical_summary.json"

IDENTITY = ["segment_id", "track_id", "group", "split", "scale_seconds"]
PHASE_VIEWS = ("path_acoustic_phase", "path_chroma_phase")
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
    fusion_scale: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_sha256(values: dict[str, str]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _classifier_sha256(coef: list[float], intercept: float) -> str:
    payload = json.dumps(
        {"coef": coef, "intercept": intercept}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, path)


def _serialize_block(
    transformer: DiscoveryMahalanobisBlock,
    input_features: list[str],
    fusion_scale: float,
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
        fusion_scale=float(fusion_scale),
    )


def _mask(identity: pd.DataFrame, split: str, scale: float = 180.0) -> np.ndarray:
    return (identity["split"].astype(str).to_numpy() == split) & np.isclose(
        identity["scale_seconds"].to_numpy(float), scale
    )


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pitch = pd.read_csv(PITCH_FILE)
    required_pitch = set(IDENTITY) | set(TOPOLOGY_METRICS) | {"status"}
    missing_pitch = required_pitch - set(pitch.columns)
    if missing_pitch:
        raise RuntimeError(f"pitch file is missing columns: {sorted(missing_pitch)}")
    if pitch.duplicated(IDENTITY).any() or (pitch["status"] == "failed").any():
        raise RuntimeError("pitch input has duplicate identities or failed rows")
    pitch_indexed = pitch.set_index(IDENTITY).sort_index()
    pitch_numeric = pitch_indexed.loc[:, TOPOLOGY_METRICS].apply(
        pd.to_numeric, errors="coerce"
    )
    if pitch_numeric.isna().any().any():
        raise RuntimeError("pitch input contains missing topology metrics")

    phase = pd.read_csv(PHASE_FILE)
    required_phase = set(IDENTITY) | {"representation", "loop_score"}
    missing_phase = required_phase - set(phase.columns)
    if missing_phase:
        raise RuntimeError(f"phase file is missing columns: {sorted(missing_phase)}")
    if phase.duplicated([*IDENTITY, "representation"]).any():
        raise RuntimeError("phase input has duplicate identity/representation rows")
    phase_pivot = phase.pivot(
        index=IDENTITY, columns="representation", values="loop_score"
    ).reindex(pitch_indexed.index)
    if phase_pivot.loc[:, list(PHASE_VIEWS)].isna().sum().max() > 2:
        raise RuntimeError("unexpected phase missingness")

    identity = pitch_indexed.index.to_frame(index=False)
    return identity, pitch_numeric, phase_pivot


def _fit_coordinates(
    identity: pd.DataFrame,
    pitch: pd.DataFrame,
    phase: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, SerializedBlock]]:
    discovery = _mask(identity, "discovery")
    # The signed v2 contract reserves 16 Pitch coordinates. The discovery
    # covariance has effective rank 13; the remaining three coordinates are
    # explicit zeros rather than platform-dependent numerical null-space noise.
    pitch_transform = DiscoveryMahalanobisBlock(output_dimensions=16).fit(
        pitch.to_numpy(float)[discovery]
    )
    pitch_block = pitch_transform.transform(pitch.to_numpy(float))

    phase_blocks: list[np.ndarray] = []
    phase_transforms: dict[str, DiscoveryMahalanobisBlock] = {}
    for view in PHASE_VIEWS:
        raw = phase.loc[:, [view]].to_numpy(float)
        transformer = DiscoveryMahalanobisBlock().fit(raw[discovery])
        phase_blocks.append(transformer.transform(raw))
        phase_transforms[view] = transformer

    phase_block = np.concatenate(phase_blocks, axis=1) / np.sqrt(2.0)
    coordinates = np.concatenate(
        [pitch_block / np.sqrt(2.0), phase_block / np.sqrt(2.0)], axis=1
    )
    serialized = {
        "pitch": _serialize_block(
            pitch_transform, list(TOPOLOGY_METRICS), fusion_scale=1.0 / np.sqrt(2.0)
        ),
        PHASE_VIEWS[0]: _serialize_block(
            phase_transforms[PHASE_VIEWS[0]], ["loop_score"], fusion_scale=0.5
        ),
        PHASE_VIEWS[1]: _serialize_block(
            phase_transforms[PHASE_VIEWS[1]], ["loop_score"], fusion_scale=0.5
        ),
    }
    if pitch_block.shape[1] != 16 or phase_block.shape[1] != 2 or coordinates.shape[1] != 18:
        raise RuntimeError(
            "frozen dimensions changed: expected Pitch=16, Phase=2, joint=18"
        )
    return pitch_block, phase_block, coordinates, serialized


def _build_directions() -> pd.DataFrame:
    pitch = pd.read_csv(PITCH_TESTS)
    pitch = pitch[
        (pitch["analysis_set"] == "primary_validation_180")
        & (pitch["p_fdr_bh"] <= 0.05)
    ]
    rows: list[dict[str, Any]] = []
    for item in pitch.itertuples(index=False):
        rows.append(
            {
                "layer": "pitch",
                "view": "pitch",
                "metric": item.metric,
                "expected_focus_direction": "greater"
                if item.focus_median > item.classical_median
                else "less",
                "validation_classical_median": item.classical_median,
                "validation_focus_median": item.focus_median,
                "validation_p_fdr_bh": item.p_fdr_bh,
                "evidence_role": "validation_180_confirmatory_pitch",
            }
        )

    phase = pd.read_csv(PHASE_TESTS)
    phase = phase[
        (phase["role"] == "primary_validation")
        & np.isclose(phase["scale_seconds"], 180.0)
        & phase["representation"].isin(PHASE_VIEWS)
    ]
    for item in phase.itertuples(index=False):
        rows.append(
            {
                "layer": "phase",
                "view": item.representation,
                "metric": "loop_score",
                "expected_focus_direction": "greater",
                "validation_classical_median": item.classical_median,
                "validation_focus_median": item.focus_median,
                "validation_p_fdr_bh": item.p_focus_greater_fdr_bh,
                "evidence_role": "post_validation_refreeze_phase",
            }
        )
    return pd.DataFrame(rows).sort_values(["layer", "view", "metric"]).reset_index(drop=True)


def _validation_reproduction(
    identity: pd.DataFrame,
    pitch_block: np.ndarray,
    phase_block: np.ndarray,
    coordinates: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    frozen: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    validation = _mask(identity, "validation")
    labels = identity.loc[validation, "group"].astype(str).to_numpy()
    binary = (labels == "focus").astype(int)
    observed = {
        "pitch_pseudo_f": float(_pseudo_f_statistic(pitch_block[validation], labels)),
        "phase_pseudo_f": float(_pseudo_f_statistic(phase_block[validation], labels)),
        "joint_pseudo_f": float(_pseudo_f_statistic(coordinates[validation], labels)),
        "joint_balanced_accuracy": float(
            balanced_accuracy_score(binary, (predictions[validation] == "focus").astype(int))
        ),
        "joint_auroc": float(roc_auc_score(binary, probabilities[validation])),
    }
    observed["joint_minus_pitch_delta_pseudo_f"] = (
        observed["joint_pseudo_f"] - observed["pitch_pseudo_f"]
    )
    observed["joint_minus_phase_delta_pseudo_f"] = (
        observed["joint_pseudo_f"] - observed["phase_pseudo_f"]
    )
    expected_primary = frozen["primary_180"]
    expected = {
        "pitch_pseudo_f": expected_primary["permanova"]["Pitch"]["pseudo_f"],
        "phase_pseudo_f": expected_primary["permanova"]["Phase"]["pseudo_f"],
        "joint_pseudo_f": expected_primary["permanova"]["PitchPhase"]["pseudo_f"],
        "joint_balanced_accuracy": expected_primary["classification"]["PitchPhase"][
            "balanced_accuracy"
        ],
        "joint_auroc": expected_primary["classification"]["PitchPhase"]["auroc"],
        "joint_minus_pitch_delta_pseudo_f": expected_primary["increments"][
            "PitchPhase_minus_Pitch"
        ]["delta_pseudo_f"],
        "joint_minus_phase_delta_pseudo_f": expected_primary["increments"][
            "PitchPhase_minus_Phase"
        ]["delta_pseudo_f"],
    }
    absolute_error = {name: abs(observed[name] - float(value)) for name, value in expected.items()}
    failures = {name: error for name, error in absolute_error.items() if error > tolerance}
    if failures:
        raise RuntimeError(f"18-D frozen validation reproduction failed: {failures}")
    return {
        "status": "passed",
        "tolerance": tolerance,
        "observed": observed,
        "expected": expected,
        "absolute_error": absolute_error,
        "frozen_permutation_evidence": {
            "pitch_q": expected_primary["permanova"]["Pitch"]["p_fdr_bh"],
            "phase_q": expected_primary["permanova"]["Phase"]["p_fdr_bh"],
            "joint_q": expected_primary["permanova"]["PitchPhase"]["p_fdr_bh"],
            "phase_added_to_pitch_q": expected_primary["increments"][
                "PitchPhase_minus_Pitch"
            ]["p_fdr_bh"],
            "pitch_added_to_phase_q": expected_primary["increments"][
                "PitchPhase_minus_Phase"
            ]["p_fdr_bh"],
        },
    }


def _save_figure(figure: Any, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _box(
    axis: Any,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    face: str,
    edge: str,
) -> None:
    axis.add_patch(
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
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        linespacing=1.35,
        color="#0F172A",
    )


def _arrow(axis: Any, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color=COLORS["gray"],
        )
    )


def _routed_arrow(axis: Any, points: list[tuple[float, float]]) -> None:
    for start, end in zip(points[:-2], points[1:-1], strict=True):
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=COLORS["gray"],
            linewidth=1.4,
        )
    _arrow(axis, points[-2], points[-1])


def _plot_composition() -> None:
    figure, axis = plt.subplots(figsize=(11.5, 4.6))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    _box(
        axis,
        (0.03, 0.64),
        0.22,
        0.20,
        "Pitch Path Homology\n16 coordinates / rank 13",
        COLORS["pale_blue"],
        COLORS["blue"],
    )
    _box(
        axis,
        (0.28, 0.48),
        0.20,
        0.16,
        "Acoustic phase\nloop_score",
        COLORS["pale_purple"],
        COLORS["purple"],
    )
    _box(
        axis,
        (0.28, 0.22),
        0.20,
        0.16,
        "Chroma phase\nloop_score",
        COLORS["pale_purple"],
        COLORS["purple"],
    )
    _box(
        axis,
        (0.56, 0.28),
        0.18,
        0.24,
        "Phase block P\n2 coordinates\nequal block",
        COLORS["pale_orange"],
        COLORS["orange"],
    )
    _box(
        axis,
        (0.81, 0.42),
        0.17,
        0.36,
        "Frozen scorer\n18-D\nPitch 1/2\nA 1/4 + C 1/4",
        COLORS["pale_red"],
        COLORS["red"],
    )
    _routed_arrow(axis, [(0.25, 0.74), (0.30, 0.90), (0.76, 0.90), (0.81, 0.69)])
    _arrow(axis, (0.48, 0.56), (0.56, 0.46))
    _arrow(axis, (0.48, 0.30), (0.56, 0.35))
    _arrow(axis, (0.74, 0.40), (0.81, 0.53))
    axis.text(
        0.50,
        0.12,
        "Rhythm, Modulation, Structure, Rhythm phase and TDA are rejected at runtime.",
        ha="center",
        fontsize=10.5,
        color=COLORS["gray"],
    )
    axis.set_title("focus_path_homology_fingerprint_v2 — issued 18-D composition", fontsize=15)
    figure.tight_layout()
    _save_figure(figure, "fingerprint_composition")


def _plot_validation(reproduction: dict[str, Any]) -> None:
    observed = reproduction["observed"]
    names = ["Pitch", "Phase", "Pitch + Phase"]
    values = [
        observed["pitch_pseudo_f"],
        observed["phase_pseudo_f"],
        observed["joint_pseudo_f"],
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    bars = axes[0].bar(names, values, color=[COLORS["blue"], COLORS["purple"], COLORS["red"]])
    axes[0].bar_label(bars, fmt="%.3f")
    axes[0].set_ylabel("validation/180s pseudo-F")
    axes[0].set_title("Frozen distance geometry")
    metrics = [observed["joint_balanced_accuracy"], observed["joint_auroc"]]
    metric_bars = axes[1].bar(
        ["Balanced accuracy", "AUROC"], metrics, color=[COLORS["green"], COLORS["orange"]]
    )
    axes[1].bar_label(metric_bars, fmt="%.3f")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Discovery-trained joint readout")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("18-D scorer release regression")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(figure, "fingerprint_validation")


def _plot_score_distribution(scores: pd.DataFrame) -> None:
    subset = scores[(scores["split"] == "validation") & np.isclose(scores["scale_seconds"], 180.0)]
    groups = ["focus", "classical"]
    values = [subset.loc[subset["group"] == group, "focus_probability"] for group in groups]
    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    boxes = axis.boxplot(values, tick_labels=["Open Focus", "Classical"], patch_artist=True)
    for patch, color in zip(boxes["boxes"], [COLORS["blue"], COLORS["orange"]], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    rng = np.random.default_rng(20260716)
    for index, series in enumerate(values, start=1):
        jitter = rng.normal(index, 0.035, len(series))
        axis.scatter(
            jitter, series, s=15, alpha=0.45, color=[COLORS["blue"], COLORS["orange"]][index - 1]
        )
    axis.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1.0)
    axis.set_ylabel("Discovery-trained Focus probability")
    axis.set_ylim(-0.03, 1.03)
    axis.set_title("Issued 18-D scorer on validation/180s")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, "focus_score_distribution")


def main() -> int:
    with CONFIG_PATH.open("rb") as handle:
        config = tomllib.load(handle)
    frozen = json.loads(FROZEN_EVIDENCE.read_text(encoding="utf-8"))
    identity, pitch, phase = _load_inputs()
    pitch_block, phase_block, coordinates, serialized = _fit_coordinates(identity, pitch, phase)

    configured_order = [str(value) for value in config["blocks"]["feature_order"]]
    weights = [float(value) for value in config["blocks"]["distance_weights"]]
    if len(configured_order) != 18 or weights != [0.5, 0.25, 0.25]:
        raise RuntimeError("configuration violates frozen 18-D order or distance weights")

    discovery = _mask(identity, config["fingerprint"]["reference_split"])
    labels = identity.loc[discovery, "group"].astype(str).to_numpy()
    classifier = LogisticRegression(
        C=float(config["classifier"]["c"]),
        class_weight=config["classifier"]["class_weight"],
        max_iter=int(config["classifier"]["max_iter"]),
        solver="lbfgs",
        random_state=int(config["fingerprint"]["random_seed"]),
    ).fit(coordinates[discovery], labels)
    if classifier.classes_.tolist() != ["classical", "focus"]:
        raise RuntimeError(f"unexpected class order: {classifier.classes_.tolist()}")
    logits = classifier.decision_function(coordinates)
    probabilities = classifier.predict_proba(coordinates)[:, 1]
    predictions = np.where(probabilities >= 0.5, "focus", "classical")
    focus_discovery = discovery & (identity["group"].astype(str).to_numpy() == "focus")
    focus_band_threshold = float(
        np.quantile(logits[focus_discovery], config["classifier"]["focus_target_logit_quantile"])
    )
    band_loss = np.maximum(0.0, focus_band_threshold - logits) ** 2

    reproduction = _validation_reproduction(
        identity,
        pitch_block,
        phase_block,
        coordinates,
        probabilities,
        predictions,
        frozen,
        float(config["release"]["primary_statistics_tolerance"]),
    )

    scores = identity.copy()
    for index, name in enumerate(configured_order):
        scores[name] = coordinates[:, index]
    scores["pitch_block_l2_norm"] = np.linalg.norm(pitch_block, axis=1)
    scores["phase_block_l2_norm"] = np.linalg.norm(phase_block, axis=1)
    scores["joint_l2_norm"] = np.linalg.norm(coordinates, axis=1)
    scores["focus_logit"] = logits
    scores["focus_probability"] = probabilities
    scores["focus_band_loss"] = band_loss
    scores["predicted_group"] = predictions
    _write_csv(SCORES_PATH, scores)

    directions = _build_directions()
    _write_csv(DIRECTIONS_PATH, directions)

    input_sources = {
        PITCH_FILE.relative_to(ROOT).as_posix(): _sha256(PITCH_FILE),
        PHASE_FILE.relative_to(ROOT).as_posix(): _sha256(PHASE_FILE),
    }
    source_paths = [PITCH_TESTS, PHASE_TESTS, FROZEN_EVIDENCE, CONFIG_PATH, Path(__file__)]
    source_sha256 = {
        **input_sources,
        **{path.resolve().relative_to(ROOT).as_posix(): _sha256(path) for path in source_paths},
    }
    coef = classifier.coef_[0].astype(float).tolist()
    intercept = float(classifier.intercept_[0])
    classifier_hash = _classifier_sha256(coef, intercept)
    holdout = _mask(identity, "holdout")
    holdout_labels = (identity.loc[holdout, "group"].astype(str) == "focus").astype(int)
    holdout_predictions = (predictions[holdout] == "focus").astype(int)

    profile = {
        "schema_version": 3,
        "fingerprint_id": config["fingerprint"]["fingerprint_id"],
        "spec_revision": config["fingerprint"]["spec_revision"],
        "dimensions": int(coordinates.shape[1]),
        "feature_order": configured_order,
        "distance_weights": weights,
        "scope": "Pitch 16-D plus Acoustic/Chroma phase loop scores",
        "reference_split": config["fingerprint"]["reference_split"],
        "reference_scale_seconds": float(config["fingerprint"]["reference_scale_seconds"]),
        "reference_sample_count": int(discovery.sum()),
        "reference_focus_count": int(focus_discovery.sum()),
        "contains_tda_features": False,
        "block_transforms": {name: asdict(block) for name, block in serialized.items()},
        "fusion": {
            "formula": "concat(Pitch/sqrt(2), Acoustic/2, Chroma/2)",
            "squared_distance_weights": {
                "pitch": 0.5,
                "path_acoustic_phase": 0.25,
                "path_chroma_phase": 0.25,
            },
        },
        "classifier_coef": coef,
        "classifier_intercept": intercept,
        "classifier_sha256": classifier_hash,
        "focus_band_threshold": focus_band_threshold,
        "classifier": {
            "kind": config["classifier"]["kind"],
            "classes": classifier.classes_.tolist(),
            "positive_class": "focus",
            "c": float(classifier.C),
            "decision_threshold_probability": float(config["classifier"]["decision_threshold"]),
            "focus_target_logit_quantile": float(
                config["classifier"]["focus_target_logit_quantile"]
            ),
            "control_loss": "max(0, focus_band_threshold - focus_logit)^2",
        },
        "input_sha256": _mapping_sha256(input_sources),
        "config_sha256": _sha256(CONFIG_PATH),
        "code_sha256": _sha256(Path(__file__)),
        "source_sha256": source_sha256,
        "validation_180_reproduction": reproduction,
        "opened_holdout_180_descriptive": {
            "n": int(holdout.sum()),
            "balanced_accuracy": float(
                balanced_accuracy_score(holdout_labels, holdout_predictions)
            ),
            "auroc": float(roc_auc_score(holdout_labels, probabilities[holdout])),
            "role": "opened descriptive compatibility only; not a signing gate",
        },
        "directional_signature_counts": {
            layer: int(count) for layer, count in directions.groupby("layer").size().items()
        },
        "excluded": [
            "Rhythm local block",
            "Modulation local block",
            "Rhythm phase",
            "Structure",
            "all Vietoris-Rips TDA endpoints",
            "H1/H2 as isolated directional targets",
        ],
        "legacy_51d": {
            "status": "rejected",
            "profile_sha256": config["release"]["legacy_profile_sha256"],
            "archive": config["release"]["legacy_archive"],
        },
        "runtime_status": {
            "frozen_18d_spec": "enabled",
            "legacy_51d_scorer": "reject",
            "exact_scoring": "enabled",
            "shadow_mode": "enabled",
            "experimental_reranking": "enabled_pending_separate_effect_gate",
            "ltsn_labeling": "blocked_until_exact_reranking_gate",
            "sampling_guidance": "disabled_until_all_gates_pass",
        },
    }
    if int(config["fingerprint"]["dimensions"]) != profile["dimensions"]:
        raise RuntimeError("built dimensions do not match the signed configuration")
    _write_json(PROFILE_PATH, profile)

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
    _plot_composition()
    _plot_validation(reproduction)
    _plot_score_distribution(scores)

    summary = {
        "generated_at": date.today().isoformat(),
        "fingerprint_id": profile["fingerprint_id"],
        "spec_revision": profile["spec_revision"],
        "profile_sha256": _sha256(PROFILE_PATH),
        "classifier_sha256": classifier_hash,
        "input_sha256": profile["input_sha256"],
        "config_sha256": profile["config_sha256"],
        "code_sha256": profile["code_sha256"],
        "dimensions": profile["dimensions"],
        "feature_order": configured_order,
        "distance_weights": weights,
        "block_output_dimensions": {
            name: block.output_dimensions for name, block in serialized.items()
        },
        "block_effective_ranks": {
            name: block.effective_rank for name, block in serialized.items()
        },
        "focus_band_threshold": focus_band_threshold,
        "validation_180_reproduction": reproduction,
        "opened_holdout_180_descriptive": profile["opened_holdout_180_descriptive"],
        "runtime_status": profile["runtime_status"],
        "legacy_51d": profile["legacy_51d"],
        "artifacts": [
            PROFILE_PATH.relative_to(ROOT).as_posix(),
            SCORES_PATH.relative_to(ROOT).as_posix(),
            DIRECTIONS_PATH.relative_to(ROOT).as_posix(),
            *[path.relative_to(ROOT).as_posix() for path in sorted(FIGURES.glob("*"))],
        ],
    }
    _write_json(SUMMARY_PATH, summary)

    release_artifacts = [PROFILE_PATH, SCORES_PATH, DIRECTIONS_PATH, SUMMARY_PATH]
    release = {
        "schema_version": 1,
        "release_status": "issued",
        "issued_at": date.today().isoformat(),
        "fingerprint_id": profile["fingerprint_id"],
        "spec_revision": profile["spec_revision"],
        "dimensions": profile["dimensions"],
        "feature_order": configured_order,
        "distance_weights": weights,
        "profile_sha256": _sha256(PROFILE_PATH),
        "classifier_sha256": classifier_hash,
        "input_sha256": profile["input_sha256"],
        "config_sha256": profile["config_sha256"],
        "code_sha256": profile["code_sha256"],
        "artifact_sha256": {
            path.relative_to(ROOT).as_posix(): _sha256(path) for path in release_artifacts
        },
        "signing_gates": {
            "frozen_dimensions": "passed",
            "feature_order": "passed",
            "distance_weights": "passed",
            "validation_180_reproduction": reproduction["status"],
            "legacy_51d_archived": "passed",
        },
        "runtime_status": profile["runtime_status"],
        "legacy_51d": profile["legacy_51d"],
    }
    _write_json(RELEASE_PATH, release)
    print(json.dumps(release, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
