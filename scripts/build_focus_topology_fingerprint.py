from __future__ import annotations

import hashlib
import json
import os
import tomllib
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = ROOT / "runs" / ".matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from generation.topology_fingerprint import (  # noqa: E402
    CHALLENGER_FEATURES,
    CORE_FEATURES,
    SUPPORT_FEATURES,
    TopologyFingerprint,
    fit_topology_fingerprint,
    write_topology_fingerprint,
)
from tda.analysis import (  # noqa: E402
    _compute_features,
    _quality_filter,
)
from tda.analysis import (  # noqa: E402
    load_config as load_tda_config,
)

matplotlib.use("Agg")

CONFIG_PATH = ROOT / "configs" / "focus_topology_fingerprint_open_v1.toml"
FEATURE_MANIFEST = ROOT / "metadata" / "feature_segments.csv"
PHASE_PATH = ROOT / "metadata" / "phase_lifted_path_homology_features.csv"
PITCH_PATH = ROOT / "metadata" / "pitch_v2_topology_segments.csv"
RHYTHM_PATH = ROOT / "metadata" / "rhythm_topology_segments.csv"
MODEL_NPZ = ROOT / "features" / "models" / "state_model.npz"
MODEL_JSON = ROOT / "features" / "models" / "state_model.json"
PIPELINE_CONFIG = ROOT / "configs" / "pipeline.toml"

METADATA = ROOT / "metadata"
OUTPUT = ROOT / "runs" / "focus_topology_fingerprint_open_v1"
FIGURES = OUTPUT / "figures"
TDA_CACHE = METADATA / "focus_topology_fingerprint_open_v1_tda_features.csv"
TDA_CACHE_META = METADATA / "focus_topology_fingerprint_open_v1_tda_cache.json"
FINGERPRINT_PATH = METADATA / "focus_topology_fingerprint_open_v1.json"
FEATURES_PATH = METADATA / "focus_topology_fingerprint_open_v1_features.csv"
SCORES_PATH = METADATA / "focus_topology_fingerprint_open_v1_scores.csv"
DIAGNOSTICS_PATH = METADATA / "focus_topology_fingerprint_open_v1_diagnostics.csv"
TESTS_PATH = METADATA / "focus_topology_fingerprint_open_v1_tests.csv"
SUMMARY_PATH = METADATA / "focus_topology_fingerprint_open_v1_summary.json"
REPORT_PATH = ROOT / "docs" / "focus-topology-fingerprint-open-v1-analysis.md"

IDENTITY = ["segment_id", "track_id", "group", "split", "scale_seconds"]
QUANTILE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
CORE_LABELS = (
    "Acoustic novelty\nH0 max persistence",
    "Rhythm\nH0 total persistence",
    "Acoustic phase\nloop score",
    "Rhythm phase\nloop score",
)
SUPPORT_LABELS = (
    "Pitch H0\nobserved persistence",
    "Pitch\npath entropy",
    "Rhythm\nedge density",
    "Rhythm\nreciprocity",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_settings() -> dict[str, Any]:
    with CONFIG_PATH.open("rb") as handle:
        values = tomllib.load(handle)["fingerprint"]
    return values


def _tda_cache_key() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): _sha256(path)
        for path in (FEATURE_MANIFEST, MODEL_NPZ, MODEL_JSON, PIPELINE_CONFIG)
    }


def _load_or_compute_tda() -> tuple[pd.DataFrame, dict[str, str], int]:
    key = _tda_cache_key()
    if TDA_CACHE.is_file() and TDA_CACHE_META.is_file():
        cached = json.loads(TDA_CACHE_META.read_text(encoding="utf-8"))
        if cached.get("input_sha256") == key:
            frame = pd.read_csv(TDA_CACHE)
            return frame, key, int(cached.get("quality_excluded", 0))

    manifest = pd.read_csv(FEATURE_MANIFEST)
    config = load_tda_config(ROOT)
    requested = ("acoustic_novelty_delay", "rhythm")
    eligible, excluded = _quality_filter(manifest, requested, config)
    frame = _compute_features(ROOT, eligible, config, requested)
    frame.to_csv(TDA_CACHE, index=False, encoding="utf-8", lineterminator="\n")
    TDA_CACHE_META.write_text(
        json.dumps(
            {
                "generated_at": date.today().isoformat(),
                "input_sha256": key,
                "quality_excluded": int(len(excluded)),
                "rows": int(len(frame)),
                "representations": list(requested),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return frame, key, int(len(excluded))


def _assemble_features(tda: pd.DataFrame) -> pd.DataFrame:
    identity_frame = pd.read_csv(FEATURE_MANIFEST).loc[:, IDENTITY]
    if identity_frame.duplicated(IDENTITY).any():
        raise RuntimeError("feature manifest has duplicate identities")
    combined = identity_frame.set_index(IDENTITY).sort_index()

    tda_mapping = {
        ("acoustic_novelty_delay", "h0_max_persistence"): CORE_FEATURES[0],
        ("rhythm", "h0_total_persistence"): CORE_FEATURES[1],
    }
    for (representation, metric), output_name in tda_mapping.items():
        subset = tda[tda["representation"] == representation].set_index(IDENTITY)
        combined[output_name] = subset[metric].reindex(combined.index)

    phase = pd.read_csv(PHASE_PATH)
    phase_pivot = phase.pivot(
        index=IDENTITY,
        columns="representation",
        values="loop_score",
    )
    phase_mapping = {
        "path_acoustic_phase": CORE_FEATURES[2],
        "path_rhythm_phase": CORE_FEATURES[3],
        "path_chroma_phase": CHALLENGER_FEATURES[0],
    }
    for representation, output_name in phase_mapping.items():
        combined[output_name] = phase_pivot[representation].reindex(combined.index)

    pitch = pd.read_csv(PITCH_PATH).set_index(IDENTITY).sort_index()
    rhythm = pd.read_csv(RHYTHM_PATH).set_index(IDENTITY).sort_index()
    combined[SUPPORT_FEATURES[0]] = pitch["h0_observed_persistence"].reindex(combined.index)
    combined[SUPPORT_FEATURES[1]] = pitch["path_entropy"].reindex(combined.index)
    combined[SUPPORT_FEATURES[2]] = rhythm["edge_density"].reindex(combined.index)
    combined[SUPPORT_FEATURES[3]] = rhythm["reciprocity"].reindex(combined.index)
    return combined.reset_index()


def _fit_fingerprint(
    features: pd.DataFrame,
    settings: dict[str, Any],
    tda_hashes: dict[str, str],
) -> TopologyFingerprint:
    reference = features[
        (features["group"] == settings["reference_group"])
        & (features["split"] == settings["reference_split"])
        & np.isclose(features["scale_seconds"], settings["reference_scale_seconds"])
    ].sort_values("segment_id")
    required = [*CORE_FEATURES, *SUPPORT_FEATURES, *CHALLENGER_FEATURES]
    if reference[required].isna().any().any():
        raise RuntimeError("reference Focus sample contains missing fingerprint features")
    if len(reference) != 195:
        raise RuntimeError(f"expected 195 discovery Focus rows, found {len(reference)}")

    source_paths = (PHASE_PATH, PITCH_PATH, RHYTHM_PATH, FEATURE_MANIFEST, CONFIG_PATH)
    source_hashes = {
        path.relative_to(ROOT).as_posix(): _sha256(path) for path in source_paths
    }
    source_hashes[TDA_CACHE.relative_to(ROOT).as_posix()] = _sha256(TDA_CACHE)
    source_hashes.update(tda_hashes)
    model_payload = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    frozen_hashes = {
        "state_model_npz_sha256": _sha256(MODEL_NPZ),
        "state_model_json_sha256": _sha256(MODEL_JSON),
        "state_model_declared_sha256": str(model_payload["model_sha256"]),
        "pipeline_config_sha256": _sha256(PIPELINE_CONFIG),
        "fingerprint_config_sha256": _sha256(CONFIG_PATH),
    }
    return fit_topology_fingerprint(
        reference.loc[:, CORE_FEATURES].to_numpy(float),
        reference.loc[:, SUPPORT_FEATURES].to_numpy(float),
        reference.loc[:, CHALLENGER_FEATURES].to_numpy(float),
        fingerprint_id=str(settings["fingerprint_id"]),
        reference_segment_ids=tuple(reference["segment_id"].astype(str)),
        covariance_shrinkage=float(settings["covariance_shrinkage"]),
        core_radius_quantile=float(settings["core_radius_quantile"]),
        support_lower_quantile=float(settings["support_lower_quantile"]),
        support_upper_quantile=float(settings["support_upper_quantile"]),
        source_sha256=source_hashes,
        frozen_hashes=frozen_hashes,
    )


def _score_features(
    features: pd.DataFrame,
    fingerprint: TopologyFingerprint,
) -> pd.DataFrame:
    scored = features.copy()
    complete = scored.loc[:, CORE_FEATURES].notna().all(axis=1)
    scored["core_distance"] = np.nan
    scored["core_inside_r90"] = pd.NA
    scored["core_shell_loss"] = np.nan
    core = scored.loc[complete, CORE_FEATURES].to_numpy(float)
    distances = fingerprint.distance(core)
    scored.loc[complete, "core_distance"] = distances
    scored.loc[complete, "core_inside_r90"] = distances <= fingerprint.core_radius
    scored.loc[complete, "core_shell_loss"] = fingerprint.core_shell_loss(core)

    support_complete = scored.loc[:, SUPPORT_FEATURES].notna().all(axis=1)
    scored["support_band_loss"] = np.nan
    scored.loc[support_complete, "support_band_loss"] = fingerprint.support_band_loss(
        scored.loc[support_complete, SUPPORT_FEATURES].to_numpy(float)
    )
    return scored


def _diagnostics(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, scale, group), frame in scored.groupby(
        ["split", "scale_seconds", "group"], sort=True
    ):
        distance = frame["core_distance"].dropna().to_numpy(float)
        inside = frame["core_inside_r90"].dropna().astype(bool).to_numpy()
        support = frame["support_band_loss"].dropna().to_numpy(float)
        rows.append(
            {
                "split": split,
                "scale_seconds": scale,
                "group": group,
                "n_total": len(frame),
                "n_core_complete": len(distance),
                "core_distance_q25": float(np.quantile(distance, 0.25)),
                "core_distance_median": float(np.median(distance)),
                "core_distance_q75": float(np.quantile(distance, 0.75)),
                "core_inside_r90_rate": float(np.mean(inside)),
                "support_band_loss_median": float(np.median(support)),
                "support_zero_loss_rate": float(np.mean(support == 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _validation_tests(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scale in (180.0, 300.0):
        subset = scored[
            (scored["split"] == "validation")
            & np.isclose(scored["scale_seconds"], scale)
        ]
        for metric, label in (
            ("core_distance", "core fingerprint distance"),
            ("support_band_loss", "local support band loss"),
        ):
            focus = subset.loc[subset["group"] == "focus", metric].dropna().to_numpy()
            classical = subset.loc[
                subset["group"] == "classical", metric
            ].dropna().to_numpy()
            statistic, p_value = mannwhitneyu(
                focus, classical, alternative="less", method="auto"
            )
            common_language = 1.0 - float(statistic) / (len(focus) * len(classical))
            rows.append(
                {
                    "analysis_role": "exploratory_validation_180"
                    if scale == 180.0
                    else "same_track_duration_sensitivity_300",
                    "scale_seconds": scale,
                    "metric": metric,
                    "alternative": f"Focus {label} < Classical {label}",
                    "n_focus": len(focus),
                    "n_classical": len(classical),
                    "focus_median": float(np.median(focus)),
                    "classical_median": float(np.median(classical)),
                    "mannwhitney_u": float(statistic),
                    "p_value_one_sided": float(p_value),
                    "probability_focus_lower": common_language,
                }
            )
    return pd.DataFrame(rows)


def _save_figure(figure: Any, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{stem}.png", dpi=240, bbox_inches="tight")
    figure.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def _plot_core_features(scored: pd.DataFrame, fingerprint: TopologyFingerprint) -> None:
    validation = scored[
        (scored["split"] == "validation") & np.isclose(scored["scale_seconds"], 180.0)
    ]
    figure, axes = plt.subplots(1, 4, figsize=(13.0, 4.2), constrained_layout=True)
    center = np.asarray(fingerprint.core_center)
    scale = np.asarray(fingerprint.core_scale)
    for index, (axis, feature, label) in enumerate(
        zip(axes, CORE_FEATURES, CORE_LABELS, strict=True)
    ):
        values = [
            (
                validation.loc[validation["group"] == group, feature]
                .dropna()
                .to_numpy()
                - center[index]
            )
            / scale[index]
            for group in ("focus", "classical")
        ]
        boxes = axis.boxplot(values, patch_artist=True, widths=0.58, showfliers=True)
        for patch, color in zip(boxes["boxes"], ("#D95F02", "#4472C4"), strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        axis.axhspan(-1.0, 1.0, color="#70AD47", alpha=0.10)
        axis.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        axis.set_xticks((1, 2), ("Focus", "Classical"))
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Robust z relative to discovery Focus")
    figure.suptitle("Core fingerprint endpoints: validation 180 s")
    _save_figure(figure, "core_endpoint_validation")


def _plot_distances(scored: pd.DataFrame, fingerprint: TopologyFingerprint) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True, constrained_layout=True)
    for axis, scale in zip(axes, (180.0, 300.0), strict=True):
        validation = scored[
            (scored["split"] == "validation")
            & np.isclose(scored["scale_seconds"], scale)
        ]
        values = [
            validation.loc[validation["group"] == group, "core_distance"].dropna().to_numpy()
            for group in ("focus", "classical")
        ]
        violins = axis.violinplot(values, positions=(1, 2), showmedians=True, widths=0.75)
        for body, color in zip(violins["bodies"], ("#D95F02", "#4472C4"), strict=True):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.45)
        axis.axhline(
            fingerprint.core_radius,
            color="#70AD47",
            linewidth=1.5,
            linestyle="--",
            label="Discovery Focus r90",
        )
        axis.set_xticks((1, 2), ("Focus", "Classical"))
        axis.set_title(f"Validation {int(scale)} s")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Shrinkage Mahalanobis distance")
    axes[1].legend(frameon=False, loc="upper right")
    figure.suptitle("Distance to the frozen Open Focus fingerprint")
    _save_figure(figure, "core_distance_validation")


def _plot_coverage(diagnostics: pd.DataFrame, fingerprint: TopologyFingerprint) -> None:
    subset = diagnostics[diagnostics["split"].isin(("discovery", "validation", "holdout"))]
    labels = [
        ("discovery", 180.0),
        ("validation", 180.0),
        ("validation", 300.0),
        ("holdout", 180.0),
    ]
    figure, axis = plt.subplots(figsize=(9.5, 4.5), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.36
    for offset, group, color in (
        (-width / 2, "focus", "#D95F02"),
        (width / 2, "classical", "#4472C4"),
    ):
        values = []
        for split, scale in labels:
            row = subset[
                (subset["split"] == split)
                & np.isclose(subset["scale_seconds"], scale)
                & (subset["group"] == group)
            ]
            values.append(float(row.iloc[0]["core_inside_r90_rate"]))
        bars = axis.bar(x + offset, values, width, label=group.title(), color=color)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                fontsize=8,
            )
    axis.axhline(
        fingerprint.core_radius_quantile,
        color="#70AD47",
        linewidth=1.0,
        linestyle="--",
        label="Reference target",
    )
    axis.set_xticks(
        x,
        (
            "Discovery\n180 s",
            "Validation\n180 s",
            "Validation\n300 s",
            "Holdout\n180 s",
        ),
    )
    axis.set_ylim(0.0, 1.06)
    axis.set_ylabel("Fraction inside frozen r90 ellipsoid")
    axis.set_title("Fingerprint neighborhood coverage")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)
    _save_figure(figure, "fingerprint_coverage")


def _plot_matrices(features: pd.DataFrame, fingerprint: TopologyFingerprint) -> None:
    reference = features[
        (features["group"] == "focus")
        & (features["split"] == "discovery")
        & np.isclose(features["scale_seconds"], 180.0)
    ]
    standardized = (
        reference.loc[:, CORE_FEATURES].to_numpy(float) - np.asarray(fingerprint.core_center)
    ) / np.asarray(fingerprint.core_scale)
    correlation = np.corrcoef(standardized, rowvar=False)
    precision = np.asarray(fingerprint.core_precision)
    partial = -precision / np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    np.fill_diagonal(partial, 1.0)
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), constrained_layout=True)
    image = None
    for axis, matrix, title in zip(
        axes,
        (correlation, partial),
        ("Robust-coordinate correlation", "Precision-derived partial correlation"),
        strict=True,
    ):
        image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set_xticks(range(4), ("A-H0", "R-H0", "A-phase", "R-phase"), rotation=30, ha="right")
        axis.set_yticks(range(4), ("A-H0", "R-H0", "A-phase", "R-phase"))
        axis.set_title(title)
        for i in range(4):
            for j in range(4):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    assert image is not None
    figure.colorbar(image, ax=axes, shrink=0.80)
    figure.suptitle("Frozen core dependence structure")
    _save_figure(figure, "core_dependence_matrices")


def _plot_support(scored: pd.DataFrame, fingerprint: TopologyFingerprint) -> None:
    validation = scored[
        (scored["split"] == "validation") & np.isclose(scored["scale_seconds"], 180.0)
    ]
    center = np.asarray(fingerprint.support_center)
    scale = np.asarray(fingerprint.support_scale)
    lower = (np.asarray(fingerprint.support_lower) - center) / scale
    upper = (np.asarray(fingerprint.support_upper) - center) / scale
    figure, axes = plt.subplots(1, 4, figsize=(13.0, 4.1), constrained_layout=True)
    for index, (axis, feature, label) in enumerate(
        zip(axes, SUPPORT_FEATURES, SUPPORT_LABELS, strict=True)
    ):
        values = [
            (validation.loc[validation["group"] == group, feature].to_numpy() - center[index])
            / scale[index]
            for group in ("focus", "classical")
        ]
        boxes = axis.boxplot(values, patch_artist=True, widths=0.58, showfliers=True)
        for patch, color in zip(boxes["boxes"], ("#D95F02", "#4472C4"), strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        axis.axhspan(lower[index], upper[index], color="#70AD47", alpha=0.10)
        axis.axhline(lower[index], color="#70AD47", linewidth=0.8, linestyle="--")
        axis.axhline(upper[index], color="#70AD47", linewidth=0.8, linestyle="--")
        axis.set_xticks((1, 2), ("Focus", "Classical"))
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Robust z; green = discovery Focus q10-q90")
    figure.suptitle("Local support constraints: validation 180 s")
    _save_figure(figure, "support_band_validation")


def _ellipse(axis: Any, points: np.ndarray, color: str) -> None:
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values = values[order]
    vectors = vectors[:, order]
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
    width, height = 2.0 * 2.4477 * np.sqrt(np.maximum(values, 0.0))
    center = points.mean(0)
    axis.add_patch(
        Ellipse(
            center,
            width,
            height,
            angle=angle,
            facecolor="none",
            edgecolor=color,
            linewidth=1.7,
        )
    )


def _plot_pca(scored: pd.DataFrame, fingerprint: TopologyFingerprint) -> None:
    discovery = scored[
        (scored["split"] == "discovery")
        & (scored["group"] == "focus")
        & np.isclose(scored["scale_seconds"], 180.0)
    ]
    validation = scored[
        (scored["split"] == "validation") & np.isclose(scored["scale_seconds"], 180.0)
    ].dropna(subset=list(CORE_FEATURES))
    center = np.asarray(fingerprint.core_center)
    scale = np.asarray(fingerprint.core_scale)
    train = (discovery.loc[:, CORE_FEATURES].to_numpy(float) - center) / scale
    test = (validation.loc[:, CORE_FEATURES].to_numpy(float) - center) / scale
    pca = PCA(n_components=2, random_state=20260716).fit(train)
    coordinates = pca.transform(test)
    groups = validation["group"].astype(str).to_numpy()
    figure, axis = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    for group, color, label in (
        ("focus", "#D95F02", "Open Focus"),
        ("classical", "#4472C4", "Classical"),
    ):
        points = coordinates[groups == group]
        axis.scatter(points[:, 0], points[:, 1], s=25, alpha=0.55, color=color, label=label)
        _ellipse(axis, points, color)
        axis.scatter(*points.mean(0), marker="X", s=85, color=color)
    variance = pca.explained_variance_ratio_
    axis.axhline(0.0, color="#BBBBBB", linewidth=0.7)
    axis.axvline(0.0, color="#BBBBBB", linewidth=0.7)
    axis.set_xlabel(f"PC1 ({variance[0] * 100:.1f}%)")
    axis.set_ylabel(f"PC2 ({variance[1] * 100:.1f}%)")
    axis.set_title("Validation 180 s in the frozen four-endpoint space")
    axis.legend(frameon=False)
    _save_figure(figure, "core_fingerprint_pca")


def main() -> int:
    settings = _load_settings()
    tda, tda_hashes, quality_excluded = _load_or_compute_tda()
    features = _assemble_features(tda)
    fingerprint = _fit_fingerprint(features, settings, tda_hashes)
    write_topology_fingerprint(FINGERPRINT_PATH, fingerprint)
    scored = _score_features(features, fingerprint)
    diagnostics = _diagnostics(scored)
    tests = _validation_tests(scored)

    features.to_csv(FEATURES_PATH, index=False, encoding="utf-8", lineterminator="\n")
    scored.to_csv(SCORES_PATH, index=False, encoding="utf-8", lineterminator="\n")
    diagnostics.to_csv(DIAGNOSTICS_PATH, index=False, encoding="utf-8", lineterminator="\n")
    tests.to_csv(TESTS_PATH, index=False, encoding="utf-8", lineterminator="\n")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    _plot_core_features(scored, fingerprint)
    _plot_distances(scored, fingerprint)
    _plot_coverage(diagnostics, fingerprint)
    _plot_matrices(features, fingerprint)
    _plot_support(scored, fingerprint)
    _plot_pca(scored, fingerprint)

    primary_tests = tests[tests["scale_seconds"] == 180.0]
    sensitivity_tests = tests[tests["scale_seconds"] == 300.0]
    primary_core = primary_tests[primary_tests["metric"] == "core_distance"].iloc[0]
    validation_180 = diagnostics[
        (diagnostics["split"] == "validation")
        & np.isclose(diagnostics["scale_seconds"], 180.0)
    ].set_index("group")
    holdout_180 = diagnostics[
        (diagnostics["split"] == "holdout")
        & np.isclose(diagnostics["scale_seconds"], 180.0)
    ].set_index("group")
    artifacts = [
        REPORT_PATH,
        FINGERPRINT_PATH,
        TDA_CACHE,
        TDA_CACHE_META,
        FEATURES_PATH,
        SCORES_PATH,
        DIAGNOSTICS_PATH,
        TESTS_PATH,
        *sorted(FIGURES.glob("*")),
    ]
    summary = {
        "generated_at": date.today().isoformat(),
        "fingerprint_id": fingerprint.fingerprint_id,
        "scope": "current 195-track discovery Open Focus topology fingerprint",
        "evidence_role": fingerprint.evidence_role,
        "legacy_profile_preserved": "runs/ace_rerank/ace_rerank_180s_v2/target_profile.json",
        "design": {
            "core_features": list(CORE_FEATURES),
            "support_features": list(SUPPORT_FEATURES),
            "challenger_features": list(CHALLENGER_FEATURES),
            "covariance_shrinkage": fingerprint.covariance_shrinkage,
            "core_radius_quantile": fingerprint.core_radius_quantile,
            "core_radius": fingerprint.core_radius,
            "support_band": [
                fingerprint.support_lower_quantile,
                fingerprint.support_upper_quantile,
            ],
        },
        "sample_counts": {
            "reference_focus_discovery_180": fingerprint.reference_sample_count,
            "feature_rows": len(features),
            "tda_rows": len(tda),
            "tda_quality_excluded": quality_excluded,
        },
        "primary_validation_180": primary_tests.to_dict(orient="records"),
        "duration_sensitivity_300": sensitivity_tests.to_dict(orient="records"),
        "validation_180_diagnostics": validation_180.to_dict(orient="index"),
        "holdout_180_descriptive": holdout_180.to_dict(orient="index"),
        "holdout_used_for_fitting_or_selection": False,
        "guidance_qualification": {
            "criterion": "validation/180 Focus median core distance < Classical median",
            "criterion_met": bool(
                primary_core["focus_median"] < primary_core["classical_median"]
            ),
            "status": "not_qualified_for_inference_guidance",
            "reason": (
                "Classical validation segments are closer to the frozen four-endpoint "
                "Open Focus center than Open Focus validation segments"
            ),
            "weights_or_features_changed_after_validation": False,
        },
        "input_sha256": fingerprint.source_sha256,
        "frozen_hashes": fingerprint.frozen_hashes,
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
