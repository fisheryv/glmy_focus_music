# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.analysis_inputs import audit_analysis_inputs
from features.batch import _sha256, _write_json_atomic
from repetition.analysis import (
    PATH_REPRESENTATIONS,
    _calibration_tests,
    _candidate_data,
    _compute_features,
    _dominant_lag_from_distance,
    _load_model,
    _quality_filter,
    _standard_distance,
    load_config,
    transposition_invariant_chroma_distance,
)
from topology.statistics import benjamini_hochberg

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "metadata"
FEATURE_PATH = METADATA / "phase_lifted_path_homology_features.csv"
TEST_PATH = METADATA / "phase_lifted_path_homology_tests.csv"
CALIBRATION_PATH = METADATA / "phase_lifted_path_homology_calibration.csv"
STABILITY_PATH = METADATA / "phase_lifted_path_homology_scale_stability.csv"
CLASSIFICATION_PATH = METADATA / "phase_lifted_path_homology_classification.csv"
REPRESENTATIVE_PATH = METADATA / "phase_lifted_path_homology_representative_edges.csv"
EXCLUSION_PATH = METADATA / "phase_lifted_path_homology_exclusions.csv"
SUMMARY_PATH = METADATA / "phase_lifted_path_homology_summary.json"

def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    temporary.replace(path)


def _balanced_calibration_manifest(
    manifest: pd.DataFrame, tracks_per_group: int, seed: int
) -> pd.DataFrame:
    base = manifest[(manifest["split"] == "discovery") & (manifest["scale_seconds"] == 180.0)]
    sensitivity = manifest[
        (manifest["split"] == "discovery") & (manifest["scale_seconds"] == 300.0)
    ]
    base = base[base["track_id"].isin(sensitivity["track_id"])]
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    for group in ("focus", "classical"):
        candidates = np.sort(base.loc[base["group"] == group, "track_id"].unique())
        if len(candidates) < tracks_per_group:
            raise RuntimeError(f"not enough eligible {group} tracks for calibration")
        chosen.extend(
            str(value) for value in rng.choice(candidates, tracks_per_group, replace=False)
        )
    return manifest[
        (manifest["split"] == "discovery")
        & manifest["track_id"].isin(chosen)
        & manifest["scale_seconds"].isin([180.0, 300.0])
    ].copy()


def _rank_biserial(first: np.ndarray, second: np.ndarray) -> float:
    statistic = mannwhitneyu(first, second, alternative="two-sided", method="auto").statistic
    return 2.0 * float(statistic) / (len(first) * len(second)) - 1.0


def _bootstrap_effect_interval(
    first: np.ndarray, second: np.ndarray, *, seed: int, repetitions: int = 3000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    effects = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        left = rng.choice(first, len(first), replace=True)
        right = rng.choice(second, len(second), replace=True)
        effects[index] = np.mean(left[:, None] > right[None, :]) - np.mean(
            left[:, None] < right[None, :]
        )
    lower, upper = np.quantile(effects, [0.025, 0.975])
    return float(lower), float(upper)


def _comparison_tests(features: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    roles = (
        ("discovery_descriptive", "discovery", 180.0),
        ("primary_validation", "validation", 180.0),
        ("duration_sensitivity", "validation", 300.0),
    )
    for role, split, scale in roles:
        subset = features[(features["split"] == split) & (features["scale_seconds"] == scale)]
        for representation in PATH_REPRESENTATIONS:
            view = subset[subset["representation"] == representation]
            focus = view.loc[view["group"] == "focus", "loop_score"].to_numpy(float)
            classical = view.loc[view["group"] == "classical", "loop_score"].to_numpy(float)
            two_sided = mannwhitneyu(focus, classical, alternative="two-sided", method="auto")
            greater = mannwhitneyu(focus, classical, alternative="greater", method="auto")
            effect = 2.0 * float(two_sided.statistic) / (len(focus) * len(classical)) - 1.0
            digest = hashlib.sha256(f"{seed}:{role}:{representation}".encode()).digest()
            ci_low, ci_high = _bootstrap_effect_interval(
                focus,
                classical,
                seed=int.from_bytes(digest[:8], "little"),
            )
            rows.append(
                {
                    "role": role,
                    "split": split,
                    "scale_seconds": scale,
                    "representation": representation,
                    "method": "phase_lifted_path_homology",
                    "n_focus": len(focus),
                    "n_classical": len(classical),
                    "focus_median": float(np.median(focus)),
                    "classical_median": float(np.median(classical)),
                    "rank_biserial_focus_minus_classical": effect,
                    "effect_ci95_low": ci_low,
                    "effect_ci95_high": ci_high,
                    "p_two_sided": float(two_sided.pvalue),
                    "p_focus_greater": float(greater.pvalue),
                }
            )
    result = pd.DataFrame(rows)
    result["p_two_sided_fdr_bh"] = np.nan
    result["p_focus_greater_fdr_bh"] = np.nan
    for role, indices in result.groupby("role").groups.items():
        del role
        result.loc[indices, "p_two_sided_fdr_bh"] = benjamini_hochberg(
            result.loc[indices, "p_two_sided"].to_numpy(float)
        )
        result.loc[indices, "p_focus_greater_fdr_bh"] = benjamini_hochberg(
            result.loc[indices, "p_focus_greater"].to_numpy(float)
        )
    return result.sort_values(["role", "p_two_sided_fdr_bh", "representation"])


def _scale_stability(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ("discovery", "validation", "holdout"):
        for representation in PATH_REPRESENTATIONS:
            subset = features[
                (features["split"] == split) & (features["representation"] == representation)
            ]
            for group in ("focus", "classical"):
                paired = subset[subset["group"] == group].pivot(
                    index="track_id", columns="scale_seconds", values="loop_score"
                )
                paired = paired.dropna(subset=[180.0, 300.0])
                correlation = spearmanr(paired[180.0], paired[300.0])
                differences = paired[300.0] - paired[180.0]
                if np.allclose(differences, 0.0):
                    signed_p = 1.0
                else:
                    signed_p = float(wilcoxon(differences, alternative="two-sided").pvalue)
                rows.append(
                    {
                        "split": split,
                        "group": group,
                        "representation": representation,
                        "n_tracks": len(paired),
                        "spearman_rho_180_vs_300": float(correlation.statistic),
                        "spearman_p": float(correlation.pvalue),
                        "median_300_minus_180": float(np.median(differences)),
                        "wilcoxon_p_two_sided": signed_p,
                    }
                )
    return pd.DataFrame(rows)


def _classification(features: pd.DataFrame, seed: int) -> pd.DataFrame:
    wide = features.pivot(
        index=["segment_id", "track_id", "group", "split", "scale_seconds"],
        columns="representation",
        values="loop_score",
    ).reset_index()
    columns = list(PATH_REPRESENTATIONS)
    subset = wide[wide["scale_seconds"] == 180.0]
    train = subset[subset["split"] == "discovery"]
    validation = subset[subset["split"] == "validation"]
    y_train = (train["group"] == "focus").astype(int).to_numpy()
    y_validation = (validation["group"] == "focus").astype(int).to_numpy()
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(
        pipeline,
        {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        scoring="f1_macro",
        cv=folds,
        n_jobs=1,
    ).fit(train[columns], y_train)
    predictions = search.predict(validation[columns])
    probabilities = search.predict_proba(validation[columns])[:, 1]
    return pd.DataFrame(
        [
            {
                "task": "open_focus_vs_classical",
                "role": "auxiliary_not_primary",
                "scale_seconds": 180.0,
                "n_train": len(train),
                "n_validation": len(validation),
                "n_features": len(columns),
                "features": ";".join(columns),
                "best_c": float(search.best_params_["classifier__C"]),
                "cv_macro_f1": float(search.best_score_),
                "balanced_accuracy": float(balanced_accuracy_score(y_validation, predictions)),
                "macro_f1": float(f1_score(y_validation, predictions, average="macro")),
                "auroc": float(roc_auc_score(y_validation, probabilities)),
            }
        ]
    )


def _phase_edge_details(
    values: np.ndarray,
    hop_seconds: float,
    transposition_invariant: bool,
    phase_bins: int,
    config,
) -> tuple[int, np.ndarray, np.ndarray]:
    distances = (
        transposition_invariant_chroma_distance(values)
        if transposition_invariant
        else _standard_distance(values)
    )
    upper = distances[np.triu_indices(len(distances), k=3)]
    positive = upper[upper > 1e-9]
    scale = float(np.median(positive)) if positive.size else 1.0
    period, _ = _dominant_lag_from_distance(distances, config)
    recurrence = np.exp(-np.diag(distances, k=period) / max(scale, 1e-8))
    phase = np.arange(len(recurrence)) % period * phase_bins // period
    coherence = np.asarray([np.mean(recurrence[phase == index]) for index in range(phase_bins)])
    edge_weights = np.minimum(coherence, np.roll(coherence, -1))
    return period, coherence, edge_weights


def _representative_edges(manifest: pd.DataFrame, features: pd.DataFrame, config) -> pd.DataFrame:
    model = _load_model(ROOT)
    rows: list[dict[str, object]] = []
    target_representation = "path_rhythm_phase"
    view = features[
        (features["split"] == "validation")
        & (features["scale_seconds"] == 180.0)
        & (features["representation"] == target_representation)
    ]
    for group in ("focus", "classical"):
        group_view = view[view["group"] == group].copy()
        median = float(group_view["loop_score"].median())
        chosen = group_view.iloc[
            int(np.argmin(np.abs(group_view["loop_score"].to_numpy(float) - median)))
        ]
        manifest_row = manifest[manifest["segment_id"] == chosen["segment_id"]].iloc[0].to_dict()
        values, hop_seconds, transposition_invariant = _candidate_data(
            ROOT, manifest_row, model, config
        )[target_representation]
        period, coherence, weights = _phase_edge_details(
            values,
            hop_seconds,
            transposition_invariant,
            config.phase_bins,
            config,
        )
        for source, weight in enumerate(weights):
            rows.append(
                {
                    "group": group,
                    "representation": target_representation,
                    "segment_id": chosen["segment_id"],
                    "track_id": chosen["track_id"],
                    "source_phase": source,
                    "target_phase": (source + 1) % config.phase_bins,
                    "source_coherence": float(coherence[source]),
                    "target_coherence": float(coherence[(source + 1) % config.phase_bins]),
                    "edge_weight": float(weight),
                    "dominant_period_blocks": period,
                    "dominant_period_seconds": float(period * hop_seconds),
                    "loop_score": float(np.min(weights)),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    input_audit = audit_analysis_inputs(root=ROOT)
    config = load_config(ROOT)
    manifest_path = METADATA / "feature_segments.csv"
    manifest = pd.read_csv(manifest_path)
    groups = set(manifest["group"].astype(str).unique())
    if groups != {"focus", "classical"}:
        raise RuntimeError(f"expected current two-group manifest, found {sorted(groups)}")
    eligible, excluded = _quality_filter(manifest, PATH_REPRESENTATIONS, config)
    features = _compute_features(
        ROOT,
        eligible,
        config,
        PATH_REPRESENTATIONS,
        calibrate=False,
    )
    calibration_manifest = _balanced_calibration_manifest(
        eligible, config.exploration_tracks_per_group, config.random_seed
    )
    calibration_features = _compute_features(
        ROOT,
        calibration_manifest,
        config,
        PATH_REPRESENTATIONS,
        calibrate=True,
    )
    calibration = _calibration_tests(calibration_features, config)
    tests = _comparison_tests(features, config.random_seed)
    stability = _scale_stability(features)
    classification = _classification(features, config.random_seed)
    representative_edges = _representative_edges(eligible, features, config)

    _write_csv(FEATURE_PATH, features)
    _write_csv(TEST_PATH, tests)
    _write_csv(CALIBRATION_PATH, calibration)
    _write_csv(STABILITY_PATH, stability)
    _write_csv(CLASSIFICATION_PATH, classification)
    _write_csv(REPRESENTATIVE_PATH, representative_edges)
    _write_csv(EXCLUSION_PATH, excluded)

    artifacts = [
        FEATURE_PATH,
        TEST_PATH,
        CALIBRATION_PATH,
        STABILITY_PATH,
        CLASSIFICATION_PATH,
        REPRESENTATIVE_PATH,
        EXCLUSION_PATH,
    ]
    primary = tests[tests["role"] == "primary_validation"]
    sensitivity = tests[tests["role"] == "duration_sensitivity"]
    payload = {
        "generated_at": date.today().isoformat(),
        "scope": "phase-lifted Path Homology analysis on current Open Focus/Classical data",
        "evidence_status": "post-migration observational reanalysis; not a replication of the former Focus-vs-Pop hypothesis",
        "ok": True,
        "input_provenance": input_audit,
        "representations": list(PATH_REPRESENTATIONS),
        "segments": int(features["segment_id"].nunique()),
        "segment_views": int(len(features)),
        "tracks": int(features["track_id"].nunique()),
        "quality_excluded_segments": int(len(excluded)),
        "group_counts_segments": {
            str(key): int(value)
            for key, value in eligible.groupby("group").size().to_dict().items()
        },
        "split_counts_segments": {
            str(key): int(value)
            for key, value in eligible.groupby("split").size().to_dict().items()
        },
        "calibration_passes": calibration.loc[
            calibration["calibration_pass"], "representation"
        ].tolist(),
        "primary_validation_fdr_discoveries": primary.loc[
            primary["p_two_sided_fdr_bh"] <= config.validation_fdr_q,
            "representation",
        ].tolist(),
        "duration_sensitivity_fdr_discoveries": sensitivity.loc[
            sensitivity["p_two_sided_fdr_bh"] <= config.validation_fdr_q,
            "representation",
        ].tolist(),
        "config": asdict(config),
        "input_sha256": {
            manifest_path.relative_to(ROOT).as_posix(): _sha256(manifest_path),
            "features/models/state_model.npz": _sha256(
                ROOT / "features" / "models" / "state_model.npz"
            ),
        },
        "outputs": [path.relative_to(ROOT).as_posix() for path in artifacts],
        "output_sha256": {path.relative_to(ROOT).as_posix(): _sha256(path) for path in artifacts},
    }
    _write_json_atomic(SUMMARY_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
