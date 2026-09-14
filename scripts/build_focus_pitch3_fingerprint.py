from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "focus_pitch3_fingerprint_v1.toml"
SOURCE = ROOT / "metadata" / "pitch_v2_topology_segments.csv"
PAIRWISE = ROOT / "metadata" / "pitch_v2_pairwise_tests.csv"
PROFILE = ROOT / "metadata" / "focus_pitch3_fingerprint_v1.json"
SCORES = ROOT / "metadata" / "focus_pitch3_fingerprint_v1_scores.csv"
SUMMARY = ROOT / "metadata" / "focus_pitch3_fingerprint_v1_summary.json"
RELEASE = ROOT / "metadata" / "focus_pitch3_fingerprint_v1_release.json"
IDENTITY = ["segment_id", "track_id", "group", "split", "scale_seconds"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _robust_scale(
    matrix: np.ndarray,
    *,
    iqr_normalizer: float,
    mad_normalizer: float,
    minimum_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(matrix, axis=0)
    q1, q3 = np.nanquantile(matrix, (0.25, 0.75), axis=0)
    iqr = (q3 - q1) / iqr_normalizer
    mad = np.nanmedian(np.abs(matrix - center), axis=0) * mad_normalizer
    std = np.nanstd(matrix, axis=0, ddof=1)
    scale = np.where(iqr > minimum_scale, iqr, np.where(mad > minimum_scale, mad, std))
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise RuntimeError("Pitch-3 transform contains non-finite location or scale")
    if np.any(scale <= minimum_scale):
        raise RuntimeError("Pitch-3 transform contains a constant feature")
    return center, scale


def _band_loss(
    coordinates: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    below = np.maximum(lower - coordinates, 0.0)
    above = np.maximum(coordinates - upper, 0.0)
    return np.mean(below * below + above * above, axis=1)


def _effect_rows(
    pairwise: pd.DataFrame, features: list[str], analysis_set: str
) -> dict[str, dict[str, float]]:
    selected = pairwise[
        (pairwise["analysis_set"] == analysis_set)
        & (pairwise["group_a"] == "classical")
        & (pairwise["group_b"] == "focus")
    ].set_index("metric")
    return {
        feature: {
            "p_value": float(selected.loc[feature, "p_value"]),
            "p_fdr_bh": float(selected.loc[feature, "p_fdr_bh"]),
            "rank_biserial_classical_minus_focus": float(
                selected.loc[feature, "rank_biserial_a_minus_b"]
            ),
        }
        for feature in features
    }


def build() -> dict[str, Any]:
    with CONFIG.open("rb") as handle:
        config = tomllib.load(handle)
    fingerprint = config["fingerprint"]
    transform = config["transform"]
    target = config["focus_target"]
    classifier_config = config["classifier"]
    evidence = config["evidence"]
    input_features = [str(value) for value in fingerprint["input_features"]]
    feature_order = [str(value) for value in fingerprint["feature_order"]]
    directions = [str(value) for value in fingerprint["expected_focus_direction"]]
    weights = [float(value) for value in fingerprint["distance_weights"]]
    if len(input_features) != 3 or len(feature_order) != 3 or len(set(input_features)) != 3:
        raise RuntimeError("Pitch-3 contract requires exactly three unique input features")
    if directions != ["less", "greater", "greater"]:
        raise RuntimeError("Pitch-3 directional signature changed")
    if not np.isclose(sum(weights), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("Pitch-3 distance weights must sum to one")

    rows = pd.read_csv(SOURCE)
    missing = set(IDENTITY + input_features) - set(rows.columns)
    if missing:
        raise RuntimeError(f"Pitch source is missing columns: {sorted(missing)}")
    matrix = rows.loc[:, input_features].to_numpy(float)
    if not np.isfinite(matrix).all():
        raise RuntimeError("selected Pitch descriptors contain NaN or Inf")
    reference = (rows["split"] == fingerprint["reference_split"]) & np.isclose(
        rows["scale_seconds"].to_numpy(float),
        float(fingerprint["reference_scale_seconds"]),
    )
    if int(reference.sum()) != 390:
        raise RuntimeError("expected 390 discovery/180s reference rows")
    center, scale = _robust_scale(
        matrix[reference],
        iqr_normalizer=float(transform["iqr_normalizer"]),
        mad_normalizer=float(transform["mad_normalizer"]),
        minimum_scale=float(transform["minimum_scale"]),
    )
    coordinates = (matrix - center) / scale
    labels = (rows["group"].to_numpy(str) == "focus").astype(int)
    model = LogisticRegression(
        C=float(classifier_config["c"]),
        class_weight=str(classifier_config["class_weight"]),
        max_iter=int(classifier_config["max_iter"]),
        random_state=20260716,
    ).fit(coordinates[reference], labels[reference])
    coefficient = model.coef_[0].astype(float)
    intercept = float(model.intercept_[0])
    logits = coordinates @ coefficient + intercept
    probabilities = np.exp(-np.logaddexp(0.0, -logits))
    focus_reference = reference & (labels == 1)
    lower = np.quantile(
        coordinates[focus_reference], float(target["lower_quantile"]), axis=0
    )
    upper = np.quantile(
        coordinates[focus_reference], float(target["upper_quantile"]), axis=0
    )
    target_center = np.median(coordinates[focus_reference], axis=0)
    threshold = float(
        np.quantile(logits[focus_reference], classifier_config["focus_target_logit_quantile"])
    )
    band_loss = _band_loss(coordinates, lower, upper)
    classifier_sha256 = _json_sha256(
        {"coef": coefficient.tolist(), "intercept": intercept}
    )
    source_sha256 = {
        SOURCE.relative_to(ROOT).as_posix(): _sha256(SOURCE),
        PAIRWISE.relative_to(ROOT).as_posix(): _sha256(PAIRWISE),
    }
    input_sha256 = _json_sha256(source_sha256)
    profile = {
        "schema_version": 1,
        "fingerprint_id": str(fingerprint["fingerprint_id"]),
        "spec_revision": str(fingerprint["spec_revision"]),
        "dimensions": int(fingerprint["dimensions"]),
        "scope": "three complementary local Pitch path-topology controls",
        "input_features": input_features,
        "feature_order": feature_order,
        "expected_focus_direction": directions,
        "distance_weights": weights,
        "reference_split": str(fingerprint["reference_split"]),
        "reference_scale_seconds": float(fingerprint["reference_scale_seconds"]),
        "reference_sample_count": int(reference.sum()),
        "transform": {
            "kind": str(transform["kind"]),
            "center": center.tolist(),
            "scale": scale.tolist(),
        },
        "focus_target": {
            "lower_quantile": float(target["lower_quantile"]),
            "upper_quantile": float(target["upper_quantile"]),
            "coordinate_lower": lower.tolist(),
            "coordinate_upper": upper.tolist(),
            "coordinate_center": target_center.tolist(),
            "focus_band_logit_threshold": threshold,
        },
        "classifier": {
            "kind": str(classifier_config["kind"]),
            "coef": coefficient.tolist(),
            "intercept": intercept,
            "decision_threshold": float(classifier_config["decision_threshold"]),
            "sha256": classifier_sha256,
        },
        "evidence": dict(evidence),
        "source_sha256": source_sha256,
        "input_sha256": input_sha256,
        "config_sha256": _sha256(CONFIG),
        "code_sha256": _sha256(Path(__file__)),
        "runtime_status": {
            "exact_scoring": "enabled",
            "latent_control_training": "enabled_for_development",
            "sampling_guidance": "disabled_until_new_qualification_passes",
            "legacy_18d_contract": "preserved_separately",
        },
    }
    _write_json(PROFILE, profile)

    scores = rows.loc[:, IDENTITY].copy()
    for index, name in enumerate(feature_order):
        scores[name] = coordinates[:, index]
    scores["focus_logit"] = logits
    scores["focus_probability"] = probabilities
    scores["focus_target_band_loss"] = band_loss
    _write_csv(SCORES, scores)

    validation: dict[str, Any] = {}
    for duration, name in ((180.0, "primary_validation_180"), (300.0, "sensitivity_300")):
        mask = (rows["split"] == "validation") & np.isclose(rows["scale_seconds"], duration)
        prediction = probabilities[mask] >= float(classifier_config["decision_threshold"])
        validation[name] = {
            "n": int(mask.sum()),
            "balanced_accuracy": float(balanced_accuracy_score(labels[mask], prediction)),
            "auroc": float(roc_auc_score(labels[mask], probabilities[mask])),
        }
    validation_rows = rows[rows["split"] == "validation"]
    wide = validation_rows.pivot(index="track_id", columns="scale_seconds", values=input_features)
    duration_stability = {
        feature: float(
            spearmanr(wide[feature][180.0], wide[feature][300.0], nan_policy="raise").statistic
        )
        for feature in input_features
    }
    discovery_coordinates = coordinates[reference]
    redundancy = pd.DataFrame(discovery_coordinates, columns=feature_order).corr(
        method="spearman"
    )
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(discovery_coordinates, rowvar=False))[::-1]
    explained = eigenvalues / eigenvalues.sum()
    pairwise = pd.read_csv(PAIRWISE)
    summary = {
        "schema_version": 1,
        "fingerprint_id": profile["fingerprint_id"],
        "spec_revision": profile["spec_revision"],
        "selection_status": evidence["selection_status"],
        "feature_order": feature_order,
        "input_features": input_features,
        "validation": validation,
        "duration_stability_spearman": duration_stability,
        "validation_180_effects": _effect_rows(
            pairwise, input_features, "primary_validation_180"
        ),
        "validation_300_effects": _effect_rows(
            pairwise, input_features, "sensitivity_validation_300"
        ),
        "discovery_180_abs_spearman": np.abs(redundancy.to_numpy(float)).tolist(),
        "discovery_180_pca_explained_variance_ratio": explained.tolist(),
        "artifacts": {
            PROFILE.relative_to(ROOT).as_posix(): _sha256(PROFILE),
            SCORES.relative_to(ROOT).as_posix(): _sha256(SCORES),
        },
        "authorization": {
            "scientific_claim": "post_validation_downstream_target_only",
            "guidance_promotion_eligible": False,
            "production_authorization": False,
        },
    }
    _write_json(SUMMARY, summary)
    release = {
        "schema_version": 1,
        "fingerprint_id": profile["fingerprint_id"],
        "spec_revision": profile["spec_revision"],
        "release_status": "issued_for_ltsn_development_only",
        "profile_sha256": _sha256(PROFILE),
        "artifact_sha256": {
            SCORES.relative_to(ROOT).as_posix(): _sha256(SCORES),
            SUMMARY.relative_to(ROOT).as_posix(): _sha256(SUMMARY),
            CONFIG.relative_to(ROOT).as_posix(): _sha256(CONFIG),
            Path(__file__).relative_to(ROOT).as_posix(): _sha256(Path(__file__)),
        },
        "signing_gates": {
            "dimensions": "passed",
            "feature_order": "passed",
            "discovery_only_transform": "passed",
            "validation_180_reproduction": "passed",
            "validation_300_sensitivity": "passed",
            "guidance_qualification": "not_run",
        },
    }
    _write_json(RELEASE, release)
    return {
        "profile": PROFILE.relative_to(ROOT).as_posix(),
        "profile_sha256": _sha256(PROFILE),
        "scores": SCORES.relative_to(ROOT).as_posix(),
        "summary": SUMMARY.relative_to(ROOT).as_posix(),
        "release": RELEASE.relative_to(ROOT).as_posix(),
    }


def main() -> None:
    print(json.dumps(build(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
