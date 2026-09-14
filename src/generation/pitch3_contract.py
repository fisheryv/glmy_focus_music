"""Frozen three-coordinate local-Pitch topology contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ltsn_contract import LTSNContractError, sha256_file

PITCH3_FINGERPRINT_ID = "focus_pitch3_fingerprint_v1"
PITCH3_DIMENSIONS = 3
PITCH3_INPUT_FEATURES = (
    "h0_observed_persistence",
    "self_transition_ratio",
    "directed_recurrence",
)
PITCH3_FEATURE_ORDER = (
    "pitch3_h0_observed_persistence_z",
    "pitch3_self_transition_ratio_z",
    "pitch3_directed_recurrence_z",
)
PITCH3_DIRECTIONS = ("less", "greater", "greater")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _finite_vector(values: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(values, list) or len(values) != PITCH3_DIMENSIONS:
        raise LTSNContractError(f"{name} must contain exactly three values")
    vector = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in vector):
        raise LTSNContractError(f"{name} contains NaN or Inf")
    return vector  # type: ignore[return-value]


def _sha(value: Any, name: str) -> str:
    text = str(value).lower()
    if not _SHA256.fullmatch(text):
        raise LTSNContractError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _classifier_sha256(coef: Sequence[float], intercept: float) -> str:
    encoded = json.dumps(
        {"coef": list(coef), "intercept": intercept},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class Pitch3Contract:
    """Validated runtime subset of the signed Pitch-3 artifact."""

    fingerprint_id: str
    spec_revision: str
    artifact_sha256: str
    input_features: tuple[str, str, str]
    feature_order: tuple[str, str, str]
    expected_focus_direction: tuple[str, str, str]
    distance_weights: tuple[float, float, float]
    transform_center: tuple[float, float, float]
    transform_scale: tuple[float, float, float]
    target_lower: tuple[float, float, float]
    target_upper: tuple[float, float, float]
    target_center: tuple[float, float, float]
    classifier_coef: tuple[float, float, float]
    classifier_intercept: float
    focus_band_logit_threshold: float
    classifier_sha256: str
    input_sha256: str
    config_sha256: str
    code_sha256: str


def load_pitch3_contract(
    path: Path, *, expected_sha256: str | None = None
) -> Pitch3Contract:
    """Load the exact Pitch-3 contract after structural and hash validation."""

    artifact_sha256 = sha256_file(path)
    if expected_sha256 is not None and artifact_sha256 != expected_sha256.lower():
        raise LTSNContractError("Pitch-3 fingerprint JSON SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint_id") != PITCH3_FINGERPRINT_ID:
        raise LTSNContractError("unexpected Pitch-3 fingerprint_id")
    if payload.get("dimensions") != PITCH3_DIMENSIONS:
        raise LTSNContractError("Pitch-3 dimensions must equal three")
    input_features = tuple(payload.get("input_features", ()))
    feature_order = tuple(payload.get("feature_order", ()))
    directions = tuple(payload.get("expected_focus_direction", ()))
    if input_features != PITCH3_INPUT_FEATURES:
        raise LTSNContractError("Pitch-3 input feature order changed")
    if feature_order != PITCH3_FEATURE_ORDER:
        raise LTSNContractError("Pitch-3 coordinate order changed")
    if directions != PITCH3_DIRECTIONS:
        raise LTSNContractError("Pitch-3 directional signature changed")
    weights = _finite_vector(payload.get("distance_weights"), "distance_weights")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12) or any(
        value <= 0.0 for value in weights
    ):
        raise LTSNContractError("Pitch-3 distance weights must be positive and sum to one")
    transform = payload.get("transform")
    target = payload.get("focus_target")
    classifier = payload.get("classifier")
    if not isinstance(transform, Mapping) or transform.get("kind") != (
        "discovery_robust_standardization"
    ):
        raise LTSNContractError("Pitch-3 transform is missing or unsupported")
    if not isinstance(target, Mapping) or not isinstance(classifier, Mapping):
        raise LTSNContractError("Pitch-3 target or classifier is missing")
    center = _finite_vector(transform.get("center"), "transform center")
    scale = _finite_vector(transform.get("scale"), "transform scale")
    if any(value <= 0.0 for value in scale):
        raise LTSNContractError("Pitch-3 transform scale must be positive")
    lower = _finite_vector(target.get("coordinate_lower"), "target lower")
    upper = _finite_vector(target.get("coordinate_upper"), "target upper")
    target_center = _finite_vector(target.get("coordinate_center"), "target center")
    if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
        raise LTSNContractError("Pitch-3 target bands must have positive width")
    coef = _finite_vector(classifier.get("coef"), "classifier coef")
    intercept = float(classifier.get("intercept"))
    threshold = float(target.get("focus_band_logit_threshold"))
    if not math.isfinite(intercept) or not math.isfinite(threshold):
        raise LTSNContractError("Pitch-3 classifier scalar is non-finite")
    classifier_sha256 = _classifier_sha256(coef, intercept)
    if _sha(classifier.get("sha256"), "classifier sha256") != classifier_sha256:
        raise LTSNContractError("Pitch-3 classifier hash does not match coefficients")
    spec_revision = str(payload.get("spec_revision", "")).strip()
    if not spec_revision:
        raise LTSNContractError("Pitch-3 spec_revision is required")
    return Pitch3Contract(
        fingerprint_id=PITCH3_FINGERPRINT_ID,
        spec_revision=spec_revision,
        artifact_sha256=artifact_sha256,
        input_features=PITCH3_INPUT_FEATURES,
        feature_order=PITCH3_FEATURE_ORDER,
        expected_focus_direction=PITCH3_DIRECTIONS,
        distance_weights=weights,
        transform_center=center,
        transform_scale=scale,
        target_lower=lower,
        target_upper=upper,
        target_center=target_center,
        classifier_coef=coef,
        classifier_intercept=intercept,
        focus_band_logit_threshold=threshold,
        classifier_sha256=classifier_sha256,
        input_sha256=_sha(payload.get("input_sha256"), "input_sha256"),
        config_sha256=_sha(payload.get("config_sha256"), "config_sha256"),
        code_sha256=_sha(payload.get("code_sha256"), "code_sha256"),
    )


def validate_pitch3_checkpoint_metadata(
    metadata: Mapping[str, Any], contract: Pitch3Contract
) -> None:
    """Reject a control-head checkpoint detached from its exact teacher."""

    expected = {
        "fingerprint_id": contract.fingerprint_id,
        "fingerprint_spec_revision": contract.spec_revision,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "dimensions": PITCH3_DIMENSIONS,
        "feature_order": list(contract.feature_order),
        "classifier_sha256": contract.classifier_sha256,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise LTSNContractError(f"Pitch-3 checkpoint {name} does not match the scorer")
    for name in ("training_config_sha256", "training_manifest_sha256"):
        _sha(metadata.get(name), f"checkpoint {name}")
