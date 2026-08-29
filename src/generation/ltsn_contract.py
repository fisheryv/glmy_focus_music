"""Frozen 18-D Path Homology contract and checkpoint validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FINGERPRINT_ID = "focus_path_homology_fingerprint_v2"
FINGERPRINT_DIMENSIONS = 18
DISTANCE_WEIGHTS = (0.5, 0.25, 0.25)
CANONICAL_FEATURE_ORDER = tuple(
    [f"pitch_whitened_{index:02d}" for index in range(16)]
    + ["path_acoustic_phase__loop_score", "path_chroma_phase__loop_score"]
)
_FORBIDDEN_TOKENS = ("rhythm", "modulation", "structure", "tda")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LTSNContractError(ValueError):
    """Raised when a scorer or LTSN checkpoint violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class FingerprintContract:
    """Validated runtime subset of the frozen exact-scorer artifact."""

    fingerprint_id: str
    spec_revision: str
    artifact_sha256: str
    feature_order: tuple[str, ...]
    distance_weights: tuple[float, float, float]
    classifier_coef: tuple[float, ...]
    classifier_intercept: float
    focus_band_threshold: float
    classifier_sha256: str
    input_sha256: str
    config_sha256: str
    code_sha256: str


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    text = str(value).lower()
    if not _SHA256.fullmatch(text):
        raise LTSNContractError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _validate_feature_order(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list) or len(values) != FINGERPRINT_DIMENSIONS:
        raise LTSNContractError("feature_order must contain exactly 18 entries")
    order = tuple(str(value) for value in values)
    lowered = tuple(value.lower() for value in order)
    if any(token in name for name in lowered for token in _FORBIDDEN_TOKENS):
        raise LTSNContractError("legacy Rhythm/Modulation/Structure/TDA feature rejected")
    if not all(name.startswith("pitch") for name in lowered[:16]):
        raise LTSNContractError("feature_order[0:16] must be Pitch coordinates")
    if "acoustic" not in lowered[16] or "loop_score" not in lowered[16]:
        raise LTSNContractError("feature_order[16] must be Acoustic phase loop_score")
    if "chroma" not in lowered[17] or "loop_score" not in lowered[17]:
        raise LTSNContractError("feature_order[17] must be Chroma phase loop_score")
    return order


def _finite_vector(values: Any, name: str, size: int) -> tuple[float, ...]:
    if not isinstance(values, list) or len(values) != size:
        raise LTSNContractError(f"{name} must contain exactly {size} values")
    vector = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in vector):
        raise LTSNContractError(f"{name} contains NaN or Inf")
    return vector


def _classifier_sha256(coef: Sequence[float], intercept: float) -> str:
    payload = json.dumps(
        {"coef": list(coef), "intercept": intercept},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_fingerprint_contract(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> FingerprintContract:
    """Load an exact scorer only when it matches the frozen 18-D specification."""

    artifact_sha256 = sha256_file(path)
    if expected_sha256 is not None and artifact_sha256 != expected_sha256.lower():
        raise LTSNContractError("fingerprint JSON SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint_id") != FINGERPRINT_ID:
        raise LTSNContractError("unexpected fingerprint_id")
    if payload.get("dimensions") != FINGERPRINT_DIMENSIONS:
        raise LTSNContractError("legacy or malformed scorer: dimensions must equal 18")
    feature_order = _validate_feature_order(payload.get("feature_order"))
    weights = _finite_vector(payload.get("distance_weights"), "distance_weights", 3)
    if any(
        abs(actual - expected) > 1e-12
        for actual, expected in zip(weights, DISTANCE_WEIGHTS, strict=True)
    ):
        raise LTSNContractError("distance_weights must be [1/2, 1/4, 1/4]")
    coef = _finite_vector(payload.get("classifier_coef"), "classifier_coef", 18)
    intercept = float(payload.get("classifier_intercept"))
    threshold = float(payload.get("focus_band_threshold"))
    if not math.isfinite(intercept) or not math.isfinite(threshold):
        raise LTSNContractError("classifier intercept and target threshold must be finite")
    spec_revision = str(payload.get("spec_revision", "")).strip()
    if not spec_revision:
        raise LTSNContractError("spec_revision is required")
    classifier_sha256 = _classifier_sha256(coef, intercept)
    if _require_sha256(payload.get("classifier_sha256"), "classifier_sha256") != classifier_sha256:
        raise LTSNContractError("classifier_sha256 does not match coefficients and intercept")
    return FingerprintContract(
        fingerprint_id=FINGERPRINT_ID,
        spec_revision=spec_revision,
        artifact_sha256=artifact_sha256,
        feature_order=feature_order,
        distance_weights=DISTANCE_WEIGHTS,
        classifier_coef=coef,
        classifier_intercept=intercept,
        focus_band_threshold=threshold,
        classifier_sha256=classifier_sha256,
        input_sha256=_require_sha256(payload.get("input_sha256"), "input_sha256"),
        config_sha256=_require_sha256(payload.get("config_sha256"), "config_sha256"),
        code_sha256=_require_sha256(payload.get("code_sha256"), "code_sha256"),
    )


def validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
    contract: FingerprintContract,
) -> None:
    """Reject checkpoints whose frozen scorer, data, model, or split identity changed."""

    exact = {
        "fingerprint_id": contract.fingerprint_id,
        "fingerprint_spec_revision": contract.spec_revision,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "dimensions": FINGERPRINT_DIMENSIONS,
        "feature_order": list(contract.feature_order),
        "distance_weights": list(DISTANCE_WEIGHTS),
        "classifier_sha256": contract.classifier_sha256,
    }
    for name, expected in exact.items():
        if metadata.get(name) != expected:
            raise LTSNContractError(f"checkpoint {name} does not match the frozen scorer")
    for name in (
        "ltsn_config_sha256",
        "training_manifest_sha256",
        "split_manifest_sha256",
        "exact_label_table_sha256",
        "ace_model_sha256",
        "vae_sha256",
    ):
        _require_sha256(metadata.get(name), f"checkpoint {name}")
