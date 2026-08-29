from __future__ import annotations

import hashlib
import json

import pytest

from generation.ltsn_contract import (
    CANONICAL_FEATURE_ORDER,
    DISTANCE_WEIGHTS,
    LTSNContractError,
    load_fingerprint_contract,
    validate_checkpoint_metadata,
)


def _write_contract(tmp_path, **overrides):
    payload = {
        "fingerprint_id": "focus_path_homology_fingerprint_v2",
        "spec_revision": "2026-08-07",
        "dimensions": 18,
        "feature_order": list(CANONICAL_FEATURE_ORDER),
        "distance_weights": list(DISTANCE_WEIGHTS),
        "classifier_coef": [0.1] * 18,
        "classifier_intercept": -0.25,
        "classifier_sha256": hashlib.sha256(
            json.dumps(
                {"coef": [0.1] * 18, "intercept": -0.25},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "focus_band_threshold": 0.5,
        "input_sha256": "1" * 64,
        "config_sha256": "9" * 64,
        "code_sha256": "2" * 64,
    }
    payload.update(overrides)
    path = tmp_path / "fingerprint.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _checkpoint_metadata(contract):
    classifier_payload = json.dumps(
        {"coef": [0.1] * 18, "intercept": -0.25},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "fingerprint_id": contract.fingerprint_id,
        "fingerprint_spec_revision": contract.spec_revision,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "dimensions": 18,
        "feature_order": list(contract.feature_order),
        "distance_weights": list(DISTANCE_WEIGHTS),
        "classifier_sha256": hashlib.sha256(classifier_payload).hexdigest(),
        "ltsn_config_sha256": "3" * 64,
        "training_manifest_sha256": "4" * 64,
        "split_manifest_sha256": "5" * 64,
        "exact_label_table_sha256": "6" * 64,
        "ace_model_sha256": "7" * 64,
        "vae_sha256": "8" * 64,
    }


def test_loads_frozen_18d_contract_and_checkpoint(tmp_path) -> None:
    path = _write_contract(tmp_path)
    contract = load_fingerprint_contract(path)

    assert len(contract.feature_order) == 18
    assert contract.distance_weights == (0.5, 0.25, 0.25)
    validate_checkpoint_metadata(_checkpoint_metadata(contract), contract)


def test_rejects_legacy_51d_artifact(tmp_path) -> None:
    path = _write_contract(tmp_path, dimensions=51)

    with pytest.raises(LTSNContractError, match="dimensions must equal 18"):
        load_fingerprint_contract(path)


def test_rejects_forbidden_runtime_view(tmp_path) -> None:
    order = list(CANONICAL_FEATURE_ORDER)
    order[3] = "rhythm_whitened_00"
    path = _write_contract(tmp_path, feature_order=order)

    with pytest.raises(LTSNContractError, match="legacy Rhythm"):
        load_fingerprint_contract(path)


def test_rejects_hash_mismatch(tmp_path) -> None:
    path = _write_contract(tmp_path)

    with pytest.raises(LTSNContractError, match="SHA-256 mismatch"):
        load_fingerprint_contract(path, expected_sha256="0" * 64)


def test_rejects_checkpoint_from_different_label_table(tmp_path) -> None:
    contract = load_fingerprint_contract(_write_contract(tmp_path))
    metadata = _checkpoint_metadata(contract)
    metadata["exact_label_table_sha256"] = "not-a-hash"

    with pytest.raises(LTSNContractError, match="exact_label_table_sha256"):
        validate_checkpoint_metadata(metadata, contract)
