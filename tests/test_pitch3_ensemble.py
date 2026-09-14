from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from generation.latent_topology_control_head import (  # noqa: E402
    LatentTopologyControlHead,
    Pitch3ControlHeadConfig,
    Pitch3ControlOutput,
)
from generation.ltsn_contract import LTSNContractError, sha256_file  # noqa: E402
from generation.pitch3_contract import load_pitch3_contract  # noqa: E402
from generation.pitch3_ensemble import (  # noqa: E402
    ENSEMBLE_AGGREGATION,
    Pitch3ControlHeadEnsemble,
    build_pitch3_ensemble_manifest,
    load_pitch3_ensemble,
)
from generation.pitch3_evaluation import _validate_upstream_model_binding  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "metadata" / "focus_pitch3_fingerprint_v1.json"


class _FixedMember(nn.Module):
    def __init__(
        self,
        mean: list[float],
        variance: list[float],
        ood_probability: float,
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("logvar", torch.tensor(variance).log())
        self.ood_logit = float(torch.logit(torch.tensor(ood_probability)))

    def forward(self, latent, timestep, step_number, attention_mask=None):
        batch = latent.shape[0]
        return Pitch3ControlOutput(
            self.mean.expand(batch, -1) + latent[:, 0, :3] * 0.0,
            self.logvar.expand(batch, -1),
            torch.full((batch,), self.ood_logit, device=latent.device),
            torch.full((batch,), -999.0, device=latent.device),
        )


def test_pitch3_ensemble_uses_frozen_aggregation_contract() -> None:
    contract = load_pitch3_contract(PROFILE)
    model = Pitch3ControlHeadEnsemble(
        [
            _FixedMember([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 0.2),
            _FixedMember([3.0, 4.0, 5.0], [4.0, 4.0, 4.0], 0.8),
        ],
        [0.25, 0.75],
        contract,
    )
    latent = torch.randn(2, 4, 64, requires_grad=True)
    output = model(latent, torch.tensor([0.2, 0.4]), torch.tensor([4, 5]))

    expected_mean = torch.tensor([2.5, 3.5, 4.5]).expand(2, -1)
    assert torch.allclose(output.coordinate_mean, expected_mean)
    assert torch.allclose(output.coordinate_logvar.exp(), torch.full((2, 3), 4.0))
    assert torch.allclose(output.ood_logit.sigmoid(), torch.full((2,), 0.65))
    expected_focus = expected_mean @ torch.tensor(contract.classifier_coef)
    expected_focus += contract.classifier_intercept
    assert torch.allclose(output.focus_logit, expected_focus)
    output.focus_logit.sum().backward()
    assert latent.grad is not None


def _write_checkpoint(
    path: Path,
    *,
    training_manifest_sha256: str,
    training_config_sha256: str,
    objective_kind: str,
    model_dim: int = 8,
) -> None:
    contract = load_pitch3_contract(PROFILE)
    config = Pitch3ControlHeadConfig(
        model_dim=model_dim,
        condition_dim=8,
        transformer_heads=2,
        transformer_layers=1,
        feedforward_dim=16,
        temporal_stride=2,
        dropout=0.0,
    )
    model = LatentTopologyControlHead(contract, config)
    metadata = {
        "schema_version": 1,
        "model_family": "latent_topology_control_head_pitch3",
        "fingerprint_id": contract.fingerprint_id,
        "fingerprint_spec_revision": contract.spec_revision,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "dimensions": 3,
        "feature_order": list(contract.feature_order),
        "classifier_sha256": contract.classifier_sha256,
        "training_config_sha256": training_config_sha256,
        "training_manifest_sha256": training_manifest_sha256,
        "architecture_revision": "pitch3_ltch_test",
        "band_training_objective": {"kind": objective_kind},
        "ood_transform_version": "pitch3_latent_ood_v2",
        "ood_label_source": "deterministic_latent_transform_v2",
    }
    torch.save(
        {
            "metadata": metadata,
            "model_config": asdict(config),
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def _build_test_ensemble(tmp_path: Path) -> tuple[Path, Path]:
    contract = load_pitch3_contract(PROFILE)
    training_manifest = tmp_path / "training.csv"
    training_manifest.write_text("sample_id\n", encoding="utf-8")
    first = tmp_path / "v23.pt"
    second = tmp_path / "v24r.pt"
    manifest_sha256 = sha256_file(training_manifest)
    _write_checkpoint(
        first,
        training_manifest_sha256=manifest_sha256,
        training_config_sha256="a" * 64,
        objective_kind="hard_excursion_v23_compatible",
    )
    _write_checkpoint(
        second,
        training_manifest_sha256=manifest_sha256,
        training_config_sha256="b" * 64,
        objective_kind="region_consistency_v24",
    )
    ensemble_manifest = tmp_path / "ensemble" / "manifest.json"
    build_pitch3_ensemble_manifest(
        contract=contract,
        training_manifest=training_manifest,
        members=(("v23", first, 0.5), ("v24r", second, 0.5)),
        output_path=ensemble_manifest,
    )
    return training_manifest, ensemble_manifest


def test_pitch3_ensemble_manifest_round_trip_and_hash_binding(tmp_path: Path) -> None:
    training_manifest, ensemble_manifest = _build_test_ensemble(tmp_path)
    contract = load_pitch3_contract(PROFILE)
    model, metadata, digest = load_pitch3_ensemble(
        manifest_path=ensemble_manifest,
        contract=contract,
        training_manifest=training_manifest,
        device=torch.device("cpu"),
        expected_manifest_sha256=sha256_file(ensemble_manifest),
    )

    assert isinstance(model, Pitch3ControlHeadEnsemble)
    assert digest == sha256_file(ensemble_manifest)
    assert metadata["aggregation"] == ENSEMBLE_AGGREGATION
    assert [member["name"] for member in metadata["members"]] == ["v23", "v24r"]

    member_path = ensemble_manifest.parent / metadata["members"][0]["checkpoint_path"]
    with member_path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(LTSNContractError, match="member SHA-256"):
        load_pitch3_ensemble(
            manifest_path=ensemble_manifest,
            contract=contract,
            training_manifest=training_manifest,
            device=torch.device("cpu"),
        )


def test_pitch3_ensemble_rejects_incompatible_model_configs(tmp_path: Path) -> None:
    contract = load_pitch3_contract(PROFILE)
    training_manifest = tmp_path / "training.csv"
    training_manifest.write_text("sample_id\n", encoding="utf-8")
    manifest_sha256 = sha256_file(training_manifest)
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    _write_checkpoint(
        first,
        training_manifest_sha256=manifest_sha256,
        training_config_sha256="a" * 64,
        objective_kind="first",
    )
    _write_checkpoint(
        second,
        training_manifest_sha256=manifest_sha256,
        training_config_sha256="b" * 64,
        objective_kind="second",
        model_dim=16,
    )

    with pytest.raises(LTSNContractError, match="share one model config"):
        build_pitch3_ensemble_manifest(
            contract=contract,
            training_manifest=training_manifest,
            members=(("first", first, 0.5), ("second", second, 0.5)),
            output_path=tmp_path / "ensemble.json",
        )


def test_pitch3_evaluation_model_binding_supports_ensemble_and_legacy_single() -> None:
    ensemble_binding = {
        "model_artifact_kind": "weighted_ensemble",
        "model_artifact_sha256": "e" * 64,
    }
    _validate_upstream_model_binding(
        dict(ensemble_binding), ensemble_binding, stage="development screen"
    )
    with pytest.raises(LTSNContractError, match="model_artifact_sha256"):
        _validate_upstream_model_binding(
            {**ensemble_binding, "model_artifact_sha256": "f" * 64},
            ensemble_binding,
            stage="development screen",
        )

    single_binding = {
        "model_artifact_kind": "single_checkpoint",
        "model_artifact_sha256": "c" * 64,
        "checkpoint_sha256": "c" * 64,
    }
    _validate_upstream_model_binding(
        {"checkpoint_sha256": "c" * 64}, single_binding, stage="development screen"
    )
