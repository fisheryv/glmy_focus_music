from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from generation.latent_topology_control_head import (  # noqa: E402
    LatentTopologyControlHead,
    Pitch3ControlHeadConfig,
)
from generation.ltsn_contract import sha256_file  # noqa: E402
from generation.pitch3_contract import load_pitch3_contract  # noqa: E402
from generation.pitch3_training import (  # noqa: E402
    Pitch3BalancedBatchSampler,
    Pitch3LossWeights,
    pitch3_loss,
    read_pitch3_manifest,
    train_pitch3_control_head,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "metadata" / "focus_pitch3_fingerprint_v1.json"


def test_pitch3_control_head_shapes_gradients_and_parameter_budget() -> None:
    contract = load_pitch3_contract(PROFILE)
    config = Pitch3ControlHeadConfig(
        model_dim=32,
        condition_dim=32,
        transformer_heads=4,
        transformer_layers=1,
        feedforward_dim=64,
        temporal_stride=2,
        dropout=0.0,
    )
    model = LatentTopologyControlHead(contract, config)
    latent = torch.randn(2, 31, 64, requires_grad=True)
    mask = torch.ones(2, 31, dtype=torch.bool)
    mask[1, 25:] = False

    output = model(latent, torch.tensor([0.5, 0.25]), torch.tensor([4, 6]), mask)
    loss, parts = pitch3_loss(
        output,
        torch.randn(2, 3),
        torch.randn(2),
        torch.tensor([0.0, 1.0]),
        Pitch3LossWeights(
            band=0.5,
            band_rank=0.1,
            ood_margin=0.1,
            ood_margin_value=1.0,
        ),
        target_lower=torch.tensor(contract.target_lower),
        target_upper=torch.tensor(contract.target_upper),
        distance_weights=torch.tensor(contract.distance_weights),
    )
    loss.backward()

    assert output.coordinate_mean.shape == (2, 3)
    assert output.coordinate_logvar.shape == (2, 3)
    assert output.ood_logit.shape == (2,)
    assert output.focus_logit.shape == (2,)
    assert set(parts) == {
        "coordinate",
        "nll",
        "focus",
        "ood",
        "band",
        "band_rank",
        "ood_margin",
    }
    assert torch.isfinite(loss)
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert model.trainable_parameters < 7_000_000


def _write_smoke_manifest(tmp_path: Path) -> Path:
    contract = load_pitch3_contract(PROFILE)
    rows = []
    for index, (split, ood) in enumerate(
        (("train", 0.0), ("train", 1.0), ("development", 0.0), ("development", 1.0))
    ):
        latent_path = tmp_path / f"latent_{index}.npy"
        np.save(latent_path, np.full((8 + index, 64), index / 10.0, np.float32))
        rows.append(
            {
                "sample_id": f"sample_{index}",
                "prompt_id": f"prompt_{index}",
                "trajectory_id": f"trajectory_{index}",
                "split": split,
                "step_number": 4,
                "timestep": 0.5,
                "latent_path": latent_path.name,
                "latent_sha256": sha256_file(latent_path),
                "coordinates_json": json.dumps([0.1 * index, 0.2 * index, 0.3 * index]),
                "focus_logit": 0.25 * index,
                "ood_label": ood,
                "ood_kind": "" if ood == 0.0 else "ood_scale_high",
                "ood_transform_version": ("" if ood == 0.0 else "pitch3_latent_ood_v2"),
                "ood_label_source": ("" if ood == 0.0 else "deterministic_latent_transform_v2"),
                "fingerprint_json_sha256": contract.artifact_sha256,
                "feature_order_json": json.dumps(list(contract.feature_order)),
                "label_scope": "per_snapshot_exact",
                "is_final": "false",
            }
        )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_pitch3_training_writes_hash_bound_non_authorizing_checkpoint(tmp_path: Path) -> None:
    manifest = _write_smoke_manifest(tmp_path)
    config = tmp_path / "training.toml"
    config.write_text(
        """
[model]
latent_dim = 64
model_dim = 32
condition_dim = 32
transformer_heads = 4
transformer_layers = 1
feedforward_dim = 64
temporal_stride = 2
dropout = 0.0
logvar_min = -8.0
logvar_max = 4.0

[training]
learning_rate = 0.001
weight_decay = 0.0
micro_batch_size = 2
max_epochs = 2
minimum_epochs = 1
early_stopping_patience = 1
gradient_clip_norm = 1.0
num_workers = 0
use_bf16 = false
seed = 17
require_ood_both_classes = true
scale_high_training_target_scales = [2.5, 3.5, 4.0]

[loss]
coordinate = 1.0
nll = 0.1
focus = 0.1
ood = 0.1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = train_pitch3_control_head(
        fingerprint_path=PROFILE,
        training_manifest=manifest,
        config_path=config,
        output_dir=tmp_path / "output",
        device_name="cpu",
    )

    checkpoint = Path(result["checkpoint"])
    assert checkpoint.is_file()
    assert result["checkpoint_sha256"] == sha256_file(checkpoint)
    assert result["guidance_promotion_eligible"] is False
    assert result["ood_class_counts"] == {
        "train": {"id": 1, "ood": 1},
        "development": {"id": 1, "ood": 1},
    }
    assert result["production_authorization"] is False
    augmentation = result["training_virtual_ood_augmentation"]
    assert augmentation["assignment_count"] == 1
    assert augmentation["target_scales"] == [2.5, 3.5, 4.0]
    assert augmentation["coordinate_targets_used"] is False
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert (
        payload["metadata"]["fingerprint_json_sha256"]
        == load_pitch3_contract(PROFILE).artifact_sha256
    )


def test_pitch3_balanced_batch_sampler_has_fixed_class_counts(tmp_path: Path) -> None:
    manifest = _write_smoke_manifest(tmp_path)
    records = [
        record
        for record in read_pitch3_manifest(manifest, load_pitch3_contract(PROFILE))
        if record.split == "train"
    ]
    sampler = Pitch3BalancedBatchSampler(
        records,
        id_per_batch=1,
        ood_per_batch=1,
        seed=17,
    )
    sampler.set_epoch(3)
    batches = list(sampler)
    assert batches
    assert all(
        sorted(records[index].ood_label for index in batch) == [0.0, 1.0] for batch in batches
    )
