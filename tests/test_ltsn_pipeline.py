from __future__ import annotations

# ruff: noqa: E402, I001

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from generation.ltsn_pipeline import (
    TrajectoryRecorder,
    build_exact_label_tables,
    synthetic_descriptor_rows,
    write_csv_atomic,
)
from generation.ltsn_contract import LTSNContractError
from generation.ltsn_training import (
    LTSNTrainingConfig,
    _resolve_training_devices,
    train_ensemble,
)
from generation.path_homology_exact_scorer import ExactPathHomologyScorer

ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = ROOT / "metadata" / "focus_path_homology_fingerprint_v2.json"


def test_three_seed_training_maps_one_explicit_gpu_per_seed() -> None:
    seeds = (20260716, 20260717, 20260718)

    assert _resolve_training_devices(
        seeds, None, ("cuda:0", "cuda:1", "cuda:2")
    ) == ("cuda:0", "cuda:1", "cuda:2")
    with pytest.raises(LTSNContractError, match="requires 3 devices"):
        _resolve_training_devices(seeds, None, ("cuda:0", "cuda:1"))
    with pytest.raises(LTSNContractError, match="must be unique"):
        _resolve_training_devices(seeds, None, ("cuda:0", "cuda:0", "cuda:2"))
    with pytest.raises(LTSNContractError, match="explicit CUDA"):
        _resolve_training_devices(seeds, None, ("cuda:0", "cuda:1", "cuda"))
    with pytest.raises(LTSNContractError, match="non-empty and unique"):
        LTSNTrainingConfig(seeds=(7, 7, 8)).validate(engineering_smoke=True)


def _record_split(recorder: TrajectoryRecorder, split: str, index: int) -> None:
    recorder.begin(
        prompt_id=f"prompt_{split}", trajectory_id=f"trajectory_{split}", split=split
    )
    generator = torch.Generator().manual_seed(index)
    latent = torch.randn(1, 48, 64, generator=generator)
    mask = torch.ones(1, 48, dtype=torch.bool)
    for step in range(8):
        velocity = torch.randn(1, 48, 64, generator=generator) * 0.05
        unchanged = recorder(
            xt_next=latent,
            xt_before_step=latent,
            velocity=velocity,
            timestep=1.0 - step / 9.0,
            next_timestep=1.0 - (step + 1) / 9.0,
            step_index=step,
            attention_mask=mask,
        )
        assert unchanged.data_ptr() == latent.data_ptr()
    recorder.end()


def test_synthetic_smoke_pipeline_records_labels_and_trains(tmp_path: Path) -> None:
    collection = tmp_path / "collection"
    recorder = TrajectoryRecorder(
        collection,
        model_family="acestep-v15-xl-turbo",
        ace_model_sha256="a" * 64,
        vae_sha256="b" * 64,
        engineering_smoke=True,
    )
    for index, split in enumerate(("train", "development", "calibration", "qualification")):
        _record_split(recorder, split, index)
    trajectory_manifest = collection / "trajectory_manifest.csv"
    recorder.write_manifest(trajectory_manifest)
    assert len(recorder.records) == 16

    scorer = ExactPathHomologyScorer.from_json(FINGERPRINT)
    descriptors = tmp_path / "descriptors.csv"
    write_csv_atomic(
        descriptors,
        synthetic_descriptor_rows(
            trajectory_manifest,
            pitch_dimensions=len(scorer.transforms["pitch"]["input_features"]),
        ),
    )
    labels = tmp_path / "labels"
    labels.mkdir()
    summary = build_exact_label_tables(
        trajectory_manifest=trajectory_manifest,
        descriptor_table=descriptors,
        output_manifest=labels / "manifest.csv",
        exact_label_table=labels / "exact.csv",
        split_manifest=labels / "splits.json",
        scorer=scorer,
        gate=None,
        engineering_smoke=True,
    )
    assert summary["samples"] == 16
    assert not summary["qualification_eligible"]

    config = tmp_path / "smoke.toml"
    config.write_text(
        """
[model]
condition_dim = 16
stem_channels = 8
local_channels = 8
global_channels = 8
transformer_heads = 2
transformer_layers = 1
dropout = 0.0

[training]
effective_batch_size = 4
micro_batch_size = 4
max_epochs = 1
minimum_epochs = 1
early_stopping_patience = 1
num_workers = 0
seeds = [7]
use_bf16 = false

[loss]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    result = train_ensemble(
        fingerprint_path=FINGERPRINT,
        manifest_path=labels / "manifest.csv",
        split_manifest_path=labels / "splits.json",
        config_path=config,
        output_dir=tmp_path / "models",
        reranking_gate_path=None,
        engineering_smoke=True,
        device_name="cpu",
    )
    assert result["status"] == "engineering_smoke_only"
    assert result["devices"] == ["cpu"]
    assert not result["parallel_training"]
    assert len(result["checkpoints"]) == 1
    assert (tmp_path / "models" / result["checkpoints"][0]["path"]).is_file()
