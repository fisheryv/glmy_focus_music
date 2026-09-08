from __future__ import annotations

# ruff: noqa: E402, I001

import csv
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from generation import ltsn_training
from generation.ltsn_evaluation import _matched_step_ood_summary, _metrics
from generation.ltsn_pipeline import (
    TrajectoryRecorder,
    build_exact_label_tables,
    synthetic_descriptor_rows,
    validate_snapshot_coverage,
    write_csv_atomic,
)
from generation.ltsn_cli import collect_main, merge_main
from generation.ltsn_contract import LTSNContractError
from generation.ltsn_dataset import LTSNSnapshot
from generation.ltsn_training import (
    LTSNTrainingConfig,
    PromptGroupedBatchSampler,
    _resolve_training_devices,
    _training_target_contract,
    train_ensemble,
)
from generation.ltsn_training_augmentation import (
    _augmentation_plan,
    _evaluation_ood_plan,
    _on_policy_plan,
    _write_augmentation_trajectory_manifest,
)
from generation.ltsn_losses import focus_band_classification_loss
from generation.path_homology_surrogate import LTSNConfig
from generation.path_homology_exact_scorer import ExactPathHomologyScorer

ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = ROOT / "metadata" / "focus_path_homology_fingerprint_v2.json"


def _snapshot(
    sample_id: str,
    trajectory_id: str,
    step: int,
    *,
    anchor: str = "",
    coordinates: tuple[float, ...] = (0.0,) * 18,
    ood_label: float = 0.0,
) -> LTSNSnapshot:
    return LTSNSnapshot(
        sample_id=sample_id,
        prompt_id="prompt",
        trajectory_id=trajectory_id,
        split="train",
        step_number=step,
        timestep=0.5,
        latent_path=Path("unused.npy"),
        latent_sha256="a" * 64,
        coordinates=coordinates,
        focus_logit=0.0,
        ood_label=ood_label,
        is_final=step == 8,
        exact_label_table_sha256="b" * 64,
        local_anchor_sample_id=anchor,
    )


def test_prompt_grouped_sampler_keeps_local_pairs_and_trajectory_steps_together() -> None:
    records = [
        _snapshot("anchor", "trajectory", 5),
        _snapshot("step4", "trajectory", 4),
        _snapshot("step6", "trajectory", 6),
        _snapshot("local_plus", "local_plus", 5, anchor="anchor"),
        _snapshot("local_minus", "local_minus", 5, anchor="anchor"),
    ]

    batches = list(PromptGroupedBatchSampler(records, batch_size=8, seed=7))

    assert len(batches) == 1
    assert set(batches[0]) == set(range(len(records)))
    assert batches[0].count(0) == 1


def test_prompt_grouped_sampler_length_matches_fragmented_units() -> None:
    records: list[LTSNSnapshot] = []
    for group in range(3):
        anchor = f"anchor_{group}"
        records.append(_snapshot(anchor, f"trajectory_{group}", 5))
        records.extend(
            _snapshot(
                f"local_{group}_{index}",
                f"local_{group}_{index}",
                5,
                anchor=anchor,
            )
            for index in range(4)
        )
    sampler = PromptGroupedBatchSampler(records, batch_size=8, seed=7)

    batches = list(sampler)

    assert len(sampler) == len(batches) == 3
    assert sorted(index for batch in batches for index in batch) == list(range(15))


def test_v2_target_contract_uses_only_id_targets_and_requires_ood_class() -> None:
    low = (0.0, 0.0, 0.0, 1.0, *((0.0,) * 14))
    high = (0.0, 0.0, 0.0, 3.0, *((0.0,) * 14))
    ood = (0.0, 0.0, 0.0, 100.0, *((0.0,) * 14))
    records = [
        _snapshot("id_low", "t1", 4, coordinates=low),
        _snapshot("id_high", "t2", 4, coordinates=high),
        _snapshot("ood", "t3", 4, coordinates=ood, ood_label=1.0),
    ]

    contract = _training_target_contract(
        records,
        LTSNConfig(inactive_coordinate_indices=(0, 1, 2)),
        LTSNTrainingConfig(require_ood_both_classes=True),
    )

    assert contract["coordinate_mean"][3] == pytest.approx(2.0)
    assert contract["coordinate_standard_deviation"][3] == pytest.approx(1.0)
    assert contract["active_coordinate_mask"][:3] == [False, False, False]
    assert contract["ood_positive_samples"] == 1
    assert contract["ood_negative_samples"] == 2


def test_focus_band_loss_uses_existing_focus_logit_and_frozen_threshold() -> None:
    exact = torch.tensor([0.5, 2.5])
    correct = torch.tensor([0.0, 3.0])
    reversed_prediction = torch.tensor([3.0, 0.0])
    in_distribution = torch.tensor([True, True])

    correct_loss = focus_band_classification_loss(
        correct,
        exact,
        in_distribution,
        focus_band_threshold=1.5,
    )
    reversed_loss = focus_band_classification_loss(
        reversed_prediction,
        exact,
        in_distribution,
        focus_band_threshold=1.5,
    )

    assert correct_loss < reversed_loss


def test_augmentation_plan_and_exact_manifest_are_deterministic(tmp_path: Path) -> None:
    anchor = _snapshot("anchor", "trajectory", 5)
    first = _augmentation_plan(
        [anchor],
        perturbations_per_anchor=2,
        rms_ratio=0.005,
        ood_per_prompt=1,
        seed=7,
    )
    second = _augmentation_plan(
        [anchor],
        perturbations_per_anchor=2,
        rms_ratio=0.005,
        ood_per_prompt=1,
        seed=7,
    )
    assert first == second
    assert [item["kind"] for item in first] == [
        "local_direction",
        "local_direction",
        "ood_zero",
    ]

    manifest = tmp_path / "training_augmentation_trajectories.csv"
    source = {
        "anchor": {
            "model_family": "acestep-v15-xl-turbo",
            "ace_model_sha256": "a" * 64,
            "vae_sha256": "b" * 64,
        }
    }
    receipts = {
        item["sample_id"]: {
            "sample_id": item["sample_id"],
            "latent_path": f"latents/{item['sample_id']}.npy",
            "latent_sha256": "c" * 64,
            "audio_path": f"audio/{item['sample_id']}.wav",
            "audio_sha256": "d" * 64,
        }
        for item in first
    }
    _write_augmentation_trajectory_manifest(
        path=manifest,
        planned=first,
        source_by_sample=source,
        receipt_by_id=receipts,
    )
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in rows] == [item["sample_id"] for item in first]
    assert rows[0]["local_anchor_sample_id"] == "anchor"
    assert rows[-1]["local_anchor_sample_id"] == ""


def test_on_policy_plan_uses_all_frozen_rms_ratios_and_preserves_split() -> None:
    anchor = _snapshot("anchor", "trajectory", 5)
    descriptions = {
        "anchor": {
            "exact_out_of_band": True,
            "proxy_out_of_band": True,
            "gradient_usable": True,
        }
    }

    planned = _on_policy_plan([anchor], descriptions, (0.0025, 0.005, 0.01))

    assert [item["rms_ratio"] for item in planned] == [0.0025, 0.005, 0.01]
    assert {item["kind"] for item in planned} == {"on_policy_direction"}
    assert {item["split"] for item in planned} == {"train"}
    assert len({item["sample_id"] for item in planned}) == 3


def test_evaluation_ood_is_prompt_held_out_and_uses_unseen_transforms() -> None:
    records = []
    for split in ("calibration", "qualification"):
        for prompt_index in range(2):
            for step in (4, 5, 6, 8):
                sample = _snapshot(
                    f"{split}_{prompt_index}_step{step}",
                    f"{split}_trajectory_{prompt_index}",
                    step,
                )
                sample = LTSNSnapshot(
                    sample_id=sample.sample_id,
                    prompt_id=f"{split}_prompt_{prompt_index}",
                    trajectory_id=sample.trajectory_id,
                    split=split,
                    step_number=sample.step_number,
                    timestep=sample.timestep,
                    latent_path=sample.latent_path,
                    latent_sha256=sample.latent_sha256,
                    coordinates=sample.coordinates,
                    focus_logit=sample.focus_logit,
                    ood_label=sample.ood_label,
                    is_final=sample.is_final,
                    exact_label_table_sha256=sample.exact_label_table_sha256,
                    local_anchor_sample_id=sample.local_anchor_sample_id,
                )
                records.append(sample)

    planned = _evaluation_ood_plan(records, ood_per_prompt=1, seed=7)

    assert len(planned) == 16
    assert {item["split"] for item in planned} == {"calibration", "qualification"}
    assert {item["step_number"] for item in planned} == {4, 5, 6, 8}
    assert {item["kind"] for item in planned} == {
        "ood_time_reverse",
        "ood_channel_roll",
    }


def test_qualification_fidelity_metrics_exclude_ood_rows() -> None:
    exact = np.repeat(np.arange(5, dtype=float)[:, None], 18, axis=1)
    exact[:, :3] = 0.0
    mean = exact.copy()
    mean[-1] = -100.0
    prediction = {
        "coordinates": exact,
        "coordinate_mean": mean,
        "focus_logit": np.arange(5, dtype=float),
        "predicted_focus_logit": np.asarray([0.0, 1.0, 2.0, 3.0, -100.0]),
        "ood_label": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0]),
        "ood_probability": np.asarray([0.1, 0.1, 0.1, 0.1, 0.9]),
        "total_variance": np.ones((5, 18), dtype=float),
        "active_coordinate_mask": np.asarray([False, False, False, *([True] * 15)]),
    }

    metrics = _metrics(prediction, np.ones(18, dtype=float))

    assert metrics["n"] == 5
    assert metrics["n_in_distribution"] == 4
    assert metrics["focus_logit_mae"] == 0.0
    assert metrics["coordinate_mae"] == [0.0] * 18
    assert metrics["ood_auroc"] == 1.0


def test_ood_gate_requires_every_step_and_uses_worst_matched_step() -> None:
    summary = _matched_step_ood_summary(
        {
            "4": {"ood_auroc": 0.91},
            "5": {"ood_auroc": 0.83},
            "6": {"ood_auroc": 0.88},
            "8": {"ood_auroc": 0.86},
        }
    )

    assert summary["required_steps_present"] is True
    assert summary["worst_step_auroc"] == pytest.approx(0.83)
    assert summary["macro_auroc"] == pytest.approx(0.87)


def test_three_seed_training_maps_one_explicit_gpu_per_seed() -> None:
    seeds = (20260716, 20260717, 20260718)

    assert _resolve_training_devices(seeds, None, ("cuda:0", "cuda:1", "cuda:2")) == (
        "cuda:0",
        "cuda:1",
        "cuda:2",
    )
    with pytest.raises(LTSNContractError, match="requires 3 devices"):
        _resolve_training_devices(seeds, None, ("cuda:0", "cuda:1"))
    with pytest.raises(LTSNContractError, match="must be unique"):
        _resolve_training_devices(seeds, None, ("cuda:0", "cuda:0", "cuda:2"))
    with pytest.raises(LTSNContractError, match="explicit CUDA"):
        _resolve_training_devices(seeds, None, ("cuda:0", "cuda:1", "cuda"))
    with pytest.raises(LTSNContractError, match="non-empty and unique"):
        LTSNTrainingConfig(seeds=(7, 7, 8)).validate(engineering_smoke=True)


def test_formal_training_rejects_three_step_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "ltsn_manifest.csv"
    write_csv_atomic(
        manifest,
        [
            {
                "sample_id": f"trajectory__step{step:02d}__b00",
                "trajectory_id": "trajectory",
                "step_number": step,
                "is_final": False,
            }
            for step in (4, 5, 6)
        ],
    )
    monkeypatch.setattr(ltsn_training, "load_fingerprint_contract", lambda _path: object())
    monkeypatch.setattr(
        ltsn_training,
        "require_surrogate_training_gate",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        ltsn_training,
        "load_training_config",
        lambda _path: (
            ltsn_training.LTSNConfig(),
            LTSNTrainingConfig(),
            ltsn_training.LTSNLossWeights(),
        ),
    )
    monkeypatch.setattr(ltsn_training, "read_ltsn_manifest", lambda *_args: [])

    with pytest.raises(LTSNContractError, match="exactly steps"):
        train_ensemble(
            fingerprint_path=tmp_path / "fingerprint.json",
            manifest_path=manifest,
            split_manifest_path=tmp_path / "split.json",
            config_path=tmp_path / "training.toml",
            output_dir=tmp_path / "models",
            surrogate_training_gate_path=tmp_path / "gate.json",
            engineering_smoke=False,
            device_name="cpu",
        )


def _record_split(recorder: TrajectoryRecorder, split: str, index: int) -> None:
    recorder.begin(prompt_id=f"prompt_{split}", trajectory_id=f"trajectory_{split}", split=split)
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


def test_synthetic_collect_writes_four_snapshots_per_trajectory(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.csv"
    write_csv_atomic(
        prompts,
        [
            {
                "prompt_id": "prompt_train",
                "caption": "soft instrumental focus music",
                "split": "train",
                "seed": "7",
                "bpm": "",
                "keyscale": "",
                "timesignature": "",
            }
        ],
    )
    output = tmp_path / "collection"

    assert (
        collect_main(
            [
                "--root",
                str(ROOT),
                "--ace-config",
                str(ROOT / "configs" / "ace_rerank_180s.toml"),
                "--prompt-manifest",
                str(prompts),
                "--output-dir",
                str(output),
                "--backend",
                "synthetic",
                "--ace-model-sha256",
                "a" * 64,
                "--vae-sha256",
                "b" * 64,
                "--engineering-smoke",
            ]
        )
        == 0
    )
    with (output / "trajectory_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    validate_snapshot_coverage(rows)


def test_four_shard_collection_resumes_and_merges_deterministically(
    tmp_path: Path,
) -> None:
    prompts = tmp_path / "prompts.csv"
    write_csv_atomic(
        prompts,
        [
            {
                "prompt_id": f"prompt_{index}",
                "caption": f"soft instrumental focus music {index}",
                "split": "train",
                "seed": "",
                "bpm": "",
                "keyscale": "",
                "timesignature": "",
            }
            for index in range(2)
        ],
    )
    trajectories = tmp_path / "trajectories"
    common = [
        "--root",
        str(ROOT),
        "--ace-config",
        str(ROOT / "configs" / "ace_rerank_180s.toml"),
        "--prompt-manifest",
        str(prompts),
        "--backend",
        "synthetic",
        "--ace-model-sha256",
        "a" * 64,
        "--vae-sha256",
        "b" * 64,
        "--seeds-per-prompt",
        "4",
        "--engineering-smoke",
        "--resume",
    ]
    for shard_index in range(4):
        assert (
            collect_main(
                [
                    *common,
                    "--output-dir",
                    str(trajectories / "shards" / f"shard_{shard_index:02d}"),
                    "--shard-index",
                    str(shard_index),
                    "--shard-count",
                    "4",
                ]
            )
            == 0
        )
    first_manifest = trajectories / "shards" / "shard_00" / "trajectory_manifest.csv"
    first_digest = first_manifest.read_bytes()
    assert (
        collect_main(
            [
                *common,
                "--output-dir",
                str(trajectories / "shards" / "shard_00"),
                "--shard-index",
                "0",
                "--shard-count",
                "4",
            ]
        )
        == 0
    )
    assert first_manifest.read_bytes() == first_digest

    assert (
        merge_main(
            [
                "--shards-root",
                str(trajectories / "shards"),
                "--shard-count",
                "4",
                "--prompt-manifest",
                str(prompts),
                "--output-dir",
                str(trajectories),
                "--seeds-per-prompt",
                "4",
            ]
        )
        == 0
    )
    with (trajectories / "trajectory_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len({row["trajectory_id"] for row in rows}) == 8
    assert len(rows) == 32
    validate_snapshot_coverage(rows)


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
        surrogate_training_gate_path=None,
        engineering_smoke=True,
        device_name="cpu",
    )
    assert result["status"] == "engineering_smoke_only"
    assert result["devices"] == ["cpu"]
    assert not result["parallel_training"]
    assert len(result["checkpoints"]) == 1
    assert (tmp_path / "models" / result["checkpoints"][0]["path"]).is_file()
