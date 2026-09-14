from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from generation.ltsn_contract import sha256_file
from generation.ltsn_pipeline import write_csv_atomic, write_json_atomic
from generation.pitch3_contract import load_pitch3_contract
from generation.pitch3_ood import (
    EVALUATION_OOD_KINDS,
    TRAIN_OOD_KINDS,
    apply_pitch3_ood_transform,
    build_pitch3_ood_augmentation,
    build_pitch3_ood_plan,
)
from topology.metrics import TOPOLOGY_METRICS

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "metadata" / "focus_pitch3_fingerprint_v1.json"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_pitch3_ood_plan_is_deterministic_split_local_and_held_out() -> None:
    rows = []
    for split in ("train", "development", "calibration", "qualification"):
        steps = (4, 5, 6, 8) if split in {"calibration", "qualification"} else (4, 5, 6)
        for prompt_index in range(2):
            for step in steps:
                for trajectory_index in range(2):
                    rows.append(
                        {
                            "sample_id": f"{split}_{prompt_index}_{step}_{trajectory_index}",
                            "prompt_id": f"{split}_prompt_{prompt_index}",
                            "trajectory_id": f"{split}_trajectory_{trajectory_index}",
                            "split": split,
                            "step_number": str(step),
                            "timestep": "0.5",
                            "is_final": str(step == 8).lower(),
                            "latent_sha256": "a" * 64,
                            "ood_label": "0.0",
                        }
                    )

    first = build_pitch3_ood_plan(rows, ood_per_prompt=1, evaluation_ood_per_prompt=1)
    second = build_pitch3_ood_plan(rows, ood_per_prompt=1, evaluation_ood_per_prompt=1)

    assert first == second
    assert len(first) == 28
    train_kinds = {
        row["ood_kind"] for row in first if row["split"] in {"train", "development"}
    }
    evaluation_kinds = {
        row["ood_kind"]
        for row in first
        if row["split"] in {"calibration", "qualification"}
    }
    assert train_kinds == set(TRAIN_OOD_KINDS)
    assert evaluation_kinds == set(EVALUATION_OOD_KINDS)
    assert all(row["source_sample_id"].startswith(row["split"]) for row in first)


def test_pitch3_ood_transforms_preserve_finite_shape() -> None:
    latent = np.arange(6 * 64, dtype=np.float32).reshape(6, 64) / 100.0

    zero = apply_pitch3_ood_transform(latent, "ood_zero")
    scaled = apply_pitch3_ood_transform(latent, "ood_scale_high")
    reversed_latent = apply_pitch3_ood_transform(latent, "ood_time_reverse")
    rolled = apply_pitch3_ood_transform(latent, "ood_channel_roll")

    assert zero.shape == latent.shape and np.count_nonzero(zero) == 0
    assert np.array_equal(scaled, latent * 4.0)
    assert np.array_equal(reversed_latent, latent[::-1])
    assert np.array_equal(rolled, np.roll(latent, 17, axis=1))
    assert all(np.isfinite(value).all() for value in (zero, scaled, reversed_latent, rolled))


def test_pitch3_ood_builder_emits_hash_bound_merged_manifest(
    tmp_path: Path, monkeypatch: Any
) -> None:
    contract = load_pitch3_contract(PROFILE)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_rows: list[dict[str, Any]] = []
    source_labels: list[dict[str, Any]] = []
    assignments: dict[str, str] = {}
    pitch = np.zeros(len(TOPOLOGY_METRICS), dtype=float)
    pitch[TOPOLOGY_METRICS.index("h0_observed_persistence")] = 4.0
    pitch[TOPOLOGY_METRICS.index("self_transition_ratio")] = 0.4
    pitch[TOPOLOGY_METRICS.index("directed_recurrence")] = 0.05
    for split in ("train", "development", "calibration", "qualification"):
        prompt_id = f"{split}_prompt"
        assignments[prompt_id] = split
        steps = (4, 5, 6, 8) if split in {"calibration", "qualification"} else (4, 5, 6)
        for step in steps:
            sample_id = f"{split}_step{step}"
            latent_path = source_dir / f"{sample_id}.npy"
            np.save(latent_path, np.full((8, 64), step / 10.0, dtype=np.float32))
            source_rows.append(
                {
                    "sample_id": sample_id,
                    "prompt_id": prompt_id,
                    "trajectory_id": f"{split}_trajectory",
                    "split": split,
                    "model_family": "acestep-v15-xl-turbo",
                    "step_number": step,
                    "timestep": 0.5,
                    "latent_path": latent_path.name,
                    "latent_sha256": sha256_file(latent_path),
                    "coordinates_json": "[0.0,0.0,0.0]",
                    "focus_logit": 0.0,
                    "ood_label": 0.0,
                    "fingerprint_json_sha256": contract.artifact_sha256,
                    "feature_order_json": json.dumps(list(contract.feature_order)),
                    "label_scope": "per_snapshot_exact",
                    "exact_label_table_sha256": "",
                    "exact_label_table_path": "pitch3_exact_labels.csv",
                    "is_final": str(step == 8).lower(),
                    "ace_model_sha256": "a" * 64,
                    "vae_sha256": "b" * 64,
                    "qualification_eligible": "true",
                    "guidance_promotion_eligible": "false",
                }
            )
            source_labels.append(
                {
                    "sample_id": sample_id,
                    "pitch_descriptors_json": json.dumps(pitch.tolist()),
                    "coordinates_json": "[0.0,0.0,0.0]",
                    "focus_logit": 0.0,
                    "focus_probability": 0.5,
                    "focus_target_band_loss": 0.0,
                    "target_center_distance": 0.0,
                    "ood_label": 0.0,
                    "label_scope": "per_snapshot_exact",
                    "fingerprint_json_sha256": contract.artifact_sha256,
                }
            )
    exact_path = source_dir / "pitch3_exact_labels.csv"
    write_csv_atomic(exact_path, source_labels)
    for row in source_rows:
        row["exact_label_table_sha256"] = sha256_file(exact_path)
    source_manifest = source_dir / "pitch3_training_manifest.csv"
    write_csv_atomic(source_manifest, source_rows)
    source_split = source_dir / "pitch3_split_manifest.json"
    write_json_atomic(
        source_split,
        {
            "schema_version": 1,
            "fingerprint_json_sha256": contract.artifact_sha256,
            "assignments": dict(sorted(assignments.items())),
            "qualification_eligible": True,
            "guidance_promotion_eligible": False,
        },
    )
    ace_config = tmp_path / "ace.toml"
    ace_config.write_text("test = true\n", encoding="utf-8")

    class FakeAdapter:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def decode_latent_to_audio(self, latent: np.ndarray, output_path: Path) -> Path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(np.asarray(latent, dtype=np.float32).tobytes())
            return output_path

    def fake_exact_builder(**kwargs: Any) -> dict[str, Any]:
        trajectories = _read_csv(kwargs["trajectory_manifest"])
        descriptors = [
            {
                "sample_id": row["sample_id"],
                "pitch_descriptors_json": json.dumps(pitch.tolist()),
                "acoustic_loop_score": 0.0,
                "chroma_loop_score": 0.0,
                "ood_label": int(row["ood_kind"] == "ood_zero"),
                "label_source": "decoded_snapshot_exact_v1",
                "audio_sha256": row["audio_sha256"],
                "pitch_v2_codebook_sha256": "c" * 64,
            }
            for row in trajectories
        ]
        write_csv_atomic(kwargs["output_path"], descriptors)
        return {
            "samples": len(descriptors),
            "descriptor_table_sha256": sha256_file(kwargs["output_path"]),
        }

    monkeypatch.setattr("generation.pitch3_ood.AceStepAdapter", FakeAdapter)
    monkeypatch.setattr(
        "generation.pitch3_ood.load_experiment_config",
        lambda *_args, **_kwargs: SimpleNamespace(ace=SimpleNamespace(checkout="ACE-Step-1.5")),
    )
    monkeypatch.setattr(
        "generation.pitch3_ood.build_exact_snapshot_descriptors", fake_exact_builder
    )
    output_dir = tmp_path / "ood"
    result = build_pitch3_ood_augmentation(
        root=tmp_path,
        source_manifest_path=source_manifest,
        source_split_manifest_path=source_split,
        ace_config_path=ace_config,
        fingerprint_path=PROFILE,
        output_dir=output_dir,
        workers=1,
        exact_batch_size=8,
        device_name="cpu",
    )

    assert result["source_samples"] == 14
    assert result["ood_samples"] == 14
    assert result["combined_samples"] == 28
    assert result["counts_by_split"] == {
        "train": {"id": 3, "ood": 3},
        "development": {"id": 3, "ood": 3},
        "calibration": {"id": 4, "ood": 4},
        "qualification": {"id": 4, "ood": 4},
    }
    combined_path = output_dir / "pitch3_training_manifest_augmented.csv"
    combined = _read_csv(combined_path)
    assert sha256_file(combined_path) == result["combined_training_manifest_sha256"]
    assert {row["ood_label"] for row in combined} == {"0.0", "1.0"}
    assert all(
        row["ood_label_source"] == "deterministic_latent_transform_v1"
        for row in combined
        if row["ood_label"] == "1.0"
    )
    second = build_pitch3_ood_augmentation(
        root=tmp_path,
        source_manifest_path=source_manifest,
        source_split_manifest_path=source_split,
        ace_config_path=ace_config,
        fingerprint_path=PROFILE,
        output_dir=output_dir,
        workers=1,
        exact_batch_size=8,
        device_name="cpu",
        resume=True,
    )
    assert second["combined_training_manifest_sha256"] == result[
        "combined_training_manifest_sha256"
    ]
