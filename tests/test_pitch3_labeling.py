from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from generation.ltsn_contract import sha256_file
from generation.ltsn_pipeline import (
    TrajectorySnapshotRecord,
    write_csv_atomic,
)
from generation.pitch3_exact_scorer import ExactPitch3Scorer
from generation.pitch3_labeling import build_pitch3_label_tables
from topology.metrics import TOPOLOGY_METRICS

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "metadata" / "focus_pitch3_fingerprint_v1.json"


def test_pitch3_label_builder_issues_hashed_non_authorizing_smoke_manifest(
    tmp_path: Path,
) -> None:
    collection = tmp_path / "collection"
    latents = collection / "latents"
    latents.mkdir(parents=True)
    trajectories = []
    descriptors = []
    for index, split in enumerate(("train", "development", "calibration", "qualification")):
        sample_id = f"sample_{index}"
        latent_path = latents / f"{sample_id}.npy"
        np.save(latent_path, np.full((12, 64), index, dtype=np.float32))
        trajectories.append(
            asdict(
                TrajectorySnapshotRecord(
                    sample_id=sample_id,
                    prompt_id=f"prompt_{index}",
                    trajectory_id=f"trajectory_{index}",
                    split=split,
                    model_family="acestep-v15-xl-turbo",
                    step_number=4,
                    timestep=0.5,
                    is_final=False,
                    latent_path=latent_path.relative_to(collection).as_posix(),
                    latent_sha256=sha256_file(latent_path),
                    audio_path="",
                    audio_sha256="",
                    ace_model_sha256="a" * 64,
                    vae_sha256="b" * 64,
                    engineering_smoke=True,
                )
            )
        )
        pitch = np.zeros(len(TOPOLOGY_METRICS), dtype=float)
        pitch[TOPOLOGY_METRICS.index("h0_observed_persistence")] = 4.0 + index
        pitch[TOPOLOGY_METRICS.index("self_transition_ratio")] = 0.4 + index / 100
        pitch[TOPOLOGY_METRICS.index("directed_recurrence")] = 0.05 + index / 100
        descriptors.append(
            {
                "sample_id": sample_id,
                "pitch_descriptors_json": json.dumps(pitch.tolist()),
                "ood_label": float(index % 2),
                "label_source": "synthetic_smoke_only",
                "audio_sha256": "",
            }
        )
    trajectory_manifest = collection / "trajectory_manifest.csv"
    descriptor_table = collection / "descriptors.csv"
    write_csv_atomic(trajectory_manifest, trajectories)
    write_csv_atomic(descriptor_table, descriptors)
    output_manifest = tmp_path / "labels" / "manifest.csv"
    exact_table = tmp_path / "labels" / "exact.csv"

    result = build_pitch3_label_tables(
        trajectory_manifest=trajectory_manifest,
        descriptor_table=descriptor_table,
        output_manifest=output_manifest,
        exact_label_table=exact_table,
        split_manifest=tmp_path / "labels" / "splits.json",
        scorer=ExactPitch3Scorer.from_json(PROFILE),
        engineering_smoke=True,
    )

    assert result["samples"] == 4
    assert result["qualification_eligible"] is False
    assert result["guidance_promotion_eligible"] is False
    assert result["exact_label_table_sha256"] == sha256_file(exact_table)
    with output_manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert all(len(json.loads(row["coordinates_json"])) == 3 for row in rows)
    assert all(row["guidance_promotion_eligible"] == "false" for row in rows)
