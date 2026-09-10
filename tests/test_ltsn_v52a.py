from __future__ import annotations

import csv
import tomllib
from pathlib import Path

import numpy as np
import pytest

from generation.ltsn_pipeline import write_csv_atomic
from generation.ltsn_v52a import (
    _plan_items,
    build_v52a_views,
    mcnemar_exact,
    orthogonal_smooth_directions,
)

ROOT = Path(__file__).resolve().parents[1]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v52a_directions_are_deterministic_unit_rms_and_orthogonal() -> None:
    latent = np.arange(96 * 64, dtype=np.float32).reshape(96, 64) / 1000.0

    first, audit = orthogonal_smooth_directions(latent, anchor_id="anchor", count=8, seed=20260910)
    second, second_audit = orthogonal_smooth_directions(
        latent, anchor_id="anchor", count=8, seed=20260910
    )

    assert audit == second_audit
    assert audit["maximum_absolute_pairwise_cosine"] < 1e-5
    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))
    assert all(
        np.sqrt(np.mean(value.astype(np.float64) ** 2)) == pytest.approx(1.0) for value in first
    )
    matrix = np.stack([value.reshape(-1) for value in first]).astype(np.float64)
    cosine = (
        matrix
        @ matrix.T
        / np.sqrt(
            np.sum(matrix * matrix, axis=1)[:, None] * np.sum(matrix * matrix, axis=1)[None, :]
        )
    )
    assert np.max(np.abs(cosine - np.eye(8))) < 1e-5


def test_v52a_plan_freezes_six_train_and_two_holdout_directions(tmp_path: Path) -> None:
    latent = tmp_path / "anchor.npy"
    np.save(latent, np.ones((64, 64), dtype=np.float32), allow_pickle=False)
    anchors = [
        {
            "sample_id": "train_anchor",
            "prompt_id": "train_prompt",
            "step_number": "4",
            "timestep": "0.5",
            "v52a_anchor_partition": "train_anchor",
            "_resolved_latent_path": str(latent),
        },
        {
            "sample_id": "unseen_anchor",
            "prompt_id": "unseen_prompt",
            "step_number": "5",
            "timestep": "0.4",
            "v52a_anchor_partition": "unseen_anchor",
            "_resolved_latent_path": str(latent),
        },
    ]

    planned, audit = _plan_items(
        anchors,
        direction_count=8,
        train_direction_count=6,
        rms_ratios=(0.0025, 0.005),
        seed=20260910,
    )

    assert len(planned) == 2 * 8 * 2 * 2
    assert sum(row["direction_partition"] == "train_direction" for row in planned) == 24
    assert sum(row["direction_partition"] == "heldout_direction" for row in planned) == 8
    assert sum(row["direction_partition"] == "unseen_anchor" for row in planned) == 32
    assert len({row["direction_group_id"] for row in planned}) == 2 * 8 * 2
    assert all(value["maximum_absolute_pairwise_cosine"] < 1e-5 for value in audit.values())


def test_v52a_views_separate_direction_and_anchor_holdouts(tmp_path: Path) -> None:
    source = tmp_path / "collection"
    source.mkdir()
    labels = source / "labels.csv"
    labels.write_text("sample_id\nbase\n", encoding="utf-8")
    manifest_rows = []
    evidence_rows = []
    anchor_specs = [
        (f"train_s{step}_{index}", "train_anchor", step) for step in (4, 5, 6) for index in range(2)
    ]
    anchor_specs.extend((f"unseen_s{step}", "unseen_anchor", step) for step in (4, 5, 6))
    for anchor_index, (anchor, anchor_partition, step) in enumerate(anchor_specs):
        for direction_index in range(8):
            direction_partition = (
                "unseen_anchor"
                if anchor_partition == "unseen_anchor"
                else ("train_direction" if direction_index < 6 else "heldout_direction")
            )
            for rms_index, rms in enumerate((0.0025, 0.005)):
                group = f"{anchor}_d{direction_index}_r{rms}"
                minus_id = f"{group}_minus"
                plus_id = f"{group}_plus"
                base = anchor_index + direction_index * 0.1 + rms_index * 0.01
                evidence_rows.append(
                    {
                        "direction_group_id": group,
                        "anchor_sample_id": anchor,
                        "anchor_partition": anchor_partition,
                        "direction_partition": direction_partition,
                        "direction_index": direction_index,
                        "direction_seed": direction_index + 100,
                        "step_number": step,
                        "rms_ratio": rms,
                        "minus_sample_id": minus_id,
                        "plus_sample_id": plus_id,
                        "exact_loss_minus": base + 0.2,
                        "exact_loss_plus": base,
                        "exact_derivative": 0.2 / (2 * rms),
                        "informative": True,
                        "pair_in_distribution": True,
                    }
                )
                for sample_id, sign, focus in (
                    (minus_id, -1, base + 0.2),
                    (plus_id, 1, base),
                ):
                    latent = source / f"{sample_id}.npy"
                    np.save(latent, np.ones((8, 64), dtype=np.float32), allow_pickle=False)
                    manifest_rows.append(
                        {
                            "sample_id": sample_id,
                            "prompt_id": f"prompt_{anchor}",
                            "trajectory_id": sample_id,
                            "split": (
                                "train"
                                if direction_partition == "train_direction"
                                else "development"
                            ),
                            "step_number": step,
                            "latent_path": latent.name,
                            "exact_label_table_path": labels.name,
                            "focus_logit": focus,
                            "local_anchor_sample_id": "",
                            "local_direction_group_id": group,
                            "local_direction_sign": sign,
                            "local_direction_rms_ratio": rms,
                            "v52a_direction_index": direction_index,
                        }
                    )
    master = source / "master.csv"
    evidence = source / "evidence.csv"
    write_csv_atomic(master, manifest_rows)
    write_csv_atomic(evidence, evidence_rows)

    summary = build_v52a_views(
        master_manifest_path=master,
        evidence_path=evidence,
        output_root=tmp_path / "views",
    )

    assert summary["informative_pairs_by_partition"] == {
        "train_direction": 72,
        "heldout_direction": 24,
        "unseen_anchor": 48,
    }
    true_rows = _read_rows(tmp_path / "views" / "true_pairs" / "ltsn_manifest_v52a.csv")
    seen_rows = _read_rows(
        tmp_path / "views" / "seen_anchor_heldout_direction" / "ltsn_manifest_v52a.csv"
    )
    control_rows = _read_rows(
        tmp_path / "views" / "permuted_pair_control" / "ltsn_manifest_v52a.csv"
    )
    assert sum(row["split"] == "train" for row in true_rows) == 72 * 2
    assert sum(row["split"] == "development" for row in true_rows) == 48 * 2
    assert len(seen_rows) == 24 * 2
    assert len(control_rows) == len(true_rows)
    true_group = {row["sample_id"]: row["local_direction_group_id"] for row in true_rows}
    control_group = {row["sample_id"]: row["local_direction_group_id"] for row in control_rows}
    control_by_id = {row["sample_id"]: row for row in control_rows}
    assert any(
        true_group[sample_id] != control_group[sample_id]
        for sample_id in true_group
        if control_by_id[sample_id]["split"] == "train"
    )
    assert summary["views"]["permuted_pair_control"]["permuted_train_pairs"] == 72


def test_v52a_config_is_three_seed_central_pair_diagnostic() -> None:
    with (ROOT / "configs" / "ltsn_training_v52a_identifiability.toml").open("rb") as handle:
        config = tomllib.load(handle)
    assert config["model"]["dropout"] == 0.0
    assert config["model"]["stem_dropout"] == 0.0
    assert config["training"]["weight_decay"] == 0.0
    assert config["training"]["seeds"] == [20260716, 20260717, 20260718]
    assert config["training"]["central_direction_grouped_batches"] is True
    assert config["training"]["minimum_epochs"] == config["training"]["max_epochs"]
    assert config["loss"]["central_direction"] == 1.0
    assert all(value == 0 for name, value in config["loss"].items() if name != "central_direction")


def test_v52a_mcnemar_uses_paired_disagreements() -> None:
    result = mcnemar_exact(
        {"a": True, "b": True, "c": True, "d": False},
        {"a": False, "b": False, "c": True, "d": True},
    )
    assert result["true_only_correct"] == 2
    assert result["control_only_correct"] == 1
    assert result["discordant_pairs"] == 3
    assert result["two_sided_exact_p"] == 1.0
