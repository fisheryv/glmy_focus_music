from __future__ import annotations

import csv
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from generation.ltsn_pipeline import write_csv_atomic
from generation.ltsn_v51c import (
    V51C_VARIANTS,
    build_v51c_ablation_suite,
    wilson_interval,
)

ROOT = Path(__file__).resolve().parents[1]


def _specs(count: int, split: str) -> list[tuple[str, str, int, int]]:
    strata = ((4, -1), (5, 1), (6, -1), (4, 1), (5, -1), (6, 1))
    return [(f"{split}_{index:03d}", split, *strata[index % len(strata)]) for index in range(count)]


def _source(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "v51a"
    source.mkdir()
    label = source / "labels.csv"
    label.write_text("sample_id\nbase\n", encoding="utf-8")
    manifest_rows = []
    evidence_rows = []
    specs = _specs(64, "train") + _specs(78, "development")
    for anchor_index, (anchor, split_name, step, sign) in enumerate(specs):
        anchor_latent = source / f"{anchor}.npy"
        anchor_latent.write_bytes(b"latent")
        manifest_rows.append(
            {
                "sample_id": anchor,
                "prompt_id": f"prompt_{anchor}",
                "split": split_name,
                "latent_path": anchor_latent.name,
                "exact_label_table_path": label.name,
                "focus_logit": str(anchor_index),
                "training_augmentation_kind": "",
                "local_anchor_sample_id": "",
                "local_direction_group_id": "",
                "local_direction_sign": "0",
                "local_direction_rms_ratio": "0",
            }
        )
        for rms_index, rms in enumerate((0.0025, 0.005)):
            group = f"{anchor}_r{rms}"
            base = anchor_index * 0.01 + rms_index * 0.001
            exact_minus = base + (0.2 if sign > 0 else 0.0)
            exact_plus = base + (0.2 if sign < 0 else 0.0)
            minus_id = f"{group}_minus"
            plus_id = f"{group}_plus"
            evidence_rows.append(
                {
                    "direction_group_id": group,
                    "anchor_sample_id": anchor,
                    "split": split_name,
                    "step_number": str(step),
                    "rms_ratio": str(rms),
                    "minus_sample_id": minus_id,
                    "plus_sample_id": plus_id,
                    "exact_loss_minus": str(exact_minus),
                    "exact_loss_plus": str(exact_plus),
                    "exact_derivative": str(sign),
                }
            )
            for sample_id, direction, focus in (
                (minus_id, -1, exact_minus),
                (plus_id, 1, exact_plus),
            ):
                latent = source / f"{sample_id}.npy"
                latent.write_bytes(b"latent")
                manifest_rows.append(
                    {
                        "sample_id": sample_id,
                        "prompt_id": f"prompt_{anchor}",
                        "split": split_name,
                        "latent_path": latent.name,
                        "exact_label_table_path": label.name,
                        "focus_logit": str(focus),
                        "training_augmentation_kind": "on_policy_symmetric",
                        "local_anchor_sample_id": anchor,
                        "local_direction_group_id": group,
                        "local_direction_sign": str(direction),
                        "local_direction_rms_ratio": str(rms),
                    }
                )
    manifest = source / "manifest.csv"
    evidence = source / "evidence.csv"
    split = source / "split.json"
    write_csv_atomic(manifest, manifest_rows)
    write_csv_atomic(evidence, evidence_rows)
    split.write_text(json.dumps({"assignments": {}}), encoding="utf-8")
    return manifest, split, evidence


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _flatten(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{section}.{name}": value
        for section, values in payload.items()
        for name, value in values.items()
    }


def test_v51c_builds_full_development_and_exact_pair_control(tmp_path: Path) -> None:
    manifest, split, evidence = _source(tmp_path)
    output = tmp_path / "v51c"

    suite = build_v51c_ablation_suite(
        source_manifest_path=manifest,
        source_split_manifest_path=split,
        central_evidence_path=evidence,
        config_root=ROOT / "configs",
        output_root=output,
    )

    assert suite["train_anchors"] == 64
    assert suite["development_anchors"] == 78
    assert suite["development_direction_pairs"] == 156
    assert suite["qualification_eligible"] is False
    assert suite["guidance_promotion_eligible"] is False
    assert {row["name"] for row in suite["variants"]} == set(V51C_VARIANTS)
    true_rows = _read_rows(output / "true_pairs" / "ltsn_manifest_v51c.csv")
    control_rows = _read_rows(output / "permuted_pair_control" / "ltsn_manifest_v51c.csv")
    assert len(true_rows) == len(control_rows) == 64 * 5 + 78 * 5
    assert sum(row["split"] == "development" for row in true_rows) == 78 * 5
    true_by_id = {row["sample_id"]: row for row in true_rows}
    control_by_id = {row["sample_id"]: row for row in control_rows}
    assert all(
        control_by_id[sample_id]["focus_logit"] == row["focus_logit"]
        for sample_id, row in true_by_id.items()
    )
    assert all(
        control_by_id[sample_id]["local_direction_group_id"] == row["local_direction_group_id"]
        for sample_id, row in true_by_id.items()
        if row["split"] == "development"
    )
    pair_map = _read_rows(output / "permuted_pair_control" / "permuted_pair_map_v51c.csv")
    assert len(pair_map) == 128
    assert all(row["minus_source_group_id"] != row["plus_source_group_id"] for row in pair_map)
    permutation = suite["views"]["permuted_pair_control"]["permutation"]
    assert permutation["preserves_exact_sample_labels"] is True
    assert permutation["original_train_pairs_retained"] == 0
    assert permutation["informative_pairs"] <= 128
    assert all(
        (output / "true_pairs" / row["latent_path"]).resolve().is_file() for row in true_rows
    )


def test_v51c_configs_restore_only_the_named_factor() -> None:
    configs = {}
    for name, specification in V51C_VARIANTS.items():
        if name == "permuted_pair_control":
            continue
        with (ROOT / "configs" / specification["config"]).open("rb") as handle:
            configs[name] = _flatten(tomllib.load(handle))
    baseline = configs["baseline"]
    with (ROOT / "configs" / "ltsn_training_v51b_memorization.toml").open("rb") as handle:
        assert baseline == _flatten(tomllib.load(handle))
    expected_differences = {
        "restore_dropout": {"model.dropout", "model.stem_dropout"},
        "restore_weight_decay": {"training.weight_decay"},
        "restore_lr_schedule": {
            "training.learning_rate",
            "training.warmup_fraction",
            "training.minimum_learning_rate",
        },
        "restore_batch32": {"training.effective_batch_size"},
        "restore_short_early_stop": {
            "training.max_epochs",
            "training.minimum_epochs",
            "training.early_stopping_patience",
        },
    }
    for name, expected in expected_differences.items():
        actual = {key for key, value in configs[name].items() if value != baseline[key]}
        assert actual == expected
        assert configs[name]["training.seeds"] == [20260716]
        assert configs[name]["training.use_bf16"] is False
        assert configs[name]["loss.central_direction"] == 1.0
        assert all(
            value == 0
            for key, value in configs[name].items()
            if key.startswith("loss.") and key != "loss.central_direction"
        )
    with (ROOT / "configs" / "ltsn_training_v51c_permuted_pair_control.toml").open("rb") as handle:
        control = tomllib.load(handle)
    assert control["training"]["prompt_grouped_batches"] is False
    assert control["training"]["central_direction_grouped_batches"] is True
    assert control["training"]["seeds"] == [20260716]


def test_v51c_wilson_interval_matches_full_development_scale() -> None:
    low, high = wilson_interval(94, 156)
    assert low == pytest.approx(0.5242, abs=1e-4)
    assert high == pytest.approx(0.6761, abs=1e-4)
    with pytest.raises(ValueError, match="invalid Wilson"):
        wilson_interval(2, 1)
