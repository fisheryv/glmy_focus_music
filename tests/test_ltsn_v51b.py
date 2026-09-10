from __future__ import annotations

import csv
import json
import tomllib
from pathlib import Path

import pytest

from generation.ltsn_pipeline import write_csv_atomic
from generation.ltsn_v51b import (
    build_v51b_memorization_ladder,
    memorization_agreement_threshold,
    nested_stratified_anchor_order,
    select_peak_memorization_epoch,
)

ROOT = Path(__file__).resolve().parents[1]


def _evidence(anchor: str, split: str, step: int, sign: int) -> list[dict[str, str]]:
    rows = []
    for rms in (0.0025, 0.005):
        difference = sign * rms * 10
        rows.append(
            {
                "direction_group_id": f"{anchor}_r{rms}",
                "anchor_sample_id": anchor,
                "split": split,
                "step_number": str(step),
                "rms_ratio": str(rms),
                "minus_sample_id": f"{anchor}_r{rms}_minus",
                "plus_sample_id": f"{anchor}_r{rms}_plus",
                "exact_loss_minus": str(max(difference, 0.0)),
                "exact_loss_plus": str(max(-difference, 0.0)),
                "exact_derivative": str(sign),
            }
        )
    return rows


def _anchor_specs(count: int, split: str) -> list[tuple[str, str, int, int]]:
    strata = ((4, -1), (5, 1), (6, -1), (4, 1), (5, -1), (6, 1))
    return [(f"{split}_{index:03d}", split, *strata[index % len(strata)]) for index in range(count)]


def test_v51b_order_is_deterministic_stratified_and_nested() -> None:
    rows = []
    specs = _anchor_specs(72, "train")
    for anchor, split, step, sign in reversed(specs):
        rows.extend(_evidence(anchor, split, step, sign))

    first, audit = nested_stratified_anchor_order(rows, split="train")
    second, _ = nested_stratified_anchor_order(list(reversed(rows)), split="train")

    assert first == second
    assert len(first) == len(set(first)) == 72
    assert set(first[:1]) < set(first[:4]) < set(first[:16]) < set(first[:64])
    first_six = {anchor: (step, sign) for anchor, _, step, sign in specs if anchor in first[:6]}
    assert set(first_six.values()) == {
        (4, -1),
        (5, 1),
        (6, -1),
        (4, 1),
        (5, -1),
        (6, 1),
    }
    assert audit["anchors"] == 72


def test_v51b_ladder_reuses_files_and_has_expected_rows(tmp_path: Path) -> None:
    source = tmp_path / "v51a"
    output = tmp_path / "v51b"
    source.mkdir()
    label = source / "labels.csv"
    label.write_text("sample_id\nbase\n", encoding="utf-8")
    evidence_rows = []
    manifest_rows = []
    specs = _anchor_specs(64, "train") + _anchor_specs(12, "development")
    for anchor, split_name, step, sign in specs:
        evidence_rows.extend(_evidence(anchor, split_name, step, sign))
        sample_specs = [(anchor, "", "", "")]
        for rms in (0.0025, 0.005):
            for direction in (-1, 1):
                sample_specs.append(
                    (
                        f"{anchor}_r{rms}_{direction}",
                        f"{anchor}_r{rms}",
                        str(direction),
                        str(rms),
                    )
                )
        for sample_id, group, direction, rms in sample_specs:
            latent = source / f"{sample_id}.npy"
            latent.write_bytes(b"latent")
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "prompt_id": f"prompt_{anchor}",
                    "split": split_name,
                    "latent_path": latent.name,
                    "exact_label_table_path": label.name,
                    "training_augmentation_kind": "on_policy_symmetric" if group else "",
                    "local_anchor_sample_id": anchor if group else "",
                    "local_direction_group_id": group,
                    "local_direction_sign": direction,
                    "local_direction_rms_ratio": rms,
                }
            )
    manifest = source / "manifest.csv"
    evidence = source / "evidence.csv"
    split = source / "split.json"
    write_csv_atomic(manifest, manifest_rows)
    write_csv_atomic(evidence, evidence_rows)
    split.write_text(json.dumps({"assignments": {}}), encoding="utf-8")

    summary = build_v51b_memorization_ladder(
        source_manifest_path=manifest,
        source_split_manifest_path=split,
        central_evidence_path=evidence,
        output_root=output,
    )

    assert summary["rung_sizes"] == [1, 4, 16, 64]
    assert summary["development_anchor_count"] == 6
    for rung, count in zip(summary["rungs"], (1, 4, 16, 64), strict=True):
        assert rung["rows_by_split"] == {"train": count * 5, "development": 30}
        assert rung["qualification_eligible"] is False
        assert rung["guidance_promotion_eligible"] is False
        rung_dir = output / rung["name"]
        with (rung_dir / "ltsn_manifest_v51b.csv").open(encoding="utf-8", newline="") as handle:
            retained = list(csv.DictReader(handle))
        assert len(retained) == count * 5 + 30
        assert all((rung_dir / row["latent_path"]).resolve().is_file() for row in retained)


def test_v51b_config_disables_regularization_and_is_central_only() -> None:
    with (ROOT / "configs" / "ltsn_training_v51b_memorization.toml").open("rb") as handle:
        config = tomllib.load(handle)

    assert config["model"]["dropout"] == 0.0
    assert config["model"]["stem_dropout"] == 0.0
    assert config["training"]["weight_decay"] == 0.0
    assert config["training"]["use_bf16"] is False
    assert config["training"]["seeds"] == [20260716]
    assert config["training"]["minimum_epochs"] == config["training"]["max_epochs"]
    assert config["training"]["central_direction_overfit_diagnostic"] is True
    assert config["training"]["central_direction_classification_only"] is True
    assert config["loss"]["central_direction"] == 1.0
    assert all(value == 0 for name, value in config["loss"].items() if name != "central_direction")


def test_v51b_thresholds_and_peak_epoch_are_preregistered() -> None:
    assert [memorization_agreement_threshold(count) for count in (1, 4, 16, 64)] == [
        1.0,
        1.0,
        0.99,
        0.98,
    ]
    with pytest.raises(ValueError, match="no frozen"):
        memorization_agreement_threshold(8)
    history = [
        {
            "epoch": 1,
            "train_loss": 0.2,
            "overfit_train_central_direction_agreement": 0.9,
            "overfit_train_central_derivative_spearman": 0.8,
        },
        {
            "epoch": 2,
            "train_loss": 0.1,
            "overfit_train_central_direction_agreement": 1.0,
            "overfit_train_central_derivative_spearman": 0.7,
        },
        {
            "epoch": 3,
            "train_loss": 0.1,
            "overfit_train_central_direction_agreement": 1.0,
            "overfit_train_central_derivative_spearman": 0.9,
        },
    ]
    assert select_peak_memorization_epoch(history)["epoch"] == 3
