from __future__ import annotations

import csv
import json
import tomllib
from pathlib import Path

from generation.ltsn_pipeline import write_csv_atomic
from generation.ltsn_v51 import _stable_anchor_selection, build_v51_training_view

ROOT = Path(__file__).resolve().parents[1]


def _evidence(
    anchor: str,
    rms: float,
    derivative: float,
    *,
    split: str = "train",
    separation: float = 0.1,
) -> dict[str, str]:
    return {
        "direction_group_id": f"{anchor}_r{rms}",
        "anchor_sample_id": anchor,
        "split": split,
        "step_number": "4",
        "rms_ratio": str(rms),
        "minus_sample_id": f"{anchor}_{rms}_minus",
        "plus_sample_id": f"{anchor}_{rms}_plus",
        "exact_loss_minus": str(separation),
        "exact_loss_plus": "0.0",
        "exact_derivative": str(derivative),
        "proxy_derivative": "1.0",
        "direction_agrees": str(derivative > 0).lower(),
    }


def test_v51_selection_keeps_only_cross_rms_stable_effects() -> None:
    rows = [
        _evidence("stable", 0.0025, 1.0),
        _evidence("stable", 0.005, 0.5),
        _evidence("opposite", 0.0025, 1.0),
        _evidence("opposite", 0.005, -1.0),
        _evidence("tied", 0.0025, 0.0),
        _evidence("tied", 0.005, 1.0),
        _evidence("weak", 0.0025, -1.0, separation=1e-6),
        _evidence("weak", 0.005, -1.0, separation=0.1),
    ]

    selected, retained, audit = _stable_anchor_selection(
        rows,
        expected_rms_ratios=(0.0025, 0.005),
        minimum_abs_loss_separation=1e-5,
    )

    assert selected == {"stable"}
    assert len(retained) == 2
    assert audit["train"] == {
        "candidate_anchors": 4,
        "stable_anchors": 1,
        "opposite_sign_anchors": 1,
        "tied_anchors": 1,
        "weak_effect_anchors": 1,
    }


def test_v51_keeps_v5_architecture_and_uses_fp32_raw_differences() -> None:
    with (ROOT / "configs" / "ltsn_training_v5.toml").open("rb") as handle:
        v5 = tomllib.load(handle)
    with (ROOT / "configs" / "ltsn_training_v51.toml").open("rb") as handle:
        v51 = tomllib.load(handle)

    assert v51["model"] == v5["model"]
    assert v51["training"]["use_bf16"] is False
    assert v51["training"]["normalize_central_direction_by_rms"] is False
    assert v51["training"]["central_direction_exact_margin"] == 1e-5
    assert v51["training"]["central_direction_primary_early_stopping"] is True


def test_v51_training_view_reuses_v5_artifacts_and_filters_manifest(tmp_path: Path) -> None:
    source = tmp_path / "v5"
    source.mkdir()
    output = tmp_path / "v51"
    label = source / "labels.csv"
    label.write_text("sample_id\nbase\n", encoding="utf-8")
    rows = []
    for sample_id, split, kind, anchor in (
        ("base_train", "train", "", ""),
        ("base_dev", "development", "", ""),
        ("ood", "train", "ood_zero", ""),
        ("stable_minus", "train", "on_policy_symmetric", "stable_train"),
        ("stable_plus", "train", "on_policy_symmetric", "stable_train"),
        ("stable2_minus", "train", "on_policy_symmetric", "stable_train"),
        ("stable2_plus", "train", "on_policy_symmetric", "stable_train"),
        ("dev_minus", "development", "on_policy_symmetric", "stable_dev"),
        ("dev_plus", "development", "on_policy_symmetric", "stable_dev"),
        ("dev2_minus", "development", "on_policy_symmetric", "stable_dev"),
        ("dev2_plus", "development", "on_policy_symmetric", "stable_dev"),
        ("drop_minus", "train", "on_policy_symmetric", "unstable"),
        ("drop_plus", "train", "on_policy_symmetric", "unstable"),
        ("drop2_minus", "train", "on_policy_symmetric", "unstable"),
        ("drop2_plus", "train", "on_policy_symmetric", "unstable"),
    ):
        latent = source / f"{sample_id}.npy"
        latent.write_bytes(b"latent")
        rms = "0.0025" if "2_" not in sample_id else "0.005"
        sign = "-1" if "minus" in sample_id else "1" if "plus" in sample_id else ""
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "latent_path": latent.name,
                "exact_label_table_path": label.name,
                "training_augmentation_kind": kind,
                "local_anchor_sample_id": anchor,
                "local_direction_group_id": f"{anchor}_{rms}" if anchor else "",
                "local_direction_sign": sign,
                "local_direction_rms_ratio": rms if anchor else "",
            }
        )
    manifest = source / "manifest.csv"
    write_csv_atomic(manifest, rows)
    split = source / "split.json"
    split.write_text(json.dumps({"assignments": {}}), encoding="utf-8")
    evidence = source / "evidence.csv"
    evidence_rows = [
        _evidence("stable_train", 0.0025, 1.0),
        _evidence("stable_train", 0.005, 1.0),
        _evidence("stable_dev", 0.0025, -1.0, split="development"),
        _evidence("stable_dev", 0.005, -1.0, split="development"),
        _evidence("unstable", 0.0025, 1.0),
        _evidence("unstable", 0.005, -1.0),
    ]
    write_csv_atomic(evidence, evidence_rows)

    summary = build_v51_training_view(
        source_manifest_path=manifest,
        source_split_manifest_path=split,
        central_evidence_path=evidence,
        output_dir=output,
    )

    assert summary["new_audio_files"] == 0
    assert summary["stable_anchors_by_split"] == {"train": 1, "development": 1}
    assert summary["retained_symmetric_samples"] == 8
    with (output / "ltsn_manifest_v51.csv").open(encoding="utf-8", newline="") as handle:
        retained = list(csv.DictReader(handle))
    assert not any(row["local_anchor_sample_id"] == "unstable" for row in retained)
    assert all((output / row["latent_path"]).resolve().is_file() for row in retained)
