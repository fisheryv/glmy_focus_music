from __future__ import annotations

import csv
import json
import tomllib
from pathlib import Path

from generation.ltsn_pipeline import write_csv_atomic
from generation.ltsn_v51a import build_v51a_overfit_view, select_v51a_overfit_anchors

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
                "proxy_derivative": "1.0",
                "direction_agrees": str(sign > 0).lower(),
            }
        )
    return rows


def test_v51a_selection_is_step_and_sign_balanced() -> None:
    rows = []
    for step in (4, 5, 6):
        for sign in (-1, 1):
            for index in range(3):
                rows.extend(_evidence(f"s{step}_{sign}_{index}", "train", step, sign))

    selected, summary = select_v51a_overfit_anchors(
        rows,
        step_quotas={4: 2, 5: 2, 6: 2},
    )

    assert len(selected) == 6
    assert summary["selected_by_step_and_sign"] == {
        "4": {"negative": 1, "positive": 1},
        "5": {"negative": 1, "positive": 1},
        "6": {"negative": 1, "positive": 1},
    }


def test_v51a_config_is_unchanged_architecture_and_central_only() -> None:
    with (ROOT / "configs" / "ltsn_training_v51.toml").open("rb") as handle:
        v51 = tomllib.load(handle)
    with (ROOT / "configs" / "ltsn_training_v51a_overfit.toml").open("rb") as handle:
        v51a = tomllib.load(handle)

    assert v51a["model"] == v51["model"]
    assert v51a["training"]["central_direction_overfit_diagnostic"] is True
    assert v51a["training"]["central_direction_classification_only"] is True
    assert v51a["training"]["use_bf16"] is False
    assert v51a["loss"]["central_direction"] == 1.0
    assert all(value == 0 for name, value in v51a["loss"].items() if name != "central_direction")


def test_v51a_view_reuses_selected_train_and_all_development_pairs(tmp_path: Path) -> None:
    source = tmp_path / "v51"
    output = tmp_path / "v51a"
    source.mkdir()
    label = source / "labels.csv"
    label.write_text("sample_id\nbase\n", encoding="utf-8")
    evidence_rows = []
    manifest_rows = []
    anchors: list[tuple[str, str, int, int]] = []
    for step in (4, 5, 6):
        for sign in (-1, 1):
            anchors.append((f"train_s{step}_{sign}", "train", step, sign))
    anchors.extend(
        [
            ("development_negative", "development", 4, -1),
            ("development_positive", "development", 5, 1),
        ]
    )
    for anchor, split, step, sign in anchors:
        evidence_rows.extend(_evidence(anchor, split, step, sign))
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
                    "split": split,
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

    summary = build_v51a_overfit_view(
        source_manifest_path=manifest,
        source_split_manifest_path=split,
        central_evidence_path=evidence,
        output_dir=output,
        step_quotas={4: 2, 5: 2, 6: 2},
    )

    assert summary["train_anchors"] == 6
    assert summary["train_direction_pairs"] == 12
    assert summary["development_anchors"] == 2
    assert summary["rows_by_split"] == {"train": 30, "development": 10}
    assert summary["new_audio_files"] == 0
    with (output / "ltsn_manifest_v51a.csv").open(encoding="utf-8", newline="") as handle:
        retained = list(csv.DictReader(handle))
    assert len(retained) == 40
    assert all((output / row["latent_path"]).resolve().is_file() for row in retained)
