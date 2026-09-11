from __future__ import annotations

import csv
import tomllib
from pathlib import Path

from generation.ltsn_contract import sha256_file
from generation.ltsn_pipeline import write_csv_atomic, write_json_atomic
from generation.ltsn_v52b import V52B_VARIANTS, prepare_v52b_probe

ROOT = Path(__file__).resolve().parents[1]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _v52a_source(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "v52a"
    source.mkdir()
    master_rows = []
    evidence_rows = []
    group_index = 0
    specifications = []
    for step in (4, 5, 6):
        specifications.extend(
            [
                (f"train_{step}_a", "train_direction", step, 1, 1),
                (f"train_{step}_b", "train_direction", step, 1, 0),
                (f"heldout_{step}", "heldout_direction", step, 1, 1),
                (f"unseen_{step}", "unseen_anchor", step, 0, 0),
            ]
        )
    for anchor, partition, step, small_target, large_target in specifications:
        for rms, target in ((0.0025, small_target), (0.005, large_target)):
            group = f"{anchor}_r{rms}"
            minus_id = f"{group}_minus"
            plus_id = f"{group}_plus"
            minus_loss = 1.0 if target else 0.0
            plus_loss = 0.0 if target else 1.0
            evidence_rows.append(
                {
                    "direction_group_id": group,
                    "anchor_sample_id": anchor,
                    "anchor_partition": (
                        "unseen_anchor" if partition == "unseen_anchor" else "train_anchor"
                    ),
                    "direction_partition": partition,
                    "direction_index": group_index,
                    "direction_seed": group_index + 100,
                    "step_number": step,
                    "rms_ratio": rms,
                    "minus_sample_id": minus_id,
                    "plus_sample_id": plus_id,
                    "exact_loss_minus": minus_loss,
                    "exact_loss_plus": plus_loss,
                    "exact_derivative": (minus_loss - plus_loss) / (2.0 * rms),
                    "informative": True,
                    "pair_in_distribution": True,
                }
            )
            for sample_id, sign in ((minus_id, -1), (plus_id, 1)):
                latent = source / f"{sample_id}.npy"
                latent.write_bytes(f"latent-{sample_id}".encode())
                master_rows.append(
                    {
                        "sample_id": sample_id,
                        "prompt_id": f"prompt_{anchor}",
                        "trajectory_id": sample_id,
                        "step_number": step,
                        "timestep": 0.5,
                        "latent_path": latent.name,
                        "latent_sha256": sha256_file(latent),
                        "local_direction_sign": sign,
                    }
                )
        group_index += 1
    master = source / "master.csv"
    evidence = source / "evidence.csv"
    views = source / "views.json"
    report = source / "report.json"
    write_csv_atomic(master, master_rows)
    write_csv_atomic(evidence, evidence_rows)
    write_json_atomic(
        views,
        {
            "schema_version": 1,
            "experiment": "ltsn_v5_2a_multi_direction_identifiability",
            "diagnostic_only": True,
            "qualification_eligible": False,
            "guidance_promotion_eligible": False,
            "master_manifest_sha256": sha256_file(master),
            "central_direction_evidence_sha256": sha256_file(evidence),
        },
    )
    write_json_atomic(
        report,
        {
            "schema_version": 1,
            "experiment": "ltsn_v5_2a_multi_direction_identifiability",
            "status": "multi_direction_identifiability_not_supported",
            "identifiability_supported": False,
            "views_summary_sha256": sha256_file(views),
            "true_pairs": [{"metrics": {"train": {"successes": 9}}} for _ in range(3)],
        },
    )
    return master, evidence, views, report


def test_v52b_freezes_scale_ceiling_and_pair_preserving_control(tmp_path: Path) -> None:
    master, evidence, views, report = _v52a_source(tmp_path)
    output = tmp_path / "v52b"

    payload = prepare_v52b_probe(
        master_manifest_path=master,
        evidence_path=evidence,
        views_summary_path=views,
        v52a_report_path=report,
        output_dir=output,
    )

    train_audit = payload["scale_consistency_audit"]["partitions"]["train_direction"]
    assert train_audit["direction_groups"] == 6
    assert train_audit["stable_sign_groups"] == 3
    assert train_audit["rms_invariant_success_ceiling"] == 9
    assert train_audit["rms_invariant_agreement_ceiling"] == 0.75
    assert payload["v52a_ceiling_check"]["observed_train_agreement_equals_invariant_ceiling"]
    assert payload["cross_rms_direction_field_supported"] is False
    assert payload["qualification_eligible"] is False
    assert payload["guidance_promotion_eligible"] is False
    assert payload["new_audio_files"] == 0

    pairs = _read_rows(output / "v52b_pair_manifest.csv")
    train = [row for row in pairs if row["evaluation_split"] == "train"]
    assert len(train) == 6
    assert all(row["rms_ratio"] == "0.005" for row in train)
    assert all(row["minus_sample_id"].endswith("_minus") for row in train)
    assert all(row["plus_sample_id"].endswith("_plus") for row in train)
    assert all(row["true_target"] != row["matched_control_target"] for row in train)
    assert payload["matched_label_control"]["endpoint_pairs_preserved"] is True
    assert payload["matched_label_control"]["changed_fraction"] == 1.0


def test_v52b_config_and_variants_are_frozen_diagnostic() -> None:
    with (ROOT / "configs" / "ltsn_training_v52b_probe.toml").open("rb") as handle:
        config = tomllib.load(handle)
    assert V52B_VARIANTS == ("scalar", "pair", "direction_field")
    assert config["model"]["dropout"] == 0.0
    assert config["model"]["stem_dropout"] == 0.0
    assert config["training"]["weight_decay"] == 0.0
    assert config["training"]["seeds"] == [20260716, 20260717, 20260718]
    assert config["training"]["minimum_epochs"] == config["training"]["max_epochs"]
