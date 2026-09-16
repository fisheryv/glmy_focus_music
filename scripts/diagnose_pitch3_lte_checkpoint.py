"""Export eval-train/selection diagnostics from an archived direct-energy checkpoint.

The source run is read-only. The output folder must be new or empty.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def diagnose(args):
    import torch

    from generation.ltsn_pipeline import write_json_atomic
    from generation.pitch3_contract import load_pitch3_contract
    from generation.pitch3_lte_training import (
        Pitch3LTETrainingConfig,
        _direct_energy_coordinate_contract,
        _read_examples,
        _to_device,
        load_pitch3_lte_checkpoint,
    )
    from generation.pitch3_lte_v38b import export_predictions, gradient_snapshot, loader
    from generation.pitch3_lte_v38b_protocol import (
        REVISION,
        file_hash,
        local_file,
        read_json,
        verify_file,
    )

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite diagnostics: {args.output_dir}")
    manifest = read_json(args.model_manifest)
    if manifest["architecture_revision"] not in {
        "v3.7_direct_topology_energy",
        "v3.8a_logit_stratified_rank_energy",
        REVISION,
    }:
        raise ValueError("Diagnostics require a V3.7, V3.8-A or V3.8-B direct-energy checkpoint")
    contract = load_pitch3_contract(args.fingerprint)
    if contract.artifact_sha256 != manifest["fingerprint_json_sha256"]:
        raise ValueError("Diagnostic fingerprint differs from the trained model")
    verify_file(args.dataset_manifest, manifest["training_manifest_sha256"])
    folder = args.model_manifest.parent
    protocol_path = folder / "pitch3_lte_run_protocol.json"
    verify_file(protocol_path, manifest["run_protocol_sha256"])
    protocol = read_json(protocol_path)
    config_path = folder / "pitch3_lte_effective_config.json"
    verify_file(config_path, manifest["training_config_sha256"])
    training = Pitch3LTETrainingConfig(**read_json(config_path)["training"])
    records = _read_examples(args.dataset_manifest.resolve(), contract.artifact_sha256)
    by_id = {r.sample_id: r for r in records}
    fit_ids = set(protocol["train_sample_ids"])
    eval_ids = set(protocol["selection_sample_ids"])
    if fit_ids & eval_ids or not fit_ids <= set(by_id) or not eval_ids <= set(by_id):
        raise ValueError("Invalid archived fit/selection IDs")
    if any(by_id[sid].split != "train" for sid in fit_ids):
        raise ValueError("Archived fit IDs include non-train samples")
    fit = [r for r in records if r.sample_id in fit_ids]
    selection = [r for r in records if r.sample_id in eval_ids]
    if manifest.get("validation_scope") == "train_family_cv":
        from generation.pitch3_lte_training import _train_family_cv_split

        expected_fit, expected_eval = _train_family_cv_split(
            [r for r in records if r.split == "train"], manifest["cv_fold"]
        )
        if fit_ids != {r.sample_id for r in expected_fit} or eval_ids != {
            r.sample_id for r in expected_eval
        }:
            raise ValueError("Archived CV IDs differ from the original grouped split")
    device = torch.device(args.device)
    model, metadata = load_pitch3_lte_checkpoint(
        local_file(folder, manifest["checkpoint"]),
        device=device,
        expected_sha256=manifest["checkpoint_sha256"],
    )
    for key in [
        "run_protocol_sha256",
        "training_manifest_sha256",
        "fingerprint_json_sha256",
        "seed",
    ]:
        if metadata[key] != manifest[key]:
            raise ValueError(f"Checkpoint/manifest mismatch: {key}")
    stats = {
        "energy_stratification": manifest["energy_stratification"],
        "local_training_scales": manifest["local_training_scales"],
        "loss_normalizers": {**manifest["loss_normalizers"], "local_shape": 1.0},
        "coordinate_auxiliary": _direct_energy_coordinate_contract(fit, model.config),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for label, selected in [("train", fit), ("selection", selection)]:
        if selected:
            path, diag = export_predictions(
                model, selected, records, device, training, stats, args.output_dir, label
            )
            reports[label] = {"predictions_sha256": file_hash(path), "diagnostics": diag}
    gradients = {}
    weights = manifest["loss_component_weights"]
    for stage in ["global", "local"]:
        active = {k: v for k, v in weights.items() if k.startswith("local_") == (stage == "local")}
        if active:
            sample = _to_device(next(iter(loader(fit, training.seed, stage))), device)
            gradients[stage] = gradient_snapshot(model, sample, stage, training, stats, active)
    report = {
        "source_manifest_sha256": file_hash(args.model_manifest),
        "source_checkpoint_sha256": manifest["checkpoint_sha256"],
        "run_protocol_sha256": manifest["run_protocol_sha256"],
        "dataset_manifest_sha256": file_hash(args.dataset_manifest),
        "fitting_performed": False,
        "model_inference_performed": True,
        "production_authorization": False,
        "reports": reports,
        "gradients": gradients,
    }
    write_json_atomic(args.output_dir / "pitch3_lte_checkpoint_diagnostics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument(
        "--fingerprint", type=Path, default=ROOT / "metadata/focus_pitch3_fingerprint_v1.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = diagnose(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "source_checkpoint_sha256": report["source_checkpoint_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
