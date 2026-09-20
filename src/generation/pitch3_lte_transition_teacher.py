"""Resume-safe frozen Pitch transition targets, reconstructed from the same x0."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from .ltsn_contract import sha256_file
from .ltsn_pipeline import write_json_atomic
from .pitch3_contract import load_pitch3_contract
from .pitch3_lte_v38b_protocol import read_csv, read_json, verify_file
from .pitch3_lte_v39a_protocol import content_hash, grouped_split
from .pitch3_lte_v39b_protocol import coordinates_from_counts, joint_counts, load_teacher


def build_teacher(
    *,
    root,
    dataset_manifest,
    fingerprint_path,
    output_dir,
    ace_config=None,
    trajectory_manifest=None,
    device="cuda:0",
    workers=4,
):
    """No new diffusion samples. Reuse signed audio or decode the original latent.

    Each receipt binds a count matrix to its latent and exact q2/q3 labels.
    Failed receipts never become a completed, trainable teacher archive.
    """
    from features.batch import _read_npz
    from features.pitch_v2 import assign_codebook, chroma_to_tonnetz

    from .exact_features import extract_candidate_features, preprocess_candidates
    from .experiment import CandidateRecord
    from .ltsn_storage import remove_tree_within

    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    dataset_manifest = Path(dataset_manifest).resolve()
    fingerprint_path = Path(fingerprint_path).resolve()
    contract = load_pitch3_contract(fingerprint_path)
    data = read_csv(dataset_manifest)
    grouped_split(data, 0)  # Original disjoint train/development contract.
    selected = sorted(
        (r for r in data if r["split"] == "train" and r["source_kind"] == "base_step4_seed"),
        key=lambda r: r["sample_id"],
    )
    dataset_plan = dataset_manifest.parent / "pitch3_lte_dataset_plan.json"
    original_plan = read_json(dataset_plan)
    original_summary = read_json(dataset_manifest.parent / "pitch3_lte_dataset_summary.json")
    verify_file(dataset_manifest, original_summary["dataset_manifest_sha256"])
    verify_file(dataset_plan, original_summary["dataset_plan_sha256"])
    if original_summary.get("local_preflight_passed") is not True:
        raise ValueError("Original dataset preflight has not passed")
    for key in ("fingerprint_json_sha256", "ace_model_sha256", "vae_sha256"):
        if {r[key] for r in selected} != {original_plan[key]}:
            raise ValueError(f"Original dataset {key} differs from its plan")
    if any(r["fingerprint_json_sha256"] != contract.artifact_sha256 for r in selected):
        raise ValueError("Teacher fingerprint differs from original labels")
    if workers < 1:
        raise ValueError("Positive worker count required")
    if ace_config is not None:
        ace_config = Path(ace_config).resolve()
        verify_file(ace_config, original_plan["ace_config_sha256"])
    codebook = root / "features/models/pitch_v2_codebook.npz"
    with np.load(codebook, allow_pickle=False) as z:
        centers = np.asarray(z["centers"], dtype=np.float64)
    if centers.shape != (16, 6) or not np.isfinite(centers).all():
        raise ValueError("Expected original finite 16-state Tonnetz codebook")
    cache = {}
    if trajectory_manifest is not None:
        trajectory_manifest = Path(trajectory_manifest).resolve()
        for r in read_csv(trajectory_manifest):
            if r["sample_id"] in cache:
                raise ValueError("Duplicate trajectory cache sample")
            cache[r["sample_id"]] = r
    plan = {
        "stage": "pitch3_lte_transition_teacher_v39b",
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "dataset_plan_sha256": sha256_file(dataset_plan),
        "fingerprint_json_sha256": contract.artifact_sha256,
        "codebook_sha256": sha256_file(codebook),
        "ace_config_sha256": sha256_file(ace_config) if ace_config else None,
        "trajectory_manifest_sha256": sha256_file(trajectory_manifest)
        if trajectory_manifest
        else None,
        "source_hashes": {
            str(p.relative_to(root)): sha256_file(p)
            for p in (
                root / "configs/pipeline.toml",
                root / "src/features/pitch_v2.py",
                root / "src/features/batch.py",
                root / "src/data/preprocess.py",
                root / "src/generation/exact_features.py",
                root / "src/generation/pitch3_lte_transition_teacher.py",
                root / "src/generation/pitch3_lte_v39b_protocol.py",
                root / "src/generation/ace_adapter.py",
            )
        },
        "sample_ids": [r["sample_id"] for r in selected],
        "target_semantics": "single_frozen_window_global_joint_counts_including_self",
        "duration_seconds": 180.0,
        "reconstruction_atol": 1e-7,
        "development_used": False,
    }
    plan_path = output_dir / "teacher_plan.json"
    completion = output_dir / "teacher_manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not plan_path.exists():
        raise FileExistsError("Nonempty teacher directory has no bound resume plan")
    output_dir.mkdir(parents=True, exist_ok=True)
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Teacher plan changed; use a new output directory")
    if not plan_path.exists():
        write_json_atomic(plan_path, plan)
    if completion.exists():
        load_teacher(completion, dataset_manifest, fingerprint_path)
        return read_json(completion)
    adapter = None
    receipts, matrices = [], []
    for index, row in enumerate(selected):
        sid = row["sample_id"]
        # IDs only name hashed files, never untrusted filesystem components.
        token = content_hash(sid)
        receipt_path = output_dir / "receipts" / f"{token}.json"
        latent_path = (dataset_manifest.parent / row["latent_path"]).resolve()
        verify_file(latent_path, row["latent_sha256"])
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if (
                receipt["plan_sha256"] != content_hash(plan)
                or receipt["sample_id"] != sid
                or receipt["latent_sha256"] != row["latent_sha256"]
            ):
                raise ValueError("Teacher resume receipt detached")
            c = np.asarray(receipt["counts"])
        else:
            work = output_dir / "scratch" / token
            if work.exists():
                remove_tree_within(work, output_dir / "scratch")
            audio_path = work / "data_raw/snapshot.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            cached = cache.get(sid)
            source = (
                (trajectory_manifest.parent / cached["audio_path"]).resolve()
                if (cached and cached.get("audio_path"))
                else None
            )
            if source is not None and source.is_file():
                if (
                    cached["latent_sha256"] != row["latent_sha256"]
                    or cached["vae_sha256"] != row["vae_sha256"]
                ):
                    raise ValueError("Cached audio belongs to another latent/VAE")
                verify_file(source, cached["audio_sha256"])
                shutil.copyfile(source, audio_path)
                audio_source = "signed_original_trajectory_audio"
            else:
                if ace_config is None:
                    raise FileNotFoundError(
                        f"No signed cached audio for {sid}; "
                        "provide original --ace-config for x0 re-decoding"
                    )
                if adapter is None:
                    from .ace_adapter import AceStepAdapter
                    from .experiment import load_experiment_config

                    cfg = load_experiment_config(root, ace_config)
                    os.environ["ACESTEP_DEVICE"] = device
                    adapter = AceStepAdapter(root / cfg.ace.checkout, cfg.ace)
                adapter.decode_latent_to_audio(np.load(latent_path, allow_pickle=False), audio_path)
                audio_source = "original_x0_redecoded_coordinate_checked"
            audio_hash = sha256_file(audio_path)
            record = CandidateRecord(
                experiment_id="pitch3_lte_v39b_teacher",
                prompt_id=row["prompt_id"],
                caption="Frozen x0 teacher",
                candidate_index=0,
                candidate_id=token,
                seed=0,
                duration_seconds=180.0,
                status="generated",
                audio_relative_path="data_raw/snapshot.wav",
                audio_sha256=audio_hash,
            )
            processed = preprocess_candidates(root, work, [record], workers=workers)
            features = extract_candidate_features(root, work, processed, workers=workers)
            if len(features) != 1 or len(processed) != 1:
                raise ValueError("Expected the original single analysis window")
            chroma_path = work / features[0]["chroma_relative_path"]
            arrays = _read_npz(chroma_path)
            chroma = np.asarray(arrays["chroma"], dtype=np.float64)
            tonnetz = chroma_to_tonnetz(chroma)
            valid = np.asarray(arrays["valid"], dtype=bool)
            valid &= np.isfinite(tonnetz).all(axis=1) & (chroma.sum(axis=1) > 1e-8)
            states = assign_codebook(tonnetz, centers, valid=valid)
            c = joint_counts(states)
            _, q = coordinates_from_counts(c, contract)
            if not np.allclose(q, json.loads(row["coordinates_json"])[1:], rtol=0, atol=1e-7):
                raise ValueError(
                    f"Re-decoded teacher mismatch for {sid}; scratch retained at {work}"
                )
            receipt = {
                "sample_id": sid,
                "plan_sha256": content_hash(plan),
                "latent_sha256": row["latent_sha256"],
                "audio_sha256": audio_hash,
                "audio_source": audio_source,
                "chroma_sha256": sha256_file(chroma_path),
                "preprocessed_audio_sha256": processed[0]["sha256"],
                "q2_q3": q.tolist(),
                "counts": c.tolist(),
            }
            write_json_atomic(receipt_path, receipt)
            remove_tree_within(work, output_dir / "scratch")
        _, q = coordinates_from_counts(c, contract)
        if not np.allclose(q, json.loads(row["coordinates_json"])[1:], rtol=0, atol=1e-7):
            raise ValueError(f"Cached teacher reconstruction mismatch: {sid}")
        matrices.append(c)
        receipts.append({k: v for k, v in receipt.items() if k != "counts"})
        print(
            json.dumps({"teacher_samples": index + 1, "total": len(selected), "sample_id": sid}),
            flush=True,
        )
    target = output_dir / "transition_targets.npz"
    with target.with_suffix(".npz.part").open("wb") as f:
        np.savez_compressed(f, sample_ids=np.array(plan["sample_ids"]), counts=np.stack(matrices))
    target.with_suffix(".npz.part").replace(target)
    result = {
        **plan,
        "completed": True,
        "targets_file": target.name,
        "targets_sha256": sha256_file(target),
        "samples": receipts,
        "production_authorization": False,
    }
    write_json_atomic(completion, result)
    load_teacher(completion, dataset_manifest, fingerprint_path)
    return result
