"""TAC V5.2e fresh-anchor local-PCA confirmation experiment."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import write_csv_atomic, write_json_atomic
from .tac_v52c import wilson_interval
from .tac_v52d import (
    V52D_CFG_STATUS,
    V52D_EXACT_MARGIN,
    OnManifoldActionHook,
    _completed_batch,
    _context,
    _exact_batch,
    _generate_record,
    _mcnemar,
    _read_csv,
    _score_rows,
)

V52E_EXPERIMENT = "tac_v5_2e_fresh_local_pca_confirmation"
V52E_SCALES = (0.125, 0.25, 0.5)
V52E_REFERENCE_SCALE = 0.25
V52E_ON_MANIFOLD_BASES = ("history_pca_1", "history_pca_2")
V52E_RANDOM_BASES = ("random_control_1", "random_control_2")
V52E_ALL_BASES = (*V52E_ON_MANIFOLD_BASES, *V52E_RANDOM_BASES)
V52E_EXPECTED_STEP_COUNTS = {4: 3, 5: 2, 6: 3}
V52E_ANCHOR_COUNT = 8


def _load_json(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LTSNContractError(f"{label} must be a JSON object")
    return payload


def load_fresh_anchor_specs(
    *,
    source_manifest_path: Path,
    v52c_plan_path: Path,
    v52d_repeatability_plan_path: Path,
    prompt_manifest_path: Path,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Select the frozen V5.2c complement of the eight V5.2d anchors."""

    source = {row["sample_id"]: row for row in _read_csv(source_manifest_path)}
    v52c_plan = _load_json(v52c_plan_path, "V5.2c plan")
    candidates = [str(value) for value in v52c_plan.get("anchor_ids", [])]
    if len(candidates) != 16 or len(set(candidates)) != 16:
        raise LTSNContractError("V5.2e requires the completed 16-anchor V5.2c plan")
    if any(anchor_id not in source for anchor_id in candidates):
        raise LTSNContractError("V5.2e V5.2c anchor is missing from the source manifest")

    repeat_plan = _load_json(v52d_repeatability_plan_path, "V5.2d repeatability plan")
    used = sorted(
        {
            str(row["anchor_sample_id"])
            for row in repeat_plan.get("planned", [])
            if isinstance(row, Mapping) and row.get("anchor_sample_id")
        }
    )
    if len(used) != V52E_ANCHOR_COUNT or not set(used).issubset(candidates):
        raise LTSNContractError("V5.2e requires exactly eight V5.2d anchors to exclude")
    fresh = [anchor_id for anchor_id in candidates if anchor_id not in set(used)]
    if len(fresh) != V52E_ANCHOR_COUNT or set(fresh) & set(used):
        raise LTSNContractError("V5.2e fresh-anchor complement is invalid")

    step_counts: dict[int, int] = {}
    for anchor_id in fresh:
        step = int(source[anchor_id]["step_number"])
        step_counts[step] = step_counts.get(step, 0) + 1
    if step_counts != V52E_EXPECTED_STEP_COUNTS:
        raise LTSNContractError(f"V5.2e fresh-anchor step distribution changed: {step_counts}")

    prompts = {row["prompt_id"]: row for row in _read_csv(prompt_manifest_path)}
    specs = []
    for anchor_id in fresh:
        row = source[anchor_id]
        prompt = prompts.get(row["prompt_id"])
        if prompt is None:
            raise LTSNContractError(f"V5.2e prompt is missing: {row['prompt_id']}")
        trajectory_id = row["trajectory_id"]
        if "__seed" not in trajectory_id:
            raise LTSNContractError(f"V5.2e trajectory has no frozen seed: {trajectory_id}")
        specs.append(
            {
                "anchor_sample_id": anchor_id,
                "prompt_id": row["prompt_id"],
                "caption": prompt["caption"],
                "seed": int(trajectory_id.rsplit("__seed", 1)[1]),
                "bpm": int(prompt["bpm"]) if prompt.get("bpm", "").strip() else None,
                "keyscale": prompt.get("keyscale", ""),
                "timesignature": prompt.get("timesignature", ""),
                "step_number": int(row["step_number"]),
                "model_family": row["model_family"],
                "split": "development",
            }
        )
    return sorted(specs, key=lambda row: row["anchor_sample_id"]), fresh, used


def _validate_prior_artifacts(
    *,
    source_manifest_path: Path,
    v52c_plan_path: Path,
    v52d_repeatability_plan_path: Path,
    v52d_repeatability_report_path: Path,
    v52d_action_report_path: Path,
    prompt_manifest_path: Path,
    ace_config_path: Path,
    fingerprint_path: Path,
    target_path: Path,
    ace_model_sha256: str,
    vae_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    repeat_plan = _load_json(v52d_repeatability_plan_path, "V5.2d repeatability plan")
    repeat_report = _load_json(v52d_repeatability_report_path, "V5.2d repeatability report")
    action_report = _load_json(v52d_action_report_path, "V5.2d action report")
    if (
        repeat_report.get("repeatability_status") != "passed"
        or repeat_report.get("action_collection_authorized") is not True
    ):
        raise LTSNContractError("V5.2e requires passed V5.2d repeatability")
    if repeat_report.get("plan_sha256") != sha256_file(v52d_repeatability_plan_path):
        raise LTSNContractError("V5.2e V5.2d repeatability plan binding changed")
    expected_plan = {
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "v52c_plan_sha256": sha256_file(v52c_plan_path),
        "prompt_manifest_sha256": sha256_file(prompt_manifest_path),
        "ace_config_sha256": sha256_file(ace_config_path),
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "target_sha256": sha256_file(target_path),
        "ace_model_sha256": ace_model_sha256,
        "vae_sha256": vae_sha256,
    }
    for name, expected in expected_plan.items():
        if repeat_plan.get(name) != expected:
            raise LTSNContractError(f"V5.2e V5.2d repeatability plan mismatch: {name}")
    expected_report = {
        "ace_model_sha256": ace_model_sha256,
        "vae_sha256": vae_sha256,
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "target_sha256": sha256_file(target_path),
        "implementation_sha256": repeat_plan.get("implementation_sha256"),
    }
    for name, expected in expected_report.items():
        if repeat_report.get(name) != expected:
            raise LTSNContractError(f"V5.2e V5.2d repeatability report mismatch: {name}")
    if (
        action_report.get("stage1d_status") != "not_supported"
        or action_report.get("next_stage_authorized") is not False
    ):
        raise LTSNContractError("V5.2e requires the frozen unsupported V5.2d action result")
    if action_report.get("repeatability_report_sha256") != sha256_file(
        v52d_repeatability_report_path
    ):
        raise LTSNContractError("V5.2e V5.2d action/repeatability binding changed")
    return repeat_plan, repeat_report, action_report


def _implementation_hashes(root: Path, config: Any) -> dict[str, str]:
    paths = {
        "src/generation/tac_v52e.py": Path(__file__).resolve(),
        "src/generation/tac_v52d.py": root / "src" / "generation" / "tac_v52d.py",
        "src/generation/ace_adapter.py": root / "src" / "generation" / "ace_adapter.py",
        "ace_xl_turbo_sampler": root
        / config.ace.checkout
        / "acestep"
        / "models"
        / "xl_turbo"
        / "modeling_acestep_v15_xl_turbo.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _agreement(values: Sequence[bool]) -> dict[str, Any]:
    if not values:
        return {"successes": 0, "comparisons": 0, "agreement": None}
    return {
        "successes": int(sum(values)),
        "comparisons": len(values),
        "agreement": float(sum(values) / len(values)),
    }


def analyze_actions(
    rows: Sequence[Mapping[str, Any]], *, repeatability_report: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate only the pre-registered fresh-anchor V5.2e confirmation gates."""

    if (
        repeatability_report.get("repeatability_status") != "passed"
        or repeatability_report.get("action_collection_authorized") is not True
    ):
        raise LTSNContractError("V5.2e analysis requires passed V5.2d repeatability")
    grouped: dict[tuple[str, str, float], dict[float, Mapping[str, Any]]] = {}
    anchor_steps: dict[str, int] = {}
    for row in rows:
        anchor_id = str(row["anchor_sample_id"])
        basis = str(row["basis_name"])
        scale = float(row["action_scale"])
        sign = float(row["sign"])
        if basis not in V52E_ALL_BASES or scale not in V52E_SCALES or sign not in {-1.0, 1.0}:
            raise LTSNContractError("V5.2e action table contains an unregistered action")
        step = int(row["step_number"])
        if anchor_id in anchor_steps and anchor_steps[anchor_id] != step:
            raise LTSNContractError(f"V5.2e anchor step changed: {anchor_id}")
        anchor_steps[anchor_id] = step
        signed = grouped.setdefault((anchor_id, basis, scale), {})
        if sign in signed:
            raise LTSNContractError(f"duplicate V5.2e action sign: {(anchor_id, basis, scale)}")
        signed[sign] = row
    anchors = sorted(anchor_steps)
    if len(anchors) != V52E_ANCHOR_COUNT:
        raise LTSNContractError("V5.2e requires exactly eight fresh anchors")
    observed_step_counts = {
        step: list(anchor_steps.values()).count(step) for step in sorted(set(anchor_steps.values()))
    }
    if observed_step_counts != V52E_EXPECTED_STEP_COUNTS:
        raise LTSNContractError("V5.2e action table has the wrong step distribution")
    expected_keys = {
        (anchor_id, basis, scale)
        for anchor_id in anchors
        for basis in V52E_ALL_BASES
        for scale in V52E_SCALES
    }
    if set(grouped) != expected_keys:
        raise LTSNContractError("V5.2e action table is incomplete or contains extra pairs")

    outcomes = []
    for (anchor_id, basis, scale), signed in sorted(grouped.items()):
        if set(signed) != {-1.0, 1.0}:
            raise LTSNContractError(f"incomplete V5.2e action pair: {(anchor_id, basis, scale)}")
        minus, plus = signed[-1.0], signed[1.0]
        difference = float(minus["target_distance"]) - float(plus["target_distance"])
        outcomes.append(
            {
                "anchor_sample_id": anchor_id,
                "step_number": anchor_steps[anchor_id],
                "basis_name": basis,
                "basis_role": "on_manifold"
                if basis in V52E_ON_MANIFOLD_BASES
                else "matched_random_control",
                "action_scale": scale,
                "target_response": difference / (2.0 * scale),
                "response_sign": int(np.sign(difference))
                if abs(difference) >= V52D_EXACT_MARGIN
                else 0,
                "absolute_half_pair_effect": abs(difference) / 2.0,
                "pair_in_distribution": not bool(float(minus["ood_label"]))
                and not bool(float(plus["ood_label"])),
            }
        )
    outcome_by_key = {
        (row["anchor_sample_id"], row["basis_name"], row["action_scale"]): row for row in outcomes
    }
    stability: dict[tuple[str, str], list[bool]] = {}
    for anchor_id in anchors:
        for basis in V52E_ALL_BASES:
            reference = outcome_by_key[(anchor_id, basis, V52E_REFERENCE_SCALE)]["response_sign"]
            stability[(anchor_id, basis)] = [
                reference != 0
                and outcome_by_key[(anchor_id, basis, scale)]["response_sign"] == reference
                for scale in V52E_SCALES
                if scale != V52E_REFERENCE_SCALE
            ]
    on_values = [
        value
        for (anchor_id, basis), values in stability.items()
        if basis in V52E_ON_MANIFOLD_BASES
        for value in values
    ]
    random_values = [
        value
        for (anchor_id, basis), values in stability.items()
        if basis in V52E_RANDOM_BASES
        for value in values
    ]
    successes = int(sum(on_values))
    interval = wilson_interval(successes, len(on_values))
    agreement = successes / len(on_values)
    paired_on = []
    paired_random = []
    for anchor_id in anchors:
        for on_basis, random_basis in zip(V52E_ON_MANIFOLD_BASES, V52E_RANDOM_BASES, strict=True):
            paired_on.extend(stability[(anchor_id, on_basis)])
            paired_random.extend(stability[(anchor_id, random_basis)])
    comparison = _mcnemar(paired_on, paired_random)
    effects = [
        float(row["absolute_half_pair_effect"])
        for row in outcomes
        if row["basis_role"] == "on_manifold"
    ]
    noise = float(repeatability_report["target_distance_noise_q95"])
    median_effect = float(np.median(effects))
    all_id = all(bool(row["pair_in_distribution"]) for row in outcomes)
    gates = {
        "on_manifold_cross_scale_agreement_at_least_0_80": agreement >= 0.80,
        "on_manifold_wilson_lower_above_0_50": interval[0] > 0.50,
        "on_manifold_better_than_random_mcnemar_p_below_0_05": (
            comparison["on_manifold_only_stable"] > comparison["random_only_stable"]
            and comparison["two_sided_exact_p"] < 0.05
        ),
        "median_effect_above_three_times_repeatability_noise": median_effect > 3.0 * noise,
        "all_pairs_in_distribution": all_id,
    }
    passed = all(gates.values())

    by_basis = []
    for basis in V52E_ALL_BASES:
        members = [row for row in outcomes if row["basis_name"] == basis]
        by_basis.append(
            {
                "basis_name": basis,
                "basis_role": members[0]["basis_role"],
                "pairs": len(members),
                "cross_scale": _agreement(
                    [value for anchor_id in anchors for value in stability[(anchor_id, basis)]]
                ),
                "median_absolute_half_pair_effect": float(
                    np.median([row["absolute_half_pair_effect"] for row in members])
                ),
                "all_pairs_in_distribution": all(row["pair_in_distribution"] for row in members),
            }
        )
    by_step = []
    for step in sorted(V52E_EXPECTED_STEP_COUNTS):
        step_anchors = [anchor_id for anchor_id in anchors if anchor_steps[anchor_id] == step]
        by_step.append(
            {
                "step_number": step,
                "anchors": len(step_anchors),
                "on_manifold_cross_scale": _agreement(
                    [
                        value
                        for anchor_id in step_anchors
                        for basis in V52E_ON_MANIFOLD_BASES
                        for value in stability[(anchor_id, basis)]
                    ]
                ),
                "random_cross_scale": _agreement(
                    [
                        value
                        for anchor_id in step_anchors
                        for basis in V52E_RANDOM_BASES
                        for value in stability[(anchor_id, basis)]
                    ]
                ),
            }
        )
    by_anchor = []
    for anchor_id in anchors:
        by_anchor.append(
            {
                "anchor_sample_id": anchor_id,
                "step_number": anchor_steps[anchor_id],
                "on_manifold_cross_scale": _agreement(
                    [
                        value
                        for basis in V52E_ON_MANIFOLD_BASES
                        for value in stability[(anchor_id, basis)]
                    ]
                ),
                "random_cross_scale": _agreement(
                    [
                        value
                        for basis in V52E_RANDOM_BASES
                        for value in stability[(anchor_id, basis)]
                    ]
                ),
            }
        )
    report = {
        "schema_version": 1,
        "experiment": V52E_EXPERIMENT,
        "diagnostic_only": True,
        "scientific_evidence": False,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
        "fresh_confirmation": True,
        "prior_action_labels_reused": False,
        "cfg_residual": V52D_CFG_STATUS,
        "action_bases": {
            "on_manifold": list(V52E_ON_MANIFOLD_BASES),
            "matched_random_control": list(V52E_RANDOM_BASES),
        },
        "scales": list(V52E_SCALES),
        "reference_scale": V52E_REFERENCE_SCALE,
        "exact_margin": V52D_EXACT_MARGIN,
        "anchors": len(anchors),
        "action_rollouts": len(rows),
        "pairs": len(outcomes),
        "on_manifold_cross_scale": {
            "successes": successes,
            "comparisons": len(on_values),
            "agreement": agreement,
            "wilson95": interval,
        },
        "random_cross_scale": _agreement(random_values),
        "matched_control_mcnemar": comparison,
        "repeatability_noise_q95": noise,
        "median_on_manifold_absolute_half_pair_effect": median_effect,
        "effect_to_noise_ratio": median_effect / max(noise, np.finfo(float).eps),
        "descriptive_strata_not_alternative_gates": {
            "by_basis": by_basis,
            "by_step": by_step,
            "by_anchor": by_anchor,
        },
        "gates": gates,
        "stage1e_status": "supported_for_critic_collection" if passed else "not_supported",
        "next_stage_authorized": passed,
    }
    return report, outcomes


def collect_v52e(
    *,
    root: Path,
    source_manifest_path: Path,
    v52c_plan_path: Path,
    v52d_repeatability_plan_path: Path,
    v52d_repeatability_report_path: Path,
    v52d_action_report_path: Path,
    prompt_manifest_path: Path,
    ace_config_path: Path,
    fingerprint_path: Path,
    target_path: Path,
    output_dir: Path,
    ace_model_sha256: str,
    vae_sha256: str,
    device_name: str = "cuda:0",
    workers: int = 8,
    batch_size: int = 12,
    materialize_mode: str = "auto",
) -> dict[str, Any]:
    """Run the pre-registered 192-rollout fresh-anchor confirmation."""

    root = root.resolve()
    raw_paths = [
        source_manifest_path,
        v52c_plan_path,
        v52d_repeatability_plan_path,
        v52d_repeatability_report_path,
        v52d_action_report_path,
        prompt_manifest_path,
        ace_config_path,
        fingerprint_path,
        target_path,
        output_dir,
    ]
    resolved = [
        path.resolve() if path.is_absolute() else (root / path).resolve() for path in raw_paths
    ]
    (
        source_manifest_path,
        v52c_plan_path,
        v52d_repeatability_plan_path,
        v52d_repeatability_report_path,
        v52d_action_report_path,
        prompt_manifest_path,
        ace_config_path,
        fingerprint_path,
        target_path,
        output_dir,
    ) = resolved
    repeat_plan, repeatability, action_report = _validate_prior_artifacts(
        source_manifest_path=source_manifest_path,
        v52c_plan_path=v52c_plan_path,
        v52d_repeatability_plan_path=v52d_repeatability_plan_path,
        v52d_repeatability_report_path=v52d_repeatability_report_path,
        v52d_action_report_path=v52d_action_report_path,
        prompt_manifest_path=prompt_manifest_path,
        ace_config_path=ace_config_path,
        fingerprint_path=fingerprint_path,
        target_path=target_path,
        ace_model_sha256=ace_model_sha256,
        vae_sha256=vae_sha256,
    )
    source_rows = _read_csv(source_manifest_path)
    if {row["ace_model_sha256"] for row in source_rows} != {ace_model_sha256}:
        raise LTSNContractError("V5.2e ACE hash differs from source manifest")
    if {row["vae_sha256"] for row in source_rows} != {vae_sha256}:
        raise LTSNContractError("V5.2e VAE hash differs from source manifest")
    config, scorer, target = _context(
        root=root,
        ace_config_path=ace_config_path,
        fingerprint_path=fingerprint_path,
        target_path=target_path,
    )
    specs, fresh_anchor_ids, excluded_anchor_ids = load_fresh_anchor_specs(
        source_manifest_path=source_manifest_path,
        v52c_plan_path=v52c_plan_path,
        v52d_repeatability_plan_path=v52d_repeatability_plan_path,
        prompt_manifest_path=prompt_manifest_path,
    )
    planned = []
    for spec in specs:
        for basis in V52E_ALL_BASES:
            for scale in V52E_SCALES:
                for sign, sign_name in ((-1.0, "minus"), (1.0, "plus")):
                    planned.append(
                        {
                            **spec,
                            "basis_name": basis,
                            "action_scale": scale,
                            "sign": sign,
                            "sample_id": (
                                f"{spec['anchor_sample_id']}__v52e_{basis}_"
                                f"s{int(scale * 1000):04d}_{sign_name}"
                            ),
                        }
                    )
    planned.sort(key=lambda row: row["sample_id"])
    if len(planned) != 192:
        raise LTSNContractError("V5.2e frozen rollout count changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": 1,
        "experiment": V52E_EXPERIMENT,
        "diagnostic_only": True,
        "scientific_evidence": False,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
        "fresh_confirmation": True,
        "prior_action_labels_reused": False,
        "repeatability_artifact_reused": True,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "v52c_plan_sha256": sha256_file(v52c_plan_path),
        "v52d_repeatability_plan_sha256": sha256_file(v52d_repeatability_plan_path),
        "v52d_repeatability_report_sha256": sha256_file(v52d_repeatability_report_path),
        "v52d_action_report_sha256": sha256_file(v52d_action_report_path),
        "prompt_manifest_sha256": sha256_file(prompt_manifest_path),
        "ace_config_sha256": sha256_file(ace_config_path),
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "target_sha256": sha256_file(target_path),
        "ace_model_sha256": ace_model_sha256,
        "vae_sha256": vae_sha256,
        "v52d_historical_implementation_sha256": repeat_plan["implementation_sha256"],
        "implementation_sha256": _implementation_hashes(root, config),
        "v52d_stage1d_status": action_report["stage1d_status"],
        "anchor_count": V52E_ANCHOR_COUNT,
        "fresh_anchor_ids": fresh_anchor_ids,
        "excluded_v52d_anchor_ids": excluded_anchor_ids,
        "fresh_step_counts": {str(key): value for key, value in V52E_EXPECTED_STEP_COUNTS.items()},
        "on_manifold_bases": list(V52E_ON_MANIFOLD_BASES),
        "matched_random_bases": list(V52E_RANDOM_BASES),
        "scales": list(V52E_SCALES),
        "reference_scale": V52E_REFERENCE_SCALE,
        "cfg_residual": V52D_CFG_STATUS,
        "planned_samples": len(planned),
        "wav_policy": "ephemeral_delete_after_each_exact_batch",
        "planned": planned,
    }
    plan_path = output_dir / "v52e_action_plan.json"
    if plan_path.is_file() and _load_json(plan_path, "V5.2e action plan") != plan:
        raise LTSNContractError("V5.2e action plan changed; use a new output directory")
    if not plan_path.exists():
        write_json_atomic(plan_path, plan)
    plan_sha256 = sha256_file(plan_path)

    from .ace_adapter import AceStepAdapter

    os.environ["ACESTEP_DEVICE"] = device_name
    adapter = AceStepAdapter(root / config.ace.checkout, config.ace)
    all_descriptors: list[dict[str, str]] = []
    all_records: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(planned), batch_size)):
        items = planned[start : start + batch_size]
        batch_dir = output_dir / "batches" / f"batch_{batch_index:05d}"
        logical_ids = [str(row["sample_id"]) for row in items]
        completed = _completed_batch(batch_dir, plan_sha256=plan_sha256, logical_ids=logical_ids)
        if completed is not None:
            descriptors, records = completed
        else:
            records = []
            for item in items:
                hook = OnManifoldActionHook(
                    anchor_id=item["anchor_sample_id"],
                    target_step=int(item["step_number"]),
                    basis_name=item["basis_name"],
                    scale=float(item["action_scale"]),
                    sign=float(item["sign"]),
                    allowed_scales=V52E_SCALES,
                )
                records.append(
                    _generate_record(
                        adapter=adapter,
                        config=config,
                        spec=item,
                        sample_id=item["sample_id"],
                        hook=hook,
                        output_dir=output_dir,
                        ace_model_sha256=ace_model_sha256,
                        vae_sha256=vae_sha256,
                        extra={
                            "basis_name": item["basis_name"],
                            "basis_role": "on_manifold"
                            if item["basis_name"] in V52E_ON_MANIFOLD_BASES
                            else "matched_random_control",
                            "action_scale": item["action_scale"],
                            "sign": item["sign"],
                        },
                    )
                )
            descriptors, records = _exact_batch(
                root=root,
                output_dir=output_dir,
                batch_index=batch_index,
                plan_sha256=plan_sha256,
                records=records,
                exact_repeats=1,
                workers=workers,
                materialize_mode=materialize_mode,
                experiment=V52E_EXPERIMENT,
            )
        all_descriptors.extend(descriptors)
        all_records.extend(records)
    basis_audits: dict[str, set[str]] = {}
    for record in all_records:
        encoded = json.dumps(
            record["action_audit"]["basis_sha256"], sort_keys=True, separators=(",", ":")
        )
        basis_audits.setdefault(record["anchor_sample_id"], set()).add(encoded)
    if set(basis_audits) != set(fresh_anchor_ids) or any(
        len(values) != 1 for values in basis_audits.values()
    ):
        raise LTSNContractError("V5.2e action basis changed across matched rollouts")
    points = _score_rows(
        descriptors=all_descriptors, records=all_records, scorer=scorer, target=target
    )
    points_path = output_dir / "v52e_action_points.csv"
    write_csv_atomic(points_path, points)
    report, outcomes = analyze_actions(points, repeatability_report=repeatability)
    outcomes_path = output_dir / "v52e_action_outcomes.csv"
    write_csv_atomic(outcomes_path, outcomes)
    report.update(
        {
            "plan_sha256": plan_sha256,
            "points_sha256": sha256_file(points_path),
            "outcomes_sha256": sha256_file(outcomes_path),
            "v52d_repeatability_report_sha256": sha256_file(v52d_repeatability_report_path),
            "v52d_action_report_sha256": sha256_file(v52d_action_report_path),
            "retained_wav_files": len(list(output_dir.rglob("*.wav"))),
            "retained_latent_files": len(list((output_dir / "latents").glob("*.npy"))),
        }
    )
    if report["retained_wav_files"] != 0:
        raise LTSNContractError("V5.2e action cleanup retained WAV")
    report_path = output_dir / "v52e_action_report.json"
    write_json_atomic(report_path, report)
    return report
