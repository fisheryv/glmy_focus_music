from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from generation.tac_v52d import OnManifoldActionHook
from generation.tac_v52e import (
    V52E_ALL_BASES,
    V52E_ON_MANIFOLD_BASES,
    V52E_RANDOM_BASES,
    V52E_SCALES,
    analyze_actions,
    load_fresh_anchor_specs,
)


def _repeatability_report() -> dict[str, Any]:
    return {
        "repeatability_status": "passed",
        "action_collection_authorized": True,
        "target_distance_noise_q95": 0.01,
    }


def _action_rows(*, tie_reference: bool = False) -> list[dict[str, Any]]:
    steps = [4, 4, 4, 5, 5, 6, 6, 6]
    rows = []
    for anchor_index, step in enumerate(steps):
        for basis in V52E_ALL_BASES:
            for scale in V52E_SCALES:
                stable = basis in V52E_ON_MANIFOLD_BASES
                response_sign = 1.0 if stable or scale == 0.25 else -1.0
                if (
                    tie_reference
                    and anchor_index == 0
                    and basis == "history_pca_1"
                    and scale == 0.25
                ):
                    response_sign = 0.0
                for sign in (-1.0, 1.0):
                    rows.append(
                        {
                            "anchor_sample_id": f"anchor_{anchor_index}",
                            "step_number": step,
                            "basis_name": basis,
                            "action_scale": scale,
                            "sign": sign,
                            "target_distance": 2.0 - sign * scale * response_sign,
                            "ood_label": 0,
                        }
                    )
    return rows


def test_confirmation_uses_frozen_counts_and_matched_random_controls() -> None:
    report, outcomes = analyze_actions(_action_rows(), repeatability_report=_repeatability_report())

    assert len(outcomes) == 8 * 4 * 3
    assert report["action_rollouts"] == 192
    assert report["on_manifold_cross_scale"] == {
        "successes": 32,
        "comparisons": 32,
        "agreement": 1.0,
        "wilson95": report["on_manifold_cross_scale"]["wilson95"],
    }
    assert report["random_cross_scale"]["agreement"] == 0.0
    assert report["matched_control_mcnemar"]["on_manifold_only_stable"] == 32
    assert report["matched_control_mcnemar"]["random_only_stable"] == 0
    assert report["stage1e_status"] == "supported_for_critic_collection"
    assert report["next_stage_authorized"] is True
    assert report["prior_action_labels_reused"] is False
    assert set(report["action_bases"]["matched_random_control"]) == set(V52E_RANDOM_BASES)


def test_reference_ties_are_counted_as_cross_scale_failures() -> None:
    report, _outcomes = analyze_actions(
        _action_rows(tie_reference=True), repeatability_report=_repeatability_report()
    )

    assert report["on_manifold_cross_scale"]["successes"] == 30
    assert report["on_manifold_cross_scale"]["comparisons"] == 32


def test_v52e_scale_extension_does_not_change_v52d_default_contract() -> None:
    with pytest.raises(ValueError, match="scale or sign changed"):
        OnManifoldActionHook(
            anchor_id="anchor", target_step=4, basis_name="history_pca_1", scale=0.125, sign=1.0
        )

    hook = OnManifoldActionHook(
        anchor_id="anchor",
        target_step=4,
        basis_name="history_pca_1",
        scale=0.125,
        sign=1.0,
        allowed_scales=V52E_SCALES,
    )
    assert hook.scale == 0.125


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_fresh_anchors_are_exact_v52c_complement(tmp_path: Path) -> None:
    candidates = [f"anchor_{index:02d}" for index in range(16)]
    used = candidates[:8]
    fresh_steps = [4, 4, 4, 5, 5, 6, 6, 6]
    source_rows = []
    for index, anchor_id in enumerate(candidates):
        step = 4 if index < 8 else fresh_steps[index - 8]
        source_rows.append(
            {
                "sample_id": anchor_id,
                "prompt_id": f"prompt_{index:02d}",
                "trajectory_id": f"trajectory_{index:02d}__seed{1000 + index}",
                "step_number": step,
                "model_family": "acestep-v15-xl-turbo",
            }
        )
    prompt_rows = [
        {
            "prompt_id": f"prompt_{index:02d}",
            "caption": f"caption {index}",
            "bpm": "76",
            "keyscale": "",
            "timesignature": "4",
        }
        for index in range(16)
    ]
    source_path = tmp_path / "source.csv"
    prompt_path = tmp_path / "prompts.csv"
    v52c_path = tmp_path / "v52c.json"
    repeat_path = tmp_path / "repeat.json"
    _write_csv(source_path, source_rows)
    _write_csv(prompt_path, prompt_rows)
    v52c_path.write_text(json.dumps({"anchor_ids": candidates}), encoding="utf-8")
    repeat_path.write_text(
        json.dumps({"planned": [{"anchor_sample_id": value} for value in used]}),
        encoding="utf-8",
    )

    specs, fresh, excluded = load_fresh_anchor_specs(
        source_manifest_path=source_path,
        v52c_plan_path=v52c_path,
        v52d_repeatability_plan_path=repeat_path,
        prompt_manifest_path=prompt_path,
    )

    assert fresh == candidates[8:]
    assert excluded == used
    assert {row["anchor_sample_id"] for row in specs} == set(candidates[8:])
    assert not set(fresh) & set(excluded)
