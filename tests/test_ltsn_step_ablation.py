from __future__ import annotations

import csv
import json
from pathlib import Path

from generation.ltsn_contract import sha256_file
from generation.ltsn_step_ablation import evaluate_step4_gradient_diagnostic


def test_step4_gradient_diagnostic_reports_exact_direction_support(tmp_path: Path) -> None:
    plan_path = tmp_path / "development_generation_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "experiment_mode": "step4_gradient_diagnostic",
                "diagnostic_only": True,
                "guidance_promotion_eligible": False,
                "ood_ablation_only": True,
                "diagnostic_prompt_limit": 16,
                "corrector": {
                    "step_weights": {"4": 0.5},
                    "rms_clip_ratio": 0.005,
                    "require_all_members_out_of_band": False,
                    "require_all_member_improvement": False,
                    "minimum_member_gradient_cosine": -1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    plan_sha256 = sha256_file(plan_path)
    rows = []
    for prompt in range(16):
        for seed in range(4):
            improved = seed != 3
            rows.append(
                {
                    "pair_id": f"p{prompt}__s{seed}",
                    "prompt_id": f"p{prompt}",
                    "seed": seed,
                    "generation_plan_sha256": plan_sha256,
                    "proxy_focus_band_loss_before": 1.0,
                    "proxy_focus_band_loss_after": 0.8,
                    "exact_focus_band_loss_before": 1.0,
                    "exact_focus_band_loss_after": 0.9 if improved else 1.1,
                    "latent_changed": "true",
                    "authorization_scope": "development_only",
                    "ood_ablation_only": "true",
                    "diagnostic_only": "true",
                }
            )
    pair_table = tmp_path / "development_pairs_raw.csv"
    with pair_table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "step4_gradient_diagnostic.json"

    report = evaluate_step4_gradient_diagnostic(
        pair_table=pair_table,
        generation_plan=plan_path,
        output_path=output,
        bootstrap_resamples=100,
    )

    assert report["status"] == "signal_supported"
    assert report["guidance_promotion_eligible"] is False
    assert report["pairs"] == 64
    assert report["proxy_optimized_both_oob_pairs"] == 64
    assert report["non_tied_exact_direction_agreement"] == 0.75
    assert report["median_exact_improvement"] > 0
