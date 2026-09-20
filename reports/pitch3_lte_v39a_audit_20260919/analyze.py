"""Read-only audit of archived V3.9-A experiments; no Torch or model selection."""

# Imports below the explicit repository path setup are intentional.
# ruff: noqa: E402
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from generation.pitch3_lte_metrics import pitch3_lte_metrics
from generation.pitch3_lte_v38b_protocol import correlation, describe_predictions
from generation.pitch3_lte_v39a_protocol import (
    file_hash,
    read_csv,
    read_json,
    verify_diagnostic,
    verify_manifest,
)
from scripts.summarize_pitch3_lte_v38b import _validate_prediction_labels, average_rows
from scripts.summarize_pitch3_lte_v39a import summarize

RUN = ROOT / "runs/pitch3_lte_v39a"
OUT = Path(__file__).resolve().parent
DATA = ROOT / "runs/pitch3_lte_v3/exact_local_dataset/pitch3_lte_examples.csv"
rows = read_csv(DATA)
truth = {r["sample_id"]: r for r in rows}
coords = {
    k: SimpleNamespace(coordinates=json.loads(v["coordinates_json"])) for k, v in truth.items()
}


def mean(values):
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def spread(values):
    a = np.array([v for v in values if v is not None])
    return (
        dict(
            n=len(a),
            min=float(a.min()),
            median=float(np.median(a)),
            mean=float(a.mean()),
            max=float(a.max()),
        )
        if len(a)
        else None
    )


def compact_fit(fit):
    families = list(fit["diagnostics"]["families"].values())
    m = fit["metrics"]
    return {
        "family_mean": mean(list(m["prompt_family_spearman"].values())),
        "family_min": m["minimum_prompt_family_spearman"],
        "families_passing": sum(v >= 0.5 for v in m["prompt_family_spearman"].values()),
        "energy_rho": m["direct_energy_spearman"],
        "rank_accuracy": m["same_prompt_ranking_accuracy"],
        "direction_accuracy": m["local_direction_sign_accuracy"],
        "zero_positive_auc": mean([f["zero_positive_auc"] for f in families]),
        "low_rho": mean([f["zero_and_low_positive_rho"] for f in families]),
        "positive_rho": mean([f["positive_only_rho"] for f in families]),
        "coordinate_rho": [mean([f["coordinate_rho"][i] for f in families]) for i in range(3)],
        "coordinate_mae_width": [
            mean([f["coordinate_mae_width"][i] for f in families]) for i in range(3)
        ],
    }


audit = {
    "scope": "archived fit diagnostics and train-family CV; no development or model replay",
    "integrity": {},
    "summaries": {},
    "runs": [],
    "cv": {},
}
errors = []
manifests = sorted(RUN.glob("cv/**/pitch3_lte_manifest.json"))
for path in manifests:
    try:
        m, p = verify_manifest(path, rows)
        with zipfile.ZipFile(path.parent / Path(m["checkpoint"]).name) as z:
            assert z.testzip() is None
        for split in ("train", "selection"):
            predictions = read_csv(path.parent / f"pitch3_lte_{split}_predictions.csv")
            for row in predictions:
                _validate_prediction_labels(row, truth[row["sample_id"]])
            expected = m["train_metrics" if split == "train" else "best_development_metrics"]
            assert pitch3_lte_metrics(predictions) == expected
    except Exception as exc:
        errors.append({"path": str(path.relative_to(ROOT)), "error": str(exc)})
diagnostics = sorted(RUN.glob("diagnostics/**/pitch3_lte_diagnostic_complete.json"))
for path in diagnostics:
    try:
        d = verify_diagnostic(path)
        preds = read_csv(path.parent / "pitch3_lte_train_predictions.csv")
        for row in preds:
            _validate_prediction_labels(row, truth[row["sample_id"]])
        assert pitch3_lte_metrics(preds) == d["final_fit_metrics"]
    except Exception as exc:
        errors.append({"path": str(path.relative_to(ROOT)), "error": str(exc)})
audit["integrity"] = {
    "cv_manifests": len(manifests),
    "diagnostic_completions": len(diagnostics),
    "errors": errors,
    "synthetic_checks_present": bool(list(RUN.glob("checks/**/*.json"))),
    "formal_final_present": (RUN / "final").exists(),
}
reference = read_json(manifests[0])
audit["integrity"]["source_hash_matches"] = {
    name: file_hash(ROOT / "src/generation" / name) == digest
    for name, digest in reference["implementation_sha256"].items()
}
audit["integrity"]["source_hash_matches_after_crlf_to_lf"] = {
    name: hashlib.sha256(
        (ROOT / "src/generation" / name).read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    == digest
    for name, digest in reference["implementation_sha256"].items()
}
audit["integrity"]["dataset_hash_matches"] = (
    file_hash(DATA) == reference["training_manifest_sha256"]
)
audit["integrity"]["config_hash_matches"] = (
    file_hash(ROOT / "configs/pitch3_lte_v39a.toml") == reference["source_training_config_sha256"]
)
fingerprint_path = ROOT / "metadata/focus_pitch3_fingerprint_v1.json"
audit["integrity"]["fingerprint_hash_matches"] = (
    file_hash(fingerprint_path) == reference["fingerprint_json_sha256"]
)
assert audit["integrity"]["fingerprint_hash_matches"]
fingerprint = read_json(fingerprint_path)
lower = np.array(fingerprint["focus_target"]["coordinate_lower"])
upper = np.array(fingerprint["focus_target"]["coordinate_upper"])
distance_weights = np.array(fingerprint["distance_weights"])


def band_components(q):
    return (np.maximum(lower - q, 0) ** 2 + np.maximum(q - upper, 0) ** 2) * distance_weights


audit["teacher_decomposition"] = {}
train_base = [r for r in rows if r["split"] == "train" and r["source_kind"] == "base_step4_seed"]
q = np.array([coords[r["sample_id"]].coordinates for r in train_base])
parts = band_components(q)
y = np.array([float(r["exact_band"]) for r in train_base])
audit["teacher_decomposition"]["max_band_reconstruction_error"] = float(abs(parts.sum(1) - y).max())
for name, mask in (
    ("all_positive", y > 0),
    ("lowest_positive_third_train_only", (y > 0) & (y <= np.quantile(y[y > 0], 1 / 3))),
):
    p = parts[mask]
    audit["teacher_decomposition"][name] = {
        "n": int(mask.sum()),
        "mean_per_row_component_fraction": (p / p.sum(1, keepdims=True)).mean(0).tolist(),
        "dominant_component_counts": np.bincount(p.argmax(1), minlength=3).tolist(),
    }
for seeds in ([20260941], [20260941, 20260942, 20260943]):
    key = "_".join(map(str, seeds))
    try:
        fresh = summarize(RUN, DATA, seeds)
        saved = read_json(RUN / f"summary/pitch3_lte_v39a_cv_baseline_transition_{key}.json")
        # Absolute artifact paths differ after server-to-desktop synchronization.
        for report in (saved, fresh):
            for variant in report["variants"].values():
                for fold in variant["folds"]:
                    for member in fold["members"]:
                        member.pop("manifest", None)
        audit["summaries"][key] = {"reproduced": fresh == saved, "report": fresh}
    except Exception as exc:
        audit["summaries"][key] = {"error": str(exc)}

for path in manifests + diagnostics:
    folder = path.parent
    meta = read_json(path)
    stage = "cv" if "manifest" in path.name else "diagnostic"
    variant = meta["experiment_contract"]["variant"] if stage == "cv" else meta["variant"]
    history = read_json(folder / "pitch3_lte_training_history.json")["history"]
    initial = read_json(folder / "pitch3_lte_initialization.json")
    curve = [{"epoch": 0, **compact_fit(read_json(folder / "pitch3_lte_fit_epoch_zero.json"))}]
    for h in history:
        curve.append(
            {
                "epoch": h["epoch"],
                **compact_fit(h["fit_evaluation"]),
                **{
                    k: h[k]
                    for k in (
                        "normalized_training_contributions",
                        "gradient_norm_mean",
                        "gradient_norm_max",
                        "clipping_fraction",
                        "block_gradient_norms_mean",
                        "block_parameter_updates",
                        "training_seconds",
                    )
                },
            }
        )
    snapshots = {
        phase: {
            k: v
            for k, v in read_json(folder / f"pitch3_lte_gradient_{phase}.json").items()
            if k != "sample_ids"
        }
        for phase in ("initial", "final")
    }
    audit["runs"].append(
        {
            "stage": stage,
            "variant": variant,
            "fold": meta["cv_fold"],
            "seed": initial["seed"],
            "fit_families": sorted(
                history[-1]["fit_evaluation"]["metrics"]["prompt_family_spearman"]
            ),
            "steps": sum(h["optimizer_steps"] for h in history),
            "curve": curve,
            "snapshots": snapshots,
            "peak_cuda_gib": meta.get("peak_cuda_allocated_bytes", 0) / 2**30,
            "initialization_equal": initial["outputs_and_losses_equal"],
        }
    )

for variant in ("baseline", "transition"):
    ensemble = {"train": {}, "selection": {}}
    all_outer = []
    per_seed = {}
    for seed in (20260941, 20260942, 20260943):
        rho = {}
        correct = pairs = 0
        rank_correct = rank_pairs = 0
        for fold in range(5):
            m = read_json(
                RUN / f"cv/{variant}/fold_{fold}/seed_{seed}/models/pitch3_lte_manifest.json"
            )["best_development_metrics"]
            rho.update(m["prompt_family_spearman"])
            pairs += m["local_direction_pairs"]
            correct += round(m["local_direction_pairs"] * m["local_direction_sign_accuracy"])
            rank_pairs += m["same_prompt_rank_pairs"]
            rank_correct += round(m["same_prompt_rank_pairs"] * m["same_prompt_ranking_accuracy"])
        per_seed[seed] = {
            "family_rho": rho,
            "families_passing": sum(v >= 0.5 for v in rho.values()),
            "mean_rho": mean(list(rho.values())),
            "min_rho": min(rho.values()),
            "direction_correct": correct,
            "direction_pairs": pairs,
            "rank_correct": rank_correct,
            "rank_pairs": rank_pairs,
        }
    for fold in range(5):
        folders = [
            RUN / f"cv/{variant}/fold_{fold}/seed_{seed}/models"
            for seed in (20260941, 20260942, 20260943)
        ]
        stats = read_json(folders[0] / "pitch3_lte_training_statistics.json")
        for split in ("train", "selection"):
            predictions = average_rows(
                [read_csv(p / f"pitch3_lte_{split}_predictions.csv") for p in folders]
            )
            if split == "selection":
                all_outer.extend(predictions)
            d = describe_predictions(
                predictions,
                coords,
                stats["energy_stratification"]["thresholds"],
                stats["local_training_scales"]["local_derivative_scale"],
                stats["coordinate_auxiliary"]["coordinate_widths"],
            )
            for family, f in d["families"].items():
                group = [
                    r
                    for r in predictions
                    if r["prompt_family"] == family and r["source_kind"] == "base_step4_seed"
                ]
                y = np.array([float(r["exact_energy"]) for r in group])
                low = y <= stats["energy_stratification"]["thresholds"][1]
                exact_q = np.array([coords[r["sample_id"]].coordinates for r in group])
                predicted_q = np.array([json.loads(r["predicted_coordinates_json"]) for r in group])
                f["low_n"] = int(low.sum())
                f["low_coordinate_rho"] = [
                    correlation(exact_q[low, i], predicted_q[low, i]) for i in range(3)
                ]
                f["low_threshold"] = stats["energy_stratification"]["thresholds"][1]
                f["zero_n"] = int((y == 0).sum())
                f["energy_quantiles"] = np.quantile(y, [0, 0.25, 0.5, 0.75, 1]).tolist()
                ensemble[split][f"fold_{fold}/{family}"] = f
    aggregates = {}
    for split, fs in ensemble.items():
        values = list(fs.values())
        aggregates[split] = {
            k: spread([f[k] for f in values])
            for k in (
                "rho",
                "zero_positive_auc",
                "positive_only_rho",
                "zero_and_low_positive_rho",
                "energy_mae",
            )
        }
        for k in ("coordinate_rho", "coordinate_mae_width", "low_coordinate_rho"):
            aggregates[split][k] = [spread([f[k][i] for f in values]) for i in range(3)]
        aggregates[split]["families_passing"] = sum(f["rho"] >= 0.5 for f in values)
        aggregates[split]["families_total"] = len(values)
    audit["cv"][variant] = {
        "per_seed": per_seed,
        "ensemble_families": ensemble,
        "ensemble_aggregates": aggregates,
        "concatenated_outer_metrics_descriptive_only": pitch3_lte_metrics(all_outer),
    }

# Curves average runs, not independent data replicates; diagnostic fit sets repeat.
audit["training_aggregates"] = {}
for stage in ("diagnostic", "cv"):
    audit["training_aggregates"][stage] = {}
    for variant in ("baseline", "transition"):
        runs = [r for r in audit["runs"] if r["stage"] == stage and r["variant"] == variant]
        last = [r["curve"][-1] for r in runs]
        all_epochs = [e for r in runs for e in r["curve"][1:]]
        groups = last[0]["block_gradient_norms_mean"]
        aggregate = {
            "run_count": len(runs),
            "unique_fit_sets": sorted({tuple(r["fit_families"]) for r in runs}),
            "last": {
                k: spread([r[k] for r in last])
                for k in (
                    "family_mean",
                    "family_min",
                    "low_rho",
                    "positive_rho",
                    "zero_positive_auc",
                    "energy_rho",
                    "clipping_fraction",
                    "gradient_norm_mean",
                    "gradient_norm_max",
                )
            },
            "all_epochs_clipping": mean([e["clipping_fraction"] for e in all_epochs]),
            "blocks": {},
            "curves": [],
        }
        for g in groups:
            aggregate["blocks"][g] = {
                "final_before": spread(
                    [r["block_gradient_norms_mean"][g]["before_clip"] for r in last]
                ),
                "final_after": spread(
                    [r["block_gradient_norms_mean"][g]["after_clip"] for r in last]
                ),
                "final_relative_update": spread(
                    [r["block_parameter_updates"][g]["relative_to_epoch_start"] for r in last]
                ),
                "min_epoch_gradient": min(
                    e["block_gradient_norms_mean"][g]["before_clip"] for e in all_epochs
                ),
                "min_epoch_update": min(
                    e["block_parameter_updates"][g]["epoch_net_update_l2"] for e in all_epochs
                ),
            }
        for idx in range(len(runs[0]["curve"])):
            aggregate["curves"].append(
                {
                    "epoch": idx,
                    **{
                        k: mean([r["curve"][idx][k] for r in runs])
                        for k in (
                            "family_mean",
                            "family_min",
                            "low_rho",
                            "zero_positive_auc",
                            "energy_rho",
                        )
                    },
                }
            )
        aggregate["final_gradient_cosine"] = {
            a: {
                b: spread([r["snapshots"]["final"]["gradient_cosine"][a][b] for r in runs])
                for b in last[0]["normalized_training_contributions"]
            }
            for a in last[0]["normalized_training_contributions"]
        }
        aggregate["normalized_contributions_first_last"] = {
            k: [
                mean([r["curve"][idx]["normalized_training_contributions"][k] for r in runs])
                for idx in (1, -1)
            ]
            for k in last[0]["normalized_training_contributions"]
        }
        aggregate["training_seconds_mean"] = mean(
            [sum(e["training_seconds"] for e in r["curve"][1:]) for r in runs]
        )
        aggregate["peak_cuda_gib"] = spread([r["peak_cuda_gib"] for r in runs])
        audit["training_aggregates"][stage][variant] = aggregate

# Untrained, retrospective readout probe: no checkpoint/gate/readout replacement.
# Apply the frozen nonlinear Band per member BEFORE averaging member energies.
audit["coordinate_readout_probe_not_a_candidate"] = {}
for variant in ("baseline", "transition"):
    probe_rows = []
    families_low = []
    for fold in range(5):
        members = []
        folders = [
            RUN / f"cv/{variant}/fold_{fold}/seed_{s}/models"
            for s in (20260941, 20260942, 20260943)
        ]
        for folder in folders:
            predictions = read_csv(folder / "pitch3_lte_selection_predictions.csv")
            for row in predictions:
                energy = float(
                    np.log1p(
                        band_components(
                            np.array(json.loads(row["predicted_coordinates_json"]))
                        ).sum()
                    )
                )
                row["predicted_energy"] = energy
                row["predicted_global_energy"] = energy
            members.append(predictions)
        combined = average_rows(members)
        probe_rows.extend(combined)
        stats = read_json(folders[0] / "pitch3_lte_training_statistics.json")
        d = describe_predictions(
            combined,
            coords,
            stats["energy_stratification"]["thresholds"],
            stats["local_training_scales"]["local_derivative_scale"],
            stats["coordinate_auxiliary"]["coordinate_widths"],
        )
        families_low.extend(f["zero_and_low_positive_rho"] for f in d["families"].values())
    m = pitch3_lte_metrics(probe_rows)
    audit["coordinate_readout_probe_not_a_candidate"][variant] = {
        "metrics_descriptive_only": m,
        "mean_family_rho": mean(list(m["prompt_family_spearman"].values())),
        "families_passing": sum(v >= 0.5 for v in m["prompt_family_spearman"].values()),
        "mean_low_rho": mean(families_low),
    }

(OUT / "audit.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
)
print(
    json.dumps(
        {
            "integrity": audit["integrity"],
            "summary_reproduction": {
                k: v.get("reproduced", v.get("error")) for k, v in audit["summaries"].items()
            },
            "output": str(OUT / "audit.json"),
        },
        indent=2,
    )
)
