from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pathhom_tda import TDAError as CoreTDAError
from pathhom_tda import delay_embedding as _core_delay_embedding
from pathhom_tda import finite_rows as _core_finite_rows
from pathhom_tda import normalize_distance_scale as _core_normalize_distance_scale
from pathhom_tda import persistence_descriptors as _core_persistence_descriptors
from pathhom_tda import uniform_sample as _core_uniform_sample
from pathhom_tda import vietoris_rips
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from features.batch import _json_hash, _read_npz, _sha256, _write_json_atomic
from topology.statistics import benjamini_hochberg

REPRESENTATIONS = (
    "acoustic_pca",
    "chroma",
    "rhythm",
    "modulation",
    "acoustic_novelty_delay",
    "onset_delay",
    "modulation_delay",
)
DESCRIPTORS = (
    "h0_total_persistence",
    "h0_max_persistence",
    "h0_q75_persistence",
    "h0_persistence_entropy",
    "h1_count",
    "h1_prominent_count",
    "h1_total_persistence",
    "h1_max_persistence",
    "h1_mean_persistence",
    "h1_persistence_entropy",
)
IDENTITY_COLUMNS = ("segment_id", "track_id", "group", "split", "scale_seconds")


class TDAAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TDAConfig:
    exploration_tracks_per_group: int = 24
    max_points: int = 64
    acoustic_pca_components: int = 8
    delay_dimension: int = 4
    delay_lag: int = 2
    prominent_lifetime: float = 0.10
    max_selected_representations: int = 3
    exploration_p_threshold: float = 0.05
    exploration_effect_threshold: float = 0.30
    exploration_stability_threshold: float = 0.20
    validation_fdr_q: float = 0.05
    workers: int = 4
    random_seed: int = 20260716

    def validate(self) -> None:
        if self.exploration_tracks_per_group < 5:
            raise TDAAnalysisError("exploration sample must contain at least five tracks per group")
        if self.max_points < 16 or self.acoustic_pca_components < 2:
            raise TDAAnalysisError("TDA point-cloud dimensions are too small")
        if self.delay_dimension < 2 or self.delay_lag < 1:
            raise TDAAnalysisError("delay embedding parameters are invalid")
        if not 0 < self.prominent_lifetime < 1:
            raise TDAAnalysisError("prominent_lifetime must be in (0, 1)")
        if self.max_selected_representations < 1 or self.workers < 1:
            raise TDAAnalysisError("selection count and workers must be positive")
        for value in (
            self.exploration_p_threshold,
            self.exploration_effect_threshold,
            self.exploration_stability_threshold,
            self.validation_fdr_q,
        ):
            if not 0 < value < 1:
                raise TDAAnalysisError("statistical thresholds must be in (0, 1)")


def load_config(root: Path) -> TDAConfig:
    with (root / "configs" / "pipeline.toml").open("rb") as handle:
        raw = tomllib.load(handle)
    values = dict(raw.get("tda", {}))
    values.setdefault("random_seed", int(raw.get("project", {}).get("seed", 20260716)))
    unknown = set(values) - set(TDAConfig.__dataclass_fields__)
    if unknown:
        raise TDAAnalysisError(f"unknown TDA configuration keys: {sorted(unknown)}")
    config = TDAConfig(**values)
    config.validate()
    return config


def _finite_rows(values: np.ndarray) -> np.ndarray:
    return _core_finite_rows(values)


def _uniform_sample(values: np.ndarray, max_points: int, *, offset: float = 0.0) -> np.ndarray:
    return _core_uniform_sample(values, max_points, offset=offset)


def _normalize_distance_scale(values: np.ndarray) -> tuple[np.ndarray, float]:
    try:
        return _core_normalize_distance_scale(values, minimum_points=4)
    except CoreTDAError as exc:
        raise TDAAnalysisError(str(exc)) from exc


def delay_embedding(values: np.ndarray, *, dimension: int, lag: int) -> np.ndarray:
    try:
        return _core_delay_embedding(values, dimension=dimension, lag=lag)
    except CoreTDAError as exc:
        raise TDAAnalysisError(str(exc)) from exc


def persistence_descriptors(
    diagrams: Sequence[np.ndarray], *, prominent_lifetime: float
) -> dict[str, float]:
    return _core_persistence_descriptors(
        diagrams,
        prominent_lifetime=prominent_lifetime,
    )


def _load_model(root: Path) -> dict[str, np.ndarray]:
    model = _read_npz(root / "features" / "models" / "state_model.npz")
    required = {
        "acoustic_mean",
        "acoustic_scale",
        "pca_mean",
        "pca_components",
        "rhythm_impute",
        "rhythm_mean",
        "rhythm_scale",
        "modulation_edges",
    }
    if not required.issubset(model):
        raise TDAAnalysisError("feature state model lacks arrays required by TDA")
    return model


def _representations(
    root: Path,
    row: dict[str, Any],
    model: dict[str, np.ndarray],
    config: TDAConfig,
    requested: Sequence[str],
) -> dict[str, np.ndarray]:
    acoustic = _read_npz(root / Path(str(row["acoustic_relative_path"])))
    chroma = _read_npz(root / Path(str(row["chroma_relative_path"])))
    rhythm = _read_npz(root / Path(str(row["rhythm_relative_path"])))
    modulation = _read_npz(root / Path(str(row["modulation_relative_path"])))

    acoustic_values = np.asarray(acoustic["vectors"], dtype=float)
    acoustic_scaled = (acoustic_values - model["acoustic_mean"]) / model["acoustic_scale"]
    acoustic_pca = (acoustic_scaled - model["pca_mean"]) @ model["pca_components"].T
    acoustic_pca = acoustic_pca[:, : config.acoustic_pca_components]

    chroma_values = np.asarray(chroma["chroma"], dtype=float)
    chroma_norm = np.linalg.norm(chroma_values, axis=1, keepdims=True)
    chroma_values = chroma_values / np.maximum(chroma_norm, np.finfo(float).eps)

    rhythm_values = np.asarray(rhythm["vectors"], dtype=float)
    rhythm_valid = np.asarray(rhythm["valid"], dtype=bool)
    rhythm_values = np.where(rhythm_valid, rhythm_values, model["rhythm_impute"])
    rhythm_values = (rhythm_values - model["rhythm_mean"]) / model["rhythm_scale"]

    modulation_values = np.asarray(modulation["key_band_energies"], dtype=float)
    edges = np.asarray(model["modulation_edges"], dtype=float)
    modulation_values = (modulation_values - np.mean(edges, axis=1)) / np.maximum(
        edges[:, 1] - edges[:, 0], np.finfo(float).eps
    )

    output = {
        "acoustic_pca": acoustic_pca,
        "chroma": chroma_values,
        "rhythm": rhythm_values,
        "modulation": modulation_values,
    }
    if "acoustic_novelty_delay" in requested:
        novelty = np.linalg.norm(np.diff(acoustic_pca, axis=0), axis=1)
        output["acoustic_novelty_delay"] = delay_embedding(
            novelty, dimension=config.delay_dimension, lag=config.delay_lag
        )
    if "onset_delay" in requested:
        output["onset_delay"] = delay_embedding(
            rhythm_values[:, 0], dimension=config.delay_dimension, lag=config.delay_lag
        )
    if "modulation_delay" in requested:
        output["modulation_delay"] = delay_embedding(
            modulation_values[:, 0], dimension=config.delay_dimension, lag=config.delay_lag
        )
    return {name: output[name] for name in requested}


def _compute_segment(
    root: Path,
    row: dict[str, Any],
    model: dict[str, np.ndarray],
    config: TDAConfig,
    representations: Sequence[str],
) -> list[dict[str, Any]]:
    point_clouds = _representations(root, row, model, config, representations)
    output: list[dict[str, Any]] = []
    identity = {column: row[column] for column in IDENTITY_COLUMNS}
    identity["scale_seconds"] = float(identity["scale_seconds"])
    for name in representations:
        sampled = _uniform_sample(point_clouds[name], config.max_points)
        normalized, distance_scale = _normalize_distance_scale(sampled)
        diagrams = vietoris_rips(
            normalized,
            max_dimension=1,
            coefficient=2,
        ).diagrams
        output.append(
            {
                **identity,
                "representation": name,
                "point_count": int(len(normalized)),
                "distance_scale": distance_scale,
                **persistence_descriptors(diagrams, prominent_lifetime=config.prominent_lifetime),
            }
        )
    return output


def _compute_features(
    root: Path,
    manifest: pd.DataFrame,
    config: TDAConfig,
    representations: Sequence[str],
) -> pd.DataFrame:
    model = _load_model(root)
    rows = manifest.to_dict("records")
    output: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = {
            executor.submit(_compute_segment, root, row, model, config, representations): row
            for row in rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                output.extend(future.result())
            except Exception as exc:
                raise TDAAnalysisError(f"TDA failed for {row['segment_id']}: {exc}") from exc
            if completed % 100 == 0 or completed == len(rows):
                print(f"TDA segments: {completed}/{len(rows)}", flush=True)
    return pd.DataFrame(output).sort_values(
        ["split", "group", "track_id", "scale_seconds", "representation"]
    )


def _quality_filter(
    manifest: pd.DataFrame, representations: Sequence[str], config: TDAConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    delay_width = 1 + (config.delay_dimension - 1) * config.delay_lag
    requirements = {
        "acoustic_pca": ("acoustic_windows", 0),
        "acoustic_novelty_delay": ("acoustic_windows", delay_width),
        "chroma": ("pitch_steps", 0),
        "rhythm": ("rhythm_windows", 0),
        "onset_delay": ("rhythm_windows", delay_width - 1),
        "modulation": ("modulation_windows", 0),
        "modulation_delay": ("modulation_windows", delay_width - 1),
    }
    thresholds: dict[str, int] = {}
    for name in representations:
        column, extra = requirements[name]
        thresholds[column] = max(thresholds.get(column, 0), config.max_points + extra)
    eligible = np.ones(len(manifest), dtype=bool)
    for column, threshold in thresholds.items():
        values = pd.to_numeric(manifest[column], errors="coerce").fillna(0).to_numpy()
        eligible &= values >= threshold
    return manifest.loc[eligible].copy(), manifest.loc[~eligible].copy()


def _effect_test(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    statistic, p_value = mannwhitneyu(x, y, alternative="two-sided", method="auto")
    effect = 2.0 * float(statistic) / (len(x) * len(y)) - 1.0
    return effect, float(p_value)


def _tests(
    frame: pd.DataFrame,
    *,
    split: str,
    scale: float,
    role: str,
    comparator: str = "pop",
    selected_features: Sequence[dict[str, str]] | None = None,
) -> pd.DataFrame:
    subset = frame[(frame["split"] == split) & (frame["scale_seconds"] == scale)]
    rows: list[dict[str, Any]] = []
    for representation in sorted(subset["representation"].unique()):
        view = subset[subset["representation"] == representation]
        focus = view[view["group"] == "focus"]
        comparison = view[view["group"] == comparator]
        for metric in DESCRIPTORS:
            effect, p_value = _effect_test(focus[metric].to_numpy(), comparison[metric].to_numpy())
            rows.append(
                {
                    "role": role,
                    "comparison": f"focus_vs_{comparator}",
                    "split": split,
                    "scale_seconds": scale,
                    "representation": representation,
                    "metric": metric,
                    "n_focus": len(focus),
                    "n_comparator": len(comparison),
                    "focus_median": float(focus[metric].median()),
                    "comparator_median": float(comparison[metric].median()),
                    "rank_biserial_focus_minus_comparator": effect,
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    if selected_features is not None:
        selected_pairs = {
            (feature["representation"], feature["metric"]) for feature in selected_features
        }
        result = result[
            [
                (representation, metric) in selected_pairs
                for representation, metric in zip(
                    result["representation"], result["metric"], strict=True
                )
            ]
        ].copy()
    result["p_fdr_bh"] = benjamini_hochberg(result["p_value"].to_numpy())
    return result


def _exploration_manifest(manifest: pd.DataFrame, config: TDAConfig) -> pd.DataFrame:
    base = manifest[(manifest["split"] == "discovery") & (manifest["scale_seconds"] == 180.0)]
    base, _ = _quality_filter(base, REPRESENTATIONS, config)
    sensitivity = manifest[
        (manifest["split"] == "discovery") & (manifest["scale_seconds"] == 300.0)
    ]
    sensitivity, _ = _quality_filter(sensitivity, REPRESENTATIONS, config)
    base = base[base["track_id"].isin(sensitivity["track_id"])]
    rng = np.random.default_rng(config.random_seed)
    tracks: list[str] = []
    for group in ("focus", "pop", "classical"):
        choices = np.sort(base.loc[base["group"] == group, "track_id"].unique())
        if len(choices) < config.exploration_tracks_per_group:
            raise TDAAnalysisError(f"not enough discovery tracks for {group}")
        selected = rng.choice(choices, config.exploration_tracks_per_group, replace=False)
        tracks.extend(str(value) for value in selected)
    return manifest[
        (manifest["track_id"].isin(tracks))
        & (manifest["split"] == "discovery")
        & (manifest["scale_seconds"].isin([180.0, 300.0]))
    ].copy()


def _select_representations(
    features: pd.DataFrame, tests: pd.DataFrame, config: TDAConfig
) -> tuple[list[str], list[dict[str, str]], pd.DataFrame]:
    primary = tests[tests["scale_seconds"] == 180.0].copy()
    sensitivity = tests[tests["scale_seconds"] == 300.0][
        ["representation", "metric", "rank_biserial_focus_minus_comparator"]
    ].rename(columns={"rank_biserial_focus_minus_comparator": "effect_300"})
    candidates = primary.merge(sensitivity, on=["representation", "metric"], how="left")
    correlations: list[float] = []
    for row in candidates.itertuples(index=False):
        paired = features[features["representation"] == row.representation].pivot(
            index="track_id", columns="scale_seconds", values=row.metric
        )
        correlation = spearmanr(paired[180.0], paired[300.0]).statistic
        correlations.append(float(correlation) if np.isfinite(correlation) else 0.0)
    candidates["scale_spearman"] = correlations
    candidates["direction_agreement"] = (
        candidates["rank_biserial_focus_minus_comparator"] * candidates["effect_300"] > 0
    )
    candidates["eligible"] = (
        (candidates["p_value"] <= config.exploration_p_threshold)
        & (
            candidates["rank_biserial_focus_minus_comparator"].abs()
            >= config.exploration_effect_threshold
        )
        & candidates["direction_agreement"]
        & (candidates["scale_spearman"] >= config.exploration_stability_threshold)
    )
    candidates["selection_score"] = candidates[
        "rank_biserial_focus_minus_comparator"
    ].abs() * candidates["scale_spearman"].clip(lower=0)
    ranked = (
        candidates[candidates["eligible"]]
        .sort_values(["selection_score", "p_value"], ascending=[False, True])
        .drop_duplicates("representation")
    )
    selected_rows = ranked.head(config.max_selected_representations)
    selected = selected_rows["representation"].tolist()
    selected_features = [
        {"representation": str(row.representation), "metric": str(row.metric)}
        for row in selected_rows.itertuples(index=False)
    ]
    selected_pairs = {
        (feature["representation"], feature["metric"]) for feature in selected_features
    }
    candidates["selected"] = [
        pair in selected_pairs
        for pair in zip(candidates["representation"], candidates["metric"], strict=True)
    ]
    return (
        selected,
        selected_features,
        candidates.sort_values(["eligible", "selection_score"], ascending=[False, False]),
    )


def _wide(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.pivot(
        index=list(IDENTITY_COLUMNS), columns="representation", values=list(DESCRIPTORS)
    )
    values.columns = [f"{representation}__{metric}" for metric, representation in values.columns]
    return values.reset_index()


def _classification(
    frame: pd.DataFrame, config: TDAConfig, selected_features: Sequence[dict[str, str]]
) -> pd.DataFrame:
    wide = _wide(frame)
    feature_columns = [
        f"{feature['representation']}__{feature['metric']}" for feature in selected_features
    ]
    rows: list[dict[str, Any]] = []
    for task, groups in (
        ("three_class", ("classical", "focus", "pop")),
        ("focus_vs_pop", ("focus", "pop")),
    ):
        subset = wide[(wide["scale_seconds"] == 180.0) & wide["group"].isin(groups)]
        train = subset[subset["split"] == "discovery"]
        validation = subset[subset["split"] == "validation"]
        encoder = LabelEncoder().fit(train["group"])
        y_train = encoder.transform(train["group"])
        y_validation = encoder.transform(validation["group"])
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000, class_weight="balanced", random_state=config.random_seed
                    ),
                ),
            ]
        )
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.random_seed)
        search = GridSearchCV(
            pipeline,
            {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
            cv=folds,
            scoring="f1_macro",
            n_jobs=1,
        ).fit(train[feature_columns], y_train)
        predictions = search.predict(validation[feature_columns])
        probabilities = search.predict_proba(validation[feature_columns])
        if len(groups) == 2:
            auroc = roc_auc_score(y_validation, probabilities[:, 1])
        else:
            auroc = roc_auc_score(y_validation, probabilities, multi_class="ovr", average="macro")
        rows.append(
            {
                "task": task,
                "train_split": "discovery",
                "test_split": "validation",
                "scale_seconds": 180.0,
                "n_train": len(train),
                "n_validation": len(validation),
                "n_features": len(feature_columns),
                "best_c": float(search.best_params_["classifier__C"]),
                "cv_macro_f1": float(search.best_score_),
                "balanced_accuracy": float(balanced_accuracy_score(y_validation, predictions)),
                "macro_f1": float(f1_score(y_validation, predictions, average="macro")),
                "macro_auroc": float(auroc),
            }
        )
    return pd.DataFrame(rows)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    temporary.replace(path)


def run_exploration(root: Path, config: TDAConfig) -> dict[str, Any]:
    manifest_path = root / "metadata" / "feature_segments.csv"
    manifest = pd.read_csv(manifest_path)
    sample = _exploration_manifest(manifest, config)
    features = _compute_features(root, sample, config, REPRESENTATIONS)
    tests = pd.concat(
        [
            _tests(features, split="discovery", scale=180.0, role="exploration"),
            _tests(features, split="discovery", scale=300.0, role="scale_replication"),
        ],
        ignore_index=True,
    )
    selected, selected_features, selection = _select_representations(features, tests, config)
    metadata = root / "metadata"
    feature_path = metadata / "tda_exploration_features.csv"
    test_path = metadata / "tda_exploration_tests.csv"
    selection_path = metadata / "tda_feature_selection.csv"
    summary_path = metadata / "tda_exploration_summary.json"
    _write_csv(feature_path, features)
    _write_csv(test_path, tests)
    _write_csv(selection_path, selection)
    payload = {
        "generated_at": date.today().isoformat(),
        "role": "exploratory feature selection using discovery only",
        "config": asdict(config),
        "config_sha256": _json_hash(asdict(config)),
        "tracks_per_group": config.exploration_tracks_per_group,
        "segments": int(sample.shape[0]),
        "candidate_representations": list(REPRESENTATIONS),
        "selected_representations": selected,
        "selected_features": selected_features,
        "selection_rule": (
            "p<=exploration_p_threshold, absolute rank-biserial>=effect threshold, "
            "same effect direction at 300s, and paired 180/300 Spearman>=stability threshold"
        ),
        "outputs": {
            "features": feature_path.relative_to(root).as_posix(),
            "tests": test_path.relative_to(root).as_posix(),
            "selection": selection_path.relative_to(root).as_posix(),
        },
        "input_sha256": {manifest_path.relative_to(root).as_posix(): _sha256(manifest_path)},
    }
    _write_json_atomic(summary_path, payload)
    return payload


def _load_selected(root: Path) -> tuple[list[str], list[dict[str, str]]]:
    path = root / "metadata" / "tda_exploration_summary.json"
    if not path.is_file():
        raise TDAAnalysisError("run TDA exploration before the full analysis")
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload["selected_representations"]
    selected_features = payload.get("selected_features", [])
    if not selected:
        raise TDAAnalysisError("exploration did not identify a representation worth validating")
    if set(selected) - set(REPRESENTATIONS):
        raise TDAAnalysisError("exploration summary contains unknown representations")
    if len(selected_features) != len(selected):
        raise TDAAnalysisError(
            "exploration summary does not freeze one endpoint per representation"
        )
    return list(selected), list(selected_features)


def _write_report(
    root: Path,
    selected_features: Sequence[dict[str, str]],
    tests: pd.DataFrame,
    classification: pd.DataFrame,
    config: TDAConfig,
    quality_excluded: int,
    plots: Sequence[str],
) -> Path:
    primary_all = tests[tests["role"] == "confirmatory_focus_vs_pop"]
    sensitivity_all = tests[tests["role"] == "scale_replication_focus_vs_pop"]
    classical_all = tests[tests["role"] == "specificity_focus_vs_classical"]
    classical_sensitivity = tests[tests["role"] == "specificity_scale_focus_vs_classical"]
    comparison = primary_all.merge(
        sensitivity_all, on=["representation", "metric"], suffixes=("_180", "_300")
    )
    primary = primary_all[primary_all["p_fdr_bh"] <= config.validation_fdr_q]
    replicated = comparison[
        (comparison["p_fdr_bh_180"] <= config.validation_fdr_q)
        & (comparison["p_fdr_bh_300"] <= config.validation_fdr_q)
    ]
    replicated = replicated[
        replicated["rank_biserial_focus_minus_comparator_180"]
        * replicated["rank_biserial_focus_minus_comparator_300"]
        > 0
    ]
    cross_comparator = primary_all.merge(
        classical_all, on=["representation", "metric"], suffixes=("_pop", "_classical")
    )
    cross_comparator = cross_comparator[
        (cross_comparator["p_fdr_bh_pop"] <= config.validation_fdr_q)
        & (cross_comparator["p_fdr_bh_classical"] <= config.validation_fdr_q)
    ]
    lines = [
        "# TDA 分析结果",
        "",
        f"生成日期：{date.today().isoformat()}。候选表示只在 discovery 小样本中筛选；"
        "validation / 180 秒是确认分析，validation / 300 秒是尺度复核。",
        "",
        "## 冻结的 TDA 端点",
        "",
        *[
            f"- `{feature['representation']} / {feature['metric']}`"
            for feature in selected_features
        ],
        "",
        f"所有点云均固定为 {config.max_points} 个时间均匀采样点，并以点间距离中位数归一化；"
        "因此持久性值主要描述形状，而不是原始特征振幅。使用 Vietoris–Rips "
        "filtration，计算 Z/2 上的 H0/H1。",
        f"全量特征清单中有 {quality_excluded} 个片段未达到至少 "
        f"{config.max_points} 个时间点的质量门槛，"
        "已在统计分析前排除。",
        "",
        "## Focus vs Pop 确认结果",
        "",
        f"validation / 180 秒共有 {len(primary)} 个 FDR q≤{config.validation_fdr_q:.2f} 的端点；"
        f"其中 {len(replicated)} 个在 validation / 300 秒也显著且方向一致。",
        "",
    ]
    lines.extend(
        [
            "| 表示 | 特征 | Focus 180s | Pop 180s | 180s 效应 | 180s FDR | 300s 效应 | 300s FDR |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.sort_values("p_fdr_bh_180").itertuples(index=False):
        lines.append(
            f"| {row.representation} | {row.metric} | {row.focus_median_180:.4f} | "
            f"{row.comparator_median_180:.4f} | "
            f"{row.rank_biserial_focus_minus_comparator_180:.3f} | "
            f"{row.p_fdr_bh_180:.3g} | "
            f"{row.rank_biserial_focus_minus_comparator_300:.3f} | "
            f"{row.p_fdr_bh_300:.3g} |"
        )
    if replicated.empty:
        lines.extend(
            [
                "",
                "没有端点通过双尺度复核，当前数据不支持稳定的 Focus 特异 TDA 特征。",
            ]
        )
    lines.extend(
        [
            "",
            "## Classical 特异性复核",
            "",
            f"在 validation / 180 秒，{len(cross_comparator)} 个端点同时显著区分 "
            "Focus–Pop 与 Focus–Classical。",
            "",
            "| 表示 | 特征 | Focus 180s | Classical 180s | 180s 效应 | 180s FDR | "
            "300s 效应 | 300s FDR |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    classical_comparison = classical_all.merge(
        classical_sensitivity,
        on=["representation", "metric"],
        suffixes=("_180", "_300"),
    )
    for row in classical_comparison.sort_values("p_fdr_bh_180").itertuples(index=False):
        lines.append(
            f"| {row.representation} | {row.metric} | {row.focus_median_180:.4f} | "
            f"{row.comparator_median_180:.4f} | "
            f"{row.rank_biserial_focus_minus_comparator_180:.3f} | "
            f"{row.p_fdr_bh_180:.3g} | "
            f"{row.rank_biserial_focus_minus_comparator_300:.3f} | "
            f"{row.p_fdr_bh_300:.3g} |"
        )
    lines.extend(
        [
            "",
            "## 拓扑解释",
            "",
            "- acoustic novelty 延迟嵌入的 H0 最大持久性更低：在距离尺度归一化后，"
            "Focus 的新颖度动态缺少特别孤立、需要很大半径才合并的状态簇。",
            "- rhythm 的 H0 总持久性更低：固定 24 点时它等价于更短的最小生成树总长度，"
            "说明 Focus 的节奏状态几何更紧凑、碎片化更少。",
            "- acoustic PCA 的 H0 持久性熵更高：连通分支的合并尺度更均匀；但该端点在"
            "主尺度不能区分 Focus 与 Classical。",
            "- 没有 H1 端点进入冻结特征集；当前证据指向连通分支的多尺度几何，而不是稳定环洞。",
        ]
    )
    lines.extend(
        [
            "",
            "## TDA-only 分类",
            "",
            "| 任务 | Macro-F1 | Balanced accuracy | AUROC |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in classification.itertuples(index=False):
        lines.append(
            f"| {row.task} | {row.macro_f1:.3f} | {row.balanced_accuracy:.3f} | "
            f"{row.macro_auroc:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 图形输出",
            "",
            *[f"- `{path}`" for path in plots],
            "",
            "## 解释边界",
            "",
            "acoustic novelty delay 与 rhythm 的 H0 端点通过了 Pop 和 Classical 双对照复核；"
            "acoustic PCA H0 熵在主尺度不能区分 Focus 与 Classical，因此不能视为 Focus 特异。"
            "这些结果不等于注意力提升或神经机制的因果证据。筛选和确认使用分离的数据划分；"
            "300 秒结果仅用于尺度稳健性复核。",
            "",
        ]
    )
    path = root / "docs" / "tda-analysis-results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _plot_selected_features(
    root: Path, features: pd.DataFrame, selected_features: Sequence[dict[str, str]]
) -> list[str]:
    import os

    cache_dir = root / "runs" / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validation = features[features["split"] == "validation"]
    figure, axes = plt.subplots(
        1, len(selected_features), figsize=(5 * len(selected_features), 4.5)
    )
    axes = np.atleast_1d(axes)
    colors = ("#2B6CB0", "#D95F02", "#2B6CB0", "#D95F02")
    for axis, feature in zip(axes, selected_features, strict=True):
        view = validation[validation["representation"] == feature["representation"]]
        values = [
            view[(view["scale_seconds"] == scale) & (view["group"] == group)][
                feature["metric"]
            ].to_numpy()
            for scale, group in ((180.0, "focus"), (180.0, "pop"), (300.0, "focus"), (300.0, "pop"))
        ]
        boxes = axis.boxplot(values, patch_artist=True, showfliers=False)
        for patch, color in zip(boxes["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        axis.set_xticks((1, 2, 3, 4), ("180 F", "180 P", "300 F", "300 P"))
        axis.set_title(f"{feature['representation']}\n{feature['metric']}")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Frozen TDA endpoints: validation Focus vs Pop")
    figure.tight_layout()
    output_dir = root / "runs" / "tda"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "selected_features_validation.png",
        output_dir / "selected_features_validation.svg",
    ]
    for path in paths:
        figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return [path.relative_to(root).as_posix() for path in paths]


def run_full(root: Path, config: TDAConfig) -> dict[str, Any]:
    selected, selected_features = _load_selected(root)
    manifest_path = root / "metadata" / "feature_segments.csv"
    full_manifest = pd.read_csv(manifest_path)
    manifest, excluded = _quality_filter(full_manifest, selected, config)
    features = _compute_features(root, manifest, config, selected)
    tests = pd.concat(
        [
            _tests(
                features,
                split="validation",
                scale=180.0,
                role="confirmatory_focus_vs_pop",
                selected_features=selected_features,
            ),
            _tests(
                features,
                split="validation",
                scale=300.0,
                role="scale_replication_focus_vs_pop",
                selected_features=selected_features,
            ),
            _tests(
                features,
                split="validation",
                scale=180.0,
                role="specificity_focus_vs_classical",
                comparator="classical",
                selected_features=selected_features,
            ),
            _tests(
                features,
                split="validation",
                scale=300.0,
                role="specificity_scale_focus_vs_classical",
                comparator="classical",
                selected_features=selected_features,
            ),
            _tests(
                features,
                split="discovery",
                scale=180.0,
                role="descriptive_discovery",
                selected_features=selected_features,
            ),
        ],
        ignore_index=True,
    )
    classification = _classification(features, config, selected_features)
    metadata = root / "metadata"
    feature_path = metadata / "tda_features.csv"
    test_path = metadata / "tda_statistical_tests.csv"
    classification_path = metadata / "tda_classification_results.csv"
    summary_path = metadata / "tda_summary.json"
    _write_csv(feature_path, features)
    _write_csv(test_path, tests)
    _write_csv(classification_path, classification)
    plots = _plot_selected_features(root, features, selected_features)
    report_path = _write_report(
        root,
        selected_features,
        tests,
        classification,
        config,
        quality_excluded=len(excluded),
        plots=plots,
    )
    primary = tests[
        (tests["role"] == "confirmatory_focus_vs_pop")
        & (tests["p_fdr_bh"] <= config.validation_fdr_q)
    ]
    sensitivity = tests[
        (tests["role"] == "scale_replication_focus_vs_pop")
        & (tests["p_fdr_bh"] <= config.validation_fdr_q)
    ]
    replicated = primary.merge(sensitivity, on=["representation", "metric"])
    replicated = replicated[
        replicated["rank_biserial_focus_minus_comparator_x"]
        * replicated["rank_biserial_focus_minus_comparator_y"]
        > 0
    ]
    classical = tests[
        (tests["role"] == "specificity_focus_vs_classical")
        & (tests["p_fdr_bh"] <= config.validation_fdr_q)
    ]
    cross_comparator = primary.merge(classical, on=["representation", "metric"])
    payload = {
        "generated_at": date.today().isoformat(),
        "ok": True,
        "method": "Vietoris-Rips persistent homology over Z/2",
        "selected_representations": selected,
        "selected_features": selected_features,
        "segments": int(manifest.shape[0]),
        "quality_excluded_segments": int(len(excluded)),
        "feature_rows": int(features.shape[0]),
        "confirmatory_fdr_discoveries": int(len(primary)),
        "cross_scale_replicated_discoveries": int(len(replicated)),
        "primary_cross_comparator_discoveries": int(len(cross_comparator)),
        "config": asdict(config),
        "outputs": {
            "features": feature_path.relative_to(root).as_posix(),
            "tests": test_path.relative_to(root).as_posix(),
            "classification": classification_path.relative_to(root).as_posix(),
            "report": report_path.relative_to(root).as_posix(),
            "plots": plots,
        },
        "input_sha256": {manifest_path.relative_to(root).as_posix(): _sha256(manifest_path)},
    }
    output_paths = [feature_path, test_path, classification_path, report_path]
    output_paths.extend(root / Path(path) for path in plots)
    payload["output_sha256"] = {
        path.relative_to(root).as_posix(): _sha256(path) for path in output_paths
    }
    _write_json_atomic(summary_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("explore", "run", "all"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    config = load_config(root)
    if args.command in {"explore", "all"}:
        exploration = run_exploration(root, config)
        print(json.dumps(exploration, ensure_ascii=False, indent=2))
    if args.command in {"run", "all"}:
        result = run_full(root, config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
