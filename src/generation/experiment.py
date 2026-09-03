from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ACE_MODEL_NAME = "acestep-v15-xl-turbo"
ACE_MODEL_REPOSITORY = "ACE-Step/acestep-v15-xl-turbo"


class ExperimentConfigError(ValueError):
    """Raised when an ACE reranking experiment configuration is invalid."""


@dataclass(frozen=True, slots=True)
class PromptSpec:
    prompt_id: str
    caption: str
    bpm: int | None = None
    keyscale: str = ""
    timesignature: str = ""


@dataclass(frozen=True, slots=True)
class AceConfig:
    checkout: str = "ACE-Step-1.5"
    model: str = ACE_MODEL_NAME
    model_repository: str = ACE_MODEL_REPOSITORY
    device: str = "cuda"
    inference_steps: int = 8
    shift: float = 3.0
    infer_method: str = "ode"
    sampler_mode: str = "euler"
    guidance_scale: float = 1.0
    compile_model: bool = False
    offload_to_cpu: bool = False
    offload_dit_to_cpu: bool = False
    dcw_enabled: bool = True
    dcw_mode: str = "double"
    dcw_scaler: float = 0.05
    dcw_high_scaler: float = 0.02
    dcw_wavelet: str = "haar"
    quantization: str | None = None
    vae_checkpoint: str | None = None
    prefer_source: str | None = None


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    target_group: str = "focus"
    target_split: str = "discovery"
    target_scale_seconds: float = 180.0
    covariance_shrinkage: float = 0.2
    technical_quality_weight: float = 0.0
    success_min_median_improvement: float = 0.10
    minimum_prompt_pools: int = 20
    permutations: int = 999
    bootstrap_resamples: int = 1000


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    run_id: str
    prompt_manifest: str
    candidate_count: int = 8
    seed_start: int = 2026071600
    duration_seconds: float = 180.0
    workers: int = 2
    save_latents: bool = True
    run_directory: str = "runs/ace_rerank"
    ace: AceConfig = field(default_factory=AceConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    def validate(self) -> None:
        if not _SAFE_ID.fullmatch(self.run_id):
            raise ExperimentConfigError("run_id must contain only letters, digits, '.', '_' or '-'")
        if self.candidate_count < 2:
            raise ExperimentConfigError("candidate_count must be at least 2")
        if self.duration_seconds < 10:
            raise ExperimentConfigError("duration_seconds must be at least 10")
        if self.workers < 1:
            raise ExperimentConfigError("workers must be positive")
        if self.ace.inference_steps < 1 or self.ace.shift <= 0:
            raise ExperimentConfigError("ACE inference_steps and shift must be positive")
        if (
            self.ace.model != ACE_MODEL_NAME
            or self.ace.model_repository != ACE_MODEL_REPOSITORY
        ):
            raise ExperimentConfigError(
                f"ACE model must be pinned to {ACE_MODEL_REPOSITORY} "
                f"with runtime name {ACE_MODEL_NAME}"
            )
        if self.ace.infer_method not in {"ode", "sde"}:
            raise ExperimentConfigError("ACE infer_method must be 'ode' or 'sde'")
        if self.ace.sampler_mode not in {"euler", "heun"}:
            raise ExperimentConfigError("ACE sampler_mode must be 'euler' or 'heun'")
        if not 0 <= self.scoring.covariance_shrinkage <= 1:
            raise ExperimentConfigError("covariance_shrinkage must lie in [0, 1]")
        if self.scoring.minimum_prompt_pools < 2:
            raise ExperimentConfigError("minimum_prompt_pools must be at least 2")
        if self.scoring.permutations < 99 or self.scoring.bootstrap_resamples < 100:
            raise ExperimentConfigError("statistical resample counts are too small")


@dataclass(slots=True)
class CandidateRecord:
    experiment_id: str
    prompt_id: str
    caption: str
    candidate_index: int
    candidate_id: str
    seed: int
    duration_seconds: float
    bpm: int | None = None
    keyscale: str = ""
    timesignature: str = ""
    status: str = "planned"
    audio_relative_path: str = ""
    audio_sha256: str = ""
    latent_relative_path: str = ""
    latent_sha256: str = ""
    metadata_relative_path: str = ""
    generated_at: str = ""
    error: str = ""


CANDIDATE_COLUMNS = tuple(CandidateRecord.__dataclass_fields__)


def _section(cls: type[Any], raw: dict[str, Any], name: str) -> Any:
    unknown = set(raw) - set(cls.__dataclass_fields__)
    if unknown:
        raise ExperimentConfigError(f"unknown [{name}] keys: {sorted(unknown)}")
    return cls(**raw)


def load_experiment_config(
    root: Path, path: Path, *, run_id: str | None = None
) -> ExperimentConfig:
    resolved = path if path.is_absolute() else root / path
    with resolved.open("rb") as handle:
        raw = tomllib.load(handle)
    experiment = dict(raw.get("experiment", {}))
    if run_id is not None:
        experiment["run_id"] = run_id
    config = ExperimentConfig(
        **experiment,
        ace=_section(AceConfig, dict(raw.get("ace", {})), "ace"),
        scoring=_section(ScoringConfig, dict(raw.get("scoring", {})), "scoring"),
    )
    config.validate()
    return config


def read_prompts(root: Path, relative_path: str) -> list[PromptSpec]:
    path = root / relative_path
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    prompts: list[PromptSpec] = []
    for row in rows:
        prompt_id = (row.get("prompt_id") or "").strip()
        caption = (row.get("caption") or "").strip()
        if not _SAFE_ID.fullmatch(prompt_id) or not caption:
            raise ExperimentConfigError(f"invalid prompt row: {row}")
        bpm_raw = (row.get("bpm") or "").strip()
        prompts.append(
            PromptSpec(
                prompt_id=prompt_id,
                caption=caption,
                bpm=int(bpm_raw) if bpm_raw else None,
                keyscale=(row.get("keyscale") or "").strip(),
                timesignature=(row.get("timesignature") or "").strip(),
            )
        )
    if not prompts or len({item.prompt_id for item in prompts}) != len(prompts):
        raise ExperimentConfigError("prompt manifest is empty or has duplicate prompt_id values")
    return prompts


def build_candidate_plan(
    config: ExperimentConfig, prompts: list[PromptSpec]
) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    for prompt_index, prompt in enumerate(prompts):
        for candidate_index in range(config.candidate_count):
            seed = config.seed_start + prompt_index * config.candidate_count + candidate_index
            candidate_id = f"{prompt.prompt_id}__c{candidate_index:02d}__s{seed}"
            records.append(
                CandidateRecord(
                    experiment_id=config.run_id,
                    prompt_id=prompt.prompt_id,
                    caption=prompt.caption,
                    candidate_index=candidate_index,
                    candidate_id=candidate_id,
                    seed=seed,
                    duration_seconds=config.duration_seconds,
                    bpm=prompt.bpm,
                    keyscale=prompt.keyscale,
                    timesignature=prompt.timesignature,
                )
            )
    return records


def write_candidate_manifest(path: Path, records: list[CandidateRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    os.replace(temporary, path)


def read_candidate_manifest(path: Path) -> list[CandidateRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = []
    for row in rows:
        row["candidate_index"] = int(row["candidate_index"])
        row["seed"] = int(row["seed"])
        row["duration_seconds"] = float(row["duration_seconds"])
        row["bpm"] = int(row["bpm"]) if row.get("bpm") else None
        records.append(CandidateRecord(**row))
    return records


def config_fingerprint(config: ExperimentConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
