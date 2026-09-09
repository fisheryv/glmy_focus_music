#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/ACE-Step-1.5/.venv/bin/python}"
STAGE="${1:-}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/ltsn_turbo_v3}"
PROMPT_MANIFEST="${PROMPT_MANIFEST:-${PROJECT_ROOT}/metadata/ltsn_prompts.csv}"
SURROGATE_TRAINING_GATE="${SURROGATE_TRAINING_GATE:-${PROJECT_ROOT}/metadata/ltsn_surrogate_training_gate.json}"
FINGERPRINT="${PROJECT_ROOT}/metadata/focus_path_homology_fingerprint_v2.json"
CONFIG="${LTSN_CONFIG:-${PROJECT_ROOT}/configs/ltsn_training_v3.toml}"
GUIDANCE_PROTOCOL="${GUIDANCE_PROTOCOL:-${PROJECT_ROOT}/configs/ltsn_guidance_noninferiority_v2.json}"
DEVELOPMENT_DIR="${DEVELOPMENT_DIR:-${RUN_ROOT}/development_pairs}"
TRAINING_AUGMENTATION_DIR="${TRAINING_AUGMENTATION_DIR:-${RUN_ROOT}/training_augmentation}"
ON_POLICY_AUGMENTATION_DIR="${ON_POLICY_AUGMENTATION_DIR:-${RUN_ROOT}/training_augmentation_v4}"
SOURCE_LTSN_MANIFEST="${SOURCE_LTSN_MANIFEST:-${RUN_ROOT}/labels/ltsn_manifest.csv}"
SOURCE_LTSN_SPLIT_MANIFEST="${SOURCE_LTSN_SPLIT_MANIFEST:-${RUN_ROOT}/labels/split_manifest.json}"
LTSN_MANIFEST="${LTSN_MANIFEST:-${TRAINING_AUGMENTATION_DIR}/ltsn_manifest_v3.csv}"
LTSN_SPLIT_MANIFEST="${LTSN_SPLIT_MANIFEST:-${TRAINING_AUGMENTATION_DIR}/split_manifest_v3.json}"
MODEL_DIR="${LTSN_MODEL_DIR:-${RUN_ROOT}/models}"
CALIBRATION_PATH="${LTSN_CALIBRATION_PATH:-${RUN_ROOT}/calibration.json}"
OOD_ABLATION_CALIBRATION_PATH="${LTSN_OOD_ABLATION_CALIBRATION_PATH:-${RUN_ROOT}/calibration_ood_ablation.json}"
GUIDANCE_DEVELOPMENT_PATH="${LTSN_GUIDANCE_DEVELOPMENT_PATH:-${RUN_ROOT}/guidance_development.json}"
QUALIFICATION_PATH="${LTSN_QUALIFICATION_PATH:-${RUN_ROOT}/qualification.json}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python environment is missing: ${PYTHON_BIN}" >&2; exit 2; }

collect() {
  : "${ACE_MODEL_SHA256:?Set ACE_MODEL_SHA256 to the 64-hex model tree digest}"
  : "${VAE_SHA256:?Set VAE_SHA256 to the 64-hex VAE tree digest}"
  local seed_start="${LTSN_SEED_START:-2026071600}"
  local seeds_per_prompt="${LTSN_SEEDS_PER_PROMPT:-4}"
  local -a collect_args=(
    "${PROJECT_ROOT}/scripts/collect_ltsn_trajectories.py"
    --root "${PROJECT_ROOT}"
    --ace-config "${PROJECT_ROOT}/configs/ace_rerank_180s.toml"
    --prompt-manifest "${PROMPT_MANIFEST}"
    --backend ace
    --ace-model-sha256 "${ACE_MODEL_SHA256}"
    --vae-sha256 "${VAE_SHA256}"
    --seed-start "${seed_start}"
    --seeds-per-prompt "${seeds_per_prompt}"
    --duration-seconds 180
    --decode-snapshots
    --discard-generator-final-audio
    --resume
  )
  if [[ -z "${COLLECT_DEVICES:-}" ]]; then
    "${PYTHON_BIN}" "${collect_args[@]}" --output-dir "${RUN_ROOT}/trajectories"
    return
  fi

  local -a collect_devices
  IFS=',' read -r -a collect_devices <<< "${COLLECT_DEVICES}"
  local shard_count="${#collect_devices[@]}"
  (( shard_count > 0 )) || { echo "COLLECT_DEVICES is empty" >&2; return 2; }
  local -A seen_devices=()
  local device
  for device in "${collect_devices[@]}"; do
    [[ "${device}" =~ ^cuda:[0-9]+$ ]] || {
      echo "COLLECT_DEVICES entries must be explicit CUDA devices: ${device}" >&2
      return 2
    }
    [[ -z "${seen_devices[${device}]:-}" ]] || {
      echo "COLLECT_DEVICES contains a duplicate device: ${device}" >&2
      return 2
    }
    seen_devices["${device}"]=1
  done

  local shards_root="${RUN_ROOT}/trajectories/shards"
  local logs_root="${RUN_ROOT}/trajectories/logs"
  mkdir -p "${shards_root}" "${logs_root}"
  local -a pids=()
  local -a logs=()
  stop_collect_workers() {
    local worker_pid
    for worker_pid in "${pids[@]}"; do
      kill -TERM "${worker_pid}" 2>/dev/null || true
    done
  }
  trap 'stop_collect_workers; exit 130' INT TERM
  local index shard_name shard_dir log_path
  for index in "${!collect_devices[@]}"; do
    printf -v shard_name 'shard_%02d' "${index}"
    shard_dir="${shards_root}/${shard_name}"
    log_path="${logs_root}/${shard_name}.log"
    echo "Starting ${shard_name} on ${collect_devices[${index}]} (log: ${log_path})"
    ACESTEP_DEVICE="${collect_devices[${index}]}" \
      "${PYTHON_BIN}" "${collect_args[@]}" \
      --output-dir "${shard_dir}" \
      --shard-index "${index}" \
      --shard-count "${shard_count}" \
      >"${log_path}" 2>&1 &
    pids+=("$!")
    logs+=("${log_path}")
  done

  local failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[${index}]}"; then
      echo "Collection shard ${index} failed; tail of ${logs[${index}]}:" >&2
      tail -n 80 "${logs[${index}]}" >&2
      failed=1
    fi
  done
  trap - INT TERM
  [[ "${failed}" -eq 0 ]] || return 1

  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/merge_ltsn_trajectory_shards.py" \
    --shards-root "${shards_root}" \
    --shard-count "${shard_count}" \
    --prompt-manifest "${PROMPT_MANIFEST}" \
    --output-dir "${RUN_ROOT}/trajectories" \
    --seed-start "${seed_start}" \
    --seeds-per-prompt "${seeds_per_prompt}" \
    --require-audio
}

labels() {
  [[ -f "${SURROGATE_TRAINING_GATE}" ]] || { echo "Passed ltsn_surrogate_training_v1 gate is required: ${SURROGATE_TRAINING_GATE}" >&2; exit 3; }
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_ltsn_labels.py" \
    --root "${PROJECT_ROOT}" \
    --fingerprint "${FINGERPRINT}" \
    --trajectory-manifest "${RUN_ROOT}/trajectories/trajectory_manifest.csv" \
    --work-dir "${RUN_ROOT}/exact_work" \
    --output-dir "${RUN_ROOT}/labels" \
    --surrogate-training-gate "${SURROGATE_TRAINING_GATE}" \
    --workers "${EXACT_WORKERS:-32}" \
    --batch-size "${EXACT_BATCH_SIZE:-256}" \
    --materialize-mode "${MATERIALIZE_MODE:-auto}"
}

augment_training() {
  : "${ACE_MODEL_SHA256:?Set ACE_MODEL_SHA256 to the 64-hex model tree digest}"
  : "${VAE_SHA256:?Set VAE_SHA256 to the 64-hex VAE tree digest}"
  ACESTEP_DEVICE="${AUGMENT_DEVICE:-cuda:0}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_ltsn_training_augmentation.py" \
    --root "${PROJECT_ROOT}" \
    --source-manifest "${SOURCE_LTSN_MANIFEST}" \
    --source-split-manifest "${SOURCE_LTSN_SPLIT_MANIFEST}" \
    --ace-config "${PROJECT_ROOT}/configs/ace_rerank_180s.toml" \
    --fingerprint "${FINGERPRINT}" \
    --output-dir "${TRAINING_AUGMENTATION_DIR}" \
    --ace-model-sha256 "${ACE_MODEL_SHA256}" \
    --vae-sha256 "${VAE_SHA256}" \
    --trajectories-per-prompt "${AUGMENT_TRAJECTORIES_PER_PROMPT:-1}" \
    --perturbations-per-anchor "${AUGMENT_PERTURBATIONS_PER_ANCHOR:-2}" \
    --rms-ratio "${AUGMENT_RMS_RATIO:-0.005}" \
    --ood-per-prompt "${AUGMENT_OOD_PER_PROMPT:-1}" \
    --evaluation-ood-per-prompt "${AUGMENT_EVALUATION_OOD_PER_PROMPT:-1}" \
    --workers "${AUGMENT_EXACT_WORKERS:-8}" \
    --exact-batch-size "${AUGMENT_EXACT_BATCH_SIZE:-256}" \
    --materialize-mode "${MATERIALIZE_MODE:-auto}" \
    --device "${AUGMENT_DEVICE:-cuda:0}" \
    --resume
}

augment_on_policy() {
  : "${ACE_MODEL_SHA256:?Set ACE_MODEL_SHA256 to the 64-hex model tree digest}"
  : "${VAE_SHA256:?Set VAE_SHA256 to the 64-hex VAE tree digest}"
  local ensemble="${ON_POLICY_ENSEMBLE_MANIFEST:-${RUN_ROOT}/models/ensemble_manifest.json}"
  [[ -f "${ensemble}" ]] || {
    echo "A frozen V3 ensemble is required for on-policy augmentation: ${ensemble}" >&2
    exit 3
  }
  ACESTEP_DEVICE="${AUGMENT_DEVICE:-cuda:0}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_ltsn_training_augmentation.py" \
    --root "${PROJECT_ROOT}" \
    --source-manifest "${SOURCE_LTSN_MANIFEST}" \
    --source-split-manifest "${SOURCE_LTSN_SPLIT_MANIFEST}" \
    --ace-config "${PROJECT_ROOT}/configs/ace_rerank_180s.toml" \
    --fingerprint "${FINGERPRINT}" \
    --output-dir "${ON_POLICY_AUGMENTATION_DIR}" \
    --ace-model-sha256 "${ACE_MODEL_SHA256}" \
    --vae-sha256 "${VAE_SHA256}" \
    --local-mode on_policy \
    --on-policy-ensemble-manifest "${ensemble}" \
    --on-policy-rms-ratios 0.0025 0.005 0.01 \
    --trajectories-per-prompt "${AUGMENT_TRAJECTORIES_PER_PROMPT:-1}" \
    --ood-per-prompt "${AUGMENT_OOD_PER_PROMPT:-1}" \
    --evaluation-ood-per-prompt "${AUGMENT_EVALUATION_OOD_PER_PROMPT:-1}" \
    --workers "${AUGMENT_EXACT_WORKERS:-8}" \
    --exact-batch-size "${AUGMENT_EXACT_BATCH_SIZE:-256}" \
    --materialize-mode "${MATERIALIZE_MODE:-auto}" \
    --device "${AUGMENT_DEVICE:-cuda:0}" \
    --resume
}

train() {
  [[ -f "${SURROGATE_TRAINING_GATE}" ]] || { echo "Passed ltsn_surrogate_training_v1 gate is required: ${SURROGATE_TRAINING_GATE}" >&2; exit 3; }
  local -a train_device_args
  if [[ -n "${TRAIN_DEVICES:-}" ]]; then
    local -a train_devices
    IFS=',' read -r -a train_devices <<< "${TRAIN_DEVICES}"
    train_device_args=(--devices "${train_devices[@]}")
  else
    train_device_args=(--device "${TRAIN_DEVICE:-cuda:0}")
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_path_homology_surrogate.py" \
    --fingerprint "${FINGERPRINT}" \
    --manifest "${LTSN_MANIFEST}" \
    --split-manifest "${LTSN_SPLIT_MANIFEST}" \
    --config "${CONFIG}" \
    --output-dir "${MODEL_DIR}" \
    --surrogate-training-gate "${SURROGATE_TRAINING_GATE}" \
    "${train_device_args[@]}"
}

calibrate() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_ltsn_qualification.py" calibrate \
    --fingerprint "${FINGERPRINT}" \
    --manifest "${LTSN_MANIFEST}" \
    --ensemble-manifest "${MODEL_DIR}/ensemble_manifest.json" \
    --output "${CALIBRATION_PATH}" \
    --device "${EVAL_DEVICE:-cuda:0}"
}

calibrate_ood_ablation() {
  [[ -f "${CALIBRATION_PATH}" ]] || {
    echo "Run calibrate before creating the development-only OOD ablation" >&2
    exit 3
  }
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_ltsn_calibration_ablation.py" \
    --calibration "${CALIBRATION_PATH}" \
    --output "${OOD_ABLATION_CALIBRATION_PATH}"
}

development_generate() {
  : "${ACE_MODEL_SHA256:?Set ACE_MODEL_SHA256 to the 64-hex model tree digest}"
  : "${VAE_SHA256:?Set VAE_SHA256 to the 64-hex VAE tree digest}"
  local development_calibration="${CALIBRATION_PATH}"
  local -a corrector_args=(--rms-clip-ratio "${DEVELOPMENT_RMS_CLIP_RATIO:-0.005}")
  local -a correction_steps
  IFS=',' read -r -a correction_steps <<< "${DEVELOPMENT_CORRECTION_STEPS:-4,5,6}"
  (( ${#correction_steps[@]} > 0 )) || { echo "DEVELOPMENT_CORRECTION_STEPS is empty" >&2; exit 2; }
  corrector_args+=(--correction-steps "${correction_steps[@]}")
  if [[ -n "${DEVELOPMENT_DIAGNOSTIC_PROMPT_LIMIT:-}" ]]; then
    corrector_args+=(--diagnostic-prompt-limit "${DEVELOPMENT_DIAGNOSTIC_PROMPT_LIMIT}")
  fi
  if [[ "${DEVELOPMENT_OOD_ABLATION:-0}" == "1" ]]; then
    development_calibration="${OOD_ABLATION_CALIBRATION_PATH}"
    [[ -f "${development_calibration}" ]] || {
      echo "Run calibrate-ood-ablation before development OOD ablation" >&2
      exit 3
    }
    corrector_args+=(--allow-ood-ablation)
  fi
  if [[ "${DEVELOPMENT_REQUIRE_ALL_MEMBERS_OOB:-0}" == "1" ]]; then
    corrector_args+=(--require-all-members-out-of-band)
  fi
  if [[ "${DEVELOPMENT_REQUIRE_ALL_MEMBER_IMPROVEMENT:-0}" == "1" ]]; then
    corrector_args+=(--require-all-member-improvement)
  fi
  if [[ -n "${DEVELOPMENT_MINIMUM_GRADIENT_COSINE:-}" ]]; then
    corrector_args+=(--minimum-member-gradient-cosine "${DEVELOPMENT_MINIMUM_GRADIENT_COSINE}")
  fi
  ACESTEP_DEVICE="${DEVELOPMENT_DEVICE:-cuda:0}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_ltsn_development_pairs.py" generate \
    --root "${PROJECT_ROOT}" \
    --ace-config "${PROJECT_ROOT}/configs/ace_rerank_180s.toml" \
    --prompt-manifest "${PROMPT_MANIFEST}" \
    --fingerprint "${FINGERPRINT}" \
    --ensemble-manifest "${MODEL_DIR}/ensemble_manifest.json" \
    --calibration "${development_calibration}" \
    --output-dir "${DEVELOPMENT_DIR}" \
    --ace-model-sha256 "${ACE_MODEL_SHA256}" \
    --vae-sha256 "${VAE_SHA256}" \
    --seed-start "${LTSN_SEED_START:-2026071600}" \
    --seeds-per-prompt "${DEVELOPMENT_SEEDS_PER_PROMPT:-4}" \
    --expected-development-prompts "${EXPECTED_DEVELOPMENT_PROMPTS:-64}" \
    --duration-seconds "${DEVELOPMENT_DURATION_SECONDS:-180}" \
    --device "${DEVELOPMENT_DEVICE:-cuda:0}" \
    "${corrector_args[@]}" \
    --resume
}

development_step4_diagnostic() {
  DEVELOPMENT_DIR="${STEP4_DIAGNOSTIC_DIR:-${RUN_ROOT}/development_step4_diagnostic}" \
  DEVELOPMENT_OOD_ABLATION=1 \
  DEVELOPMENT_CORRECTION_STEPS=4 \
  DEVELOPMENT_DIAGNOSTIC_PROMPT_LIMIT=16 \
  DEVELOPMENT_RMS_CLIP_RATIO=0.005 \
  DEVELOPMENT_REQUIRE_ALL_MEMBERS_OOB=0 \
  DEVELOPMENT_REQUIRE_ALL_MEMBER_IMPROVEMENT=0 \
  DEVELOPMENT_MINIMUM_GRADIENT_COSINE=-1 \
    development_generate
}

development_step4_score() {
  DEVELOPMENT_DIR="${STEP4_DIAGNOSTIC_DIR:-${RUN_ROOT}/development_step4_diagnostic}" \
    development_score
}

development_step4_report() {
  local diagnostic_dir="${STEP4_DIAGNOSTIC_DIR:-${RUN_ROOT}/development_step4_diagnostic}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_ltsn_step_ablation.py" \
    --pair-table "${diagnostic_dir}/development_pairs_raw.csv" \
    --generation-plan "${diagnostic_dir}/development_generation_plan.json" \
    --output "${diagnostic_dir}/step4_gradient_diagnostic.json" \
    --bootstrap-resamples "${DEVELOPMENT_BOOTSTRAP_RESAMPLES:-2000}"
}

development_score() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_ltsn_development_pairs.py" score \
    --root "${PROJECT_ROOT}" \
    --ace-config "${PROJECT_ROOT}/configs/ace_rerank_180s.toml" \
    --fingerprint "${FINGERPRINT}" \
    --output-dir "${DEVELOPMENT_DIR}" \
    --workers "${DEVELOPMENT_EXACT_WORKERS:-8}"
}

development_evidence() {
  : "${CLAP_REVISION:?Set CLAP_REVISION to the frozen 40-hex Hugging Face commit SHA}"
  local -a evidence_args=(
    "${PROJECT_ROOT}/scripts/build_ltsn_development_noninferiority_metrics.py"
    --root "${PROJECT_ROOT}"
    --run-root "${DEVELOPMENT_DIR}"
    --model-id "${CLAP_MODEL_ID:-laion/clap-htsat-fused}"
    --model-revision "${CLAP_REVISION}"
    --device "${CLAP_DEVICE:-cuda:0}"
    --batch-size "${CLAP_BATCH_SIZE:-8}"
    --segment-seconds "${CLAP_SEGMENT_SECONDS:-10}"
  )
  # Blind quality is intentionally outside the V2 promotion gate. Keep the
  # standard pipeline independent of any stale DEVELOPMENT_QUALITY_TABLE value.
  "${PYTHON_BIN}" "${evidence_args[@]}"
}

development_finalize() {
  local evidence="${NONINFERIORITY_EVIDENCE:-${DEVELOPMENT_DIR}/development_noninferiority_metrics.csv}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_ltsn_development_pairs.py" finalize \
    --raw-pair-table "${DEVELOPMENT_DIR}/development_pairs_raw.csv" \
    --evidence-table "${evidence}" \
    --protocol "${GUIDANCE_PROTOCOL}" \
    --output-dir "${DEVELOPMENT_DIR}" \
    --bootstrap-resamples "${DEVELOPMENT_BOOTSTRAP_RESAMPLES:-2000}"
}

qualify() {
  [[ -f "${GUIDANCE_DEVELOPMENT_PATH}" ]] || {
    echo "Run guidance-development with a decoded exact pair table before qualification" >&2
    exit 3
  }
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_ltsn_qualification.py" qualify \
    --fingerprint "${FINGERPRINT}" \
    --manifest "${LTSN_MANIFEST}" \
    --ensemble-manifest "${MODEL_DIR}/ensemble_manifest.json" \
    --calibration "${CALIBRATION_PATH}" \
    --guidance-development-report "${GUIDANCE_DEVELOPMENT_PATH}" \
    --output "${QUALIFICATION_PATH}" \
    --device "${EVAL_DEVICE:-cuda:0}"
}

guidance_development() {
  local pair_table="${PAIR_TABLE:-${DEVELOPMENT_DIR}/development_pairs.csv}"
  [[ -f "${pair_table}" ]] || {
    echo "Finalized development-only pair table is required: ${pair_table}" >&2
    exit 3
  }
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_path_homology_guidance.py" \
    --fingerprint "${FINGERPRINT}" \
    --pair-table "${pair_table}" \
    --output "${GUIDANCE_DEVELOPMENT_PATH}" \
    --mode development
}

guidance_confirmation() {
  : "${PAIR_TABLE:?Set PAIR_TABLE to the fresh 32-prompt x 8-seed confirmation pair CSV}"
  [[ -f "${QUALIFICATION_PATH}" ]] || {
    echo "Passed independent qualification report is required: ${QUALIFICATION_PATH}" >&2
    exit 3
  }
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_path_homology_guidance.py" \
    --fingerprint "${FINGERPRINT}" \
    --pair-table "${PAIR_TABLE}" \
    --output "${RUN_ROOT}/guidance_confirmation.json" \
    --qualification-report "${QUALIFICATION_PATH}" \
    --mode confirmation
}

case "${STAGE}" in
  collect) collect ;;
  labels) labels ;;
  augment-training) augment_training ;;
  augment-on-policy) augment_on_policy ;;
  train) train ;;
  calibrate) calibrate ;;
  calibrate-ood-ablation) calibrate_ood_ablation ;;
  development-generate) development_generate ;;
  development-step4-diagnostic) development_step4_diagnostic ;;
  development-step4-score) development_step4_score ;;
  development-step4-report) development_step4_report ;;
  development-score) development_score ;;
  development-evidence) development_evidence ;;
  development-finalize) development_finalize ;;
  guidance-development) guidance_development ;;
  qualify) qualify ;;
  guidance-confirmation) guidance_confirmation ;;
  *) echo "Usage: $0 {collect|labels|augment-training|augment-on-policy|train|calibrate|calibrate-ood-ablation|development-generate|development-step4-diagnostic|development-step4-score|development-step4-report|development-score|development-evidence|development-finalize|guidance-development|qualify|guidance-confirmation}" >&2; exit 2 ;;
esac
