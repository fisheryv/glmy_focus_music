#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/ACE-Step-1.5/.venv/bin/python}"
STAGE="${1:-}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/ltsn_turbo}"
PROMPT_MANIFEST="${PROMPT_MANIFEST:-${PROJECT_ROOT}/metadata/ltsn_prompts.csv}"
SURROGATE_TRAINING_GATE="${SURROGATE_TRAINING_GATE:-${PROJECT_ROOT}/metadata/ltsn_surrogate_training_gate.json}"
FINGERPRINT="${PROJECT_ROOT}/metadata/focus_path_homology_fingerprint_v2.json"
CONFIG="${PROJECT_ROOT}/configs/ltsn_training.toml"

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
    --manifest "${RUN_ROOT}/labels/ltsn_manifest.csv" \
    --split-manifest "${RUN_ROOT}/labels/split_manifest.json" \
    --config "${CONFIG}" \
    --output-dir "${RUN_ROOT}/models" \
    --surrogate-training-gate "${SURROGATE_TRAINING_GATE}" \
    "${train_device_args[@]}"
}

calibrate() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_ltsn_qualification.py" calibrate \
    --fingerprint "${FINGERPRINT}" \
    --manifest "${RUN_ROOT}/labels/ltsn_manifest.csv" \
    --ensemble-manifest "${RUN_ROOT}/models/ensemble_manifest.json" \
    --output "${RUN_ROOT}/calibration.json" \
    --device "${EVAL_DEVICE:-cuda:0}"
}

qualify() {
  [[ -f "${RUN_ROOT}/guidance_development.json" ]] || {
    echo "Run guidance-development with a decoded exact pair table before qualification" >&2
    exit 3
  }
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_ltsn_qualification.py" qualify \
    --fingerprint "${FINGERPRINT}" \
    --manifest "${RUN_ROOT}/labels/ltsn_manifest.csv" \
    --ensemble-manifest "${RUN_ROOT}/models/ensemble_manifest.json" \
    --calibration "${RUN_ROOT}/calibration.json" \
    --guidance-development-report "${RUN_ROOT}/guidance_development.json" \
    --output "${RUN_ROOT}/qualification.json" \
    --device "${EVAL_DEVICE:-cuda:0}"
}

guidance_development() {
  : "${PAIR_TABLE:?Set PAIR_TABLE to the decoded exact/proxy development pair CSV}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_path_homology_guidance.py" \
    --fingerprint "${FINGERPRINT}" \
    --pair-table "${PAIR_TABLE}" \
    --output "${RUN_ROOT}/guidance_development.json" \
    --mode development
}

guidance_confirmation() {
  : "${PAIR_TABLE:?Set PAIR_TABLE to the fresh 32-prompt x 8-seed confirmation pair CSV}"
  [[ -f "${RUN_ROOT}/qualification.json" ]] || {
    echo "Passed independent qualification report is required: ${RUN_ROOT}/qualification.json" >&2
    exit 3
  }
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_path_homology_guidance.py" \
    --fingerprint "${FINGERPRINT}" \
    --pair-table "${PAIR_TABLE}" \
    --output "${RUN_ROOT}/guidance_confirmation.json" \
    --qualification-report "${RUN_ROOT}/qualification.json" \
    --mode confirmation
}

case "${STAGE}" in
  collect) collect ;;
  labels) labels ;;
  train) train ;;
  calibrate) calibrate ;;
  guidance-development) guidance_development ;;
  qualify) qualify ;;
  guidance-confirmation) guidance_confirmation ;;
  *) echo "Usage: $0 {collect|labels|train|calibrate|guidance-development|qualify|guidance-confirmation}" >&2; exit 2 ;;
esac
