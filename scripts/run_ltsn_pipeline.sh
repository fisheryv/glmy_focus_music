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
V5_AUGMENTATION_DIR="${V5_AUGMENTATION_DIR:-${RUN_ROOT}/training_augmentation_v5}"
V5_MODEL_DIR="${V5_MODEL_DIR:-${RUN_ROOT}/models_v5}"
V5_CALIBRATION_PATH="${V5_CALIBRATION_PATH:-${RUN_ROOT}/calibration_v5.json}"
V5_OOD_ABLATION_CALIBRATION_PATH="${V5_OOD_ABLATION_CALIBRATION_PATH:-${RUN_ROOT}/calibration_v5_ood_ablation.json}"
V5_DEVELOPMENT_DIR="${V5_DEVELOPMENT_DIR:-${RUN_ROOT}/development_pairs_v5}"
V5_GUIDANCE_DEVELOPMENT_PATH="${V5_GUIDANCE_DEVELOPMENT_PATH:-${RUN_ROOT}/guidance_development_v5.json}"
V5_QUALIFICATION_PATH="${V5_QUALIFICATION_PATH:-${RUN_ROOT}/qualification_v5.json}"
V51_TRAINING_VIEW_DIR="${V51_TRAINING_VIEW_DIR:-${RUN_ROOT}/training_augmentation_v51}"
V51_MODEL_DIR="${V51_MODEL_DIR:-${RUN_ROOT}/models_v51}"
V51_SCREEN_MODEL_DIR="${V51_SCREEN_MODEL_DIR:-${RUN_ROOT}/models_v51_screen}"
V51_SCREEN_REPORT="${V51_SCREEN_REPORT:-${RUN_ROOT}/v51_screen_report.json}"
V51A_DIAGNOSTIC_VIEW_DIR="${V51A_DIAGNOSTIC_VIEW_DIR:-${RUN_ROOT}/training_augmentation_v51a}"
V51A_MODEL_DIR="${V51A_MODEL_DIR:-${RUN_ROOT}/models_v51a_overfit}"
V51A_REPORT="${V51A_REPORT:-${RUN_ROOT}/v51a_overfit_report.json}"
V51B_LADDER_DIR="${V51B_LADDER_DIR:-${RUN_ROOT}/training_augmentation_v51b}"
V51B_MODEL_ROOT="${V51B_MODEL_ROOT:-${RUN_ROOT}/models_v51b}"
V51B_REPORT="${V51B_REPORT:-${RUN_ROOT}/v51b_memorization_report.json}"
V51B_CONFIG="${V51B_CONFIG:-${PROJECT_ROOT}/configs/ltsn_training_v51b_memorization.toml}"
V51C_SUITE_DIR="${V51C_SUITE_DIR:-${RUN_ROOT}/training_augmentation_v51c}"
V51C_MODEL_ROOT="${V51C_MODEL_ROOT:-${RUN_ROOT}/models_v51c}"
V51C_REPORT="${V51C_REPORT:-${RUN_ROOT}/v51c_ablation_report.json}"
V52A_DIR="${V52A_DIR:-${RUN_ROOT}/training_augmentation_v52a}"
V52A_MODEL_ROOT="${V52A_MODEL_ROOT:-${RUN_ROOT}/models_v52a}"
V52A_REPORT="${V52A_REPORT:-${RUN_ROOT}/v52a_identifiability_report.json}"
V52A_CONFIG="${V52A_CONFIG:-${PROJECT_ROOT}/configs/ltsn_training_v52a_identifiability.toml}"
V52B_DIR="${V52B_DIR:-${RUN_ROOT}/training_augmentation_v52b}"
V52B_MODEL_ROOT="${V52B_MODEL_ROOT:-${RUN_ROOT}/models_v52b}"
V52B_REPORT="${V52B_REPORT:-${RUN_ROOT}/v52b_probe_report.json}"
V52B_CONFIG="${V52B_CONFIG:-${PROJECT_ROOT}/configs/ltsn_training_v52b_probe.toml}"
TAC_TARGET="${TAC_TARGET:-${PROJECT_ROOT}/metadata/tac_topology_target_v1.json}"
V52C_DIR="${V52C_DIR:-${RUN_ROOT}/tac_v52c}"
V52C_REPORT="${V52C_REPORT:-${V52C_DIR}/v52c_response_report.json}"
V52D_DIR="${V52D_DIR:-${RUN_ROOT}/tac_v52d}"
V52D_REPEAT_DIR="${V52D_REPEAT_DIR:-${V52D_DIR}/repeatability}"
V52D_ACTION_DIR="${V52D_ACTION_DIR:-${V52D_DIR}/actions}"
V52D_REPEAT_REPORT="${V52D_REPEAT_REPORT:-${V52D_REPEAT_DIR}/v52d_repeatability_report.json}"
V52D_ACTION_REPORT="${V52D_ACTION_REPORT:-${V52D_ACTION_DIR}/v52d_action_report.json}"
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

augment_v5() {
  : "${ACE_MODEL_SHA256:?Set ACE_MODEL_SHA256 to the 64-hex model tree digest}"
  : "${VAE_SHA256:?Set VAE_SHA256 to the 64-hex VAE tree digest}"
  local ensemble="${V5_DIRECTION_ENSEMBLE_MANIFEST:-${RUN_ROOT}/models/ensemble_manifest.json}"
  [[ -f "${ensemble}" ]] || {
    echo "A frozen V4 ensemble is required for V5 direction generation: ${ensemble}" >&2
    exit 3
  }
  ACESTEP_DEVICE="${AUGMENT_DEVICE:-cuda:0}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_ltsn_training_augmentation.py" \
    --root "${PROJECT_ROOT}" \
    --source-manifest "${SOURCE_LTSN_MANIFEST}" \
    --source-split-manifest "${SOURCE_LTSN_SPLIT_MANIFEST}" \
    --ace-config "${PROJECT_ROOT}/configs/ace_rerank_180s.toml" \
    --fingerprint "${FINGERPRINT}" \
    --output-dir "${V5_AUGMENTATION_DIR}" \
    --ace-model-sha256 "${ACE_MODEL_SHA256}" \
    --vae-sha256 "${VAE_SHA256}" \
    --local-mode symmetric_on_policy \
    --on-policy-ensemble-manifest "${ensemble}" \
    --on-policy-rms-ratios 0.0025 0.005 \
    --trajectories-per-prompt "${V5_TRAJECTORIES_PER_PROMPT:-4}" \
    --v5-train-anchor-count "${V5_TRAIN_ANCHORS:-512}" \
    --v5-development-anchor-count "${V5_DEVELOPMENT_ANCHORS:-128}" \
    --ood-per-prompt "${AUGMENT_OOD_PER_PROMPT:-1}" \
    --evaluation-ood-per-prompt "${AUGMENT_EVALUATION_OOD_PER_PROMPT:-1}" \
    --workers "${AUGMENT_EXACT_WORKERS:-8}" \
    --exact-batch-size "${AUGMENT_EXACT_BATCH_SIZE:-256}" \
    --materialize-mode "${MATERIALIZE_MODE:-auto}" \
    --device "${AUGMENT_DEVICE:-cuda:0}" \
    --resume
}

train_v5() {
  LTSN_MANIFEST="${V5_AUGMENTATION_DIR}/ltsn_manifest_v5.csv" \
  LTSN_SPLIT_MANIFEST="${V5_AUGMENTATION_DIR}/split_manifest_v5.json" \
  CONFIG="${V5_CONFIG:-${PROJECT_ROOT}/configs/ltsn_training_v5.toml}" \
  MODEL_DIR="${V5_MODEL_DIR}" \
    train
}

train_v5_screen() {
  V5_CONFIG="${PROJECT_ROOT}/configs/ltsn_training_v5_screen.toml" \
  V5_MODEL_DIR="${V5_SCREEN_MODEL_DIR:-${RUN_ROOT}/models_v5_screen}" \
  TRAIN_ENGINEERING_SMOKE=1 \
  TRAIN_DEVICES= \
    train_v5
}

prepare_v51() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_ltsn_v51_training_view.py" \
    --source-manifest "${V5_AUGMENTATION_DIR}/ltsn_manifest_v5.csv" \
    --source-split-manifest "${V5_AUGMENTATION_DIR}/split_manifest_v5.json" \
    --central-evidence "${V5_AUGMENTATION_DIR}/central_direction_exact_evidence.csv" \
    --output-dir "${V51_TRAINING_VIEW_DIR}" \
    --minimum-abs-loss-separation "${V51_MINIMUM_ABS_LOSS_SEPARATION:-0.00001}"
}

train_v51_screen() {
  LTSN_MANIFEST="${V51_TRAINING_VIEW_DIR}/ltsn_manifest_v51.csv" \
  LTSN_SPLIT_MANIFEST="${V51_TRAINING_VIEW_DIR}/split_manifest_v51.json" \
  CONFIG="${PROJECT_ROOT}/configs/ltsn_training_v51_screen.toml" \
  MODEL_DIR="${V51_SCREEN_MODEL_DIR}" \
  TRAIN_ENGINEERING_SMOKE=1 \
  TRAIN_DEVICES= \
    train
}

report_v51_screen() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_ltsn_v51_screen.py" \
    --ensemble-manifest "${V51_SCREEN_MODEL_DIR}/ensemble_manifest.json" \
    --training-view-summary "${V51_TRAINING_VIEW_DIR}/training_view_summary.json" \
    --output "${V51_SCREEN_REPORT}"
}

train_v51() {
  [[ -f "${V51_SCREEN_REPORT}" ]] || {
    echo "Run report-v51-screen before formal V5.1 training: ${V51_SCREEN_REPORT}" >&2
    exit 3
  }
  "${PYTHON_BIN}" -c \
    'import hashlib,json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); d=hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest(); raise SystemExit(0 if p.get("formal_v51_training_recommended") is True and p.get("training_view_summary_sha256") == d else 3)' \
    "${V51_SCREEN_REPORT}" "${V51_TRAINING_VIEW_DIR}/training_view_summary.json" || {
      echo "V5.1 screen failed or is not bound to the current training view; inspect ${V51_SCREEN_REPORT}" >&2
      exit 3
    }
  LTSN_MANIFEST="${V51_TRAINING_VIEW_DIR}/ltsn_manifest_v51.csv" \
  LTSN_SPLIT_MANIFEST="${V51_TRAINING_VIEW_DIR}/split_manifest_v51.json" \
  CONFIG="${PROJECT_ROOT}/configs/ltsn_training_v51.toml" \
  MODEL_DIR="${V51_MODEL_DIR}" \
    train
}

prepare_v51a() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_ltsn_v51a_overfit_view.py" \
    --source-manifest "${V51_TRAINING_VIEW_DIR}/ltsn_manifest_v51.csv" \
    --source-split-manifest "${V51_TRAINING_VIEW_DIR}/split_manifest_v51.json" \
    --central-evidence "${V51_TRAINING_VIEW_DIR}/central_direction_exact_evidence_v51.csv" \
    --output-dir "${V51A_DIAGNOSTIC_VIEW_DIR}" \
    --step4-anchors "${V51A_STEP4_ANCHORS:-32}" \
    --step5-anchors "${V51A_STEP5_ANCHORS:-16}" \
    --step6-anchors "${V51A_STEP6_ANCHORS:-16}"
}

train_v51a() {
  LTSN_MANIFEST="${V51A_DIAGNOSTIC_VIEW_DIR}/ltsn_manifest_v51a.csv" \
  LTSN_SPLIT_MANIFEST="${V51A_DIAGNOSTIC_VIEW_DIR}/split_manifest_v51a.json" \
  CONFIG="${PROJECT_ROOT}/configs/ltsn_training_v51a_overfit.toml" \
  MODEL_DIR="${V51A_MODEL_DIR}" \
  TRAIN_ENGINEERING_SMOKE=1 \
  TRAIN_DEVICES= \
    train
}

report_v51a() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_ltsn_v51a_overfit.py" \
    --ensemble-manifest "${V51A_MODEL_DIR}/ensemble_manifest.json" \
    --diagnostic-view-summary "${V51A_DIAGNOSTIC_VIEW_DIR}/diagnostic_view_summary.json" \
    --output "${V51A_REPORT}"
}

prepare_v51b() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_ltsn_v51b_memorization_ladder.py" \
    --source-manifest "${V51A_DIAGNOSTIC_VIEW_DIR}/ltsn_manifest_v51a.csv" \
    --source-split-manifest "${V51A_DIAGNOSTIC_VIEW_DIR}/split_manifest_v51a.json" \
    --central-evidence "${V51A_DIAGNOSTIC_VIEW_DIR}/central_direction_exact_evidence_v51a.csv" \
    --output-root "${V51B_LADDER_DIR}" \
    --rungs 1 4 16 64 \
    --development-anchors "${V51B_DEVELOPMENT_ANCHORS:-6}"
}

train_v51b() {
  local -a rung_values
  IFS=',' read -r -a rung_values <<< "${V51B_RUNGS:-1,4,16,64}"
  (( ${#rung_values[@]} > 0 )) || { echo "V51B_RUNGS is empty" >&2; return 2; }
  local rung rung_name
  for rung in "${rung_values[@]}"; do
    case "${rung}" in
      1|4|16|64) ;;
      *) echo "V51B_RUNGS entries must be one of 1,4,16,64: ${rung}" >&2; return 2 ;;
    esac
    printf -v rung_name 'rung_%03d' "${rung}"
    [[ -f "${V51B_LADDER_DIR}/${rung_name}/ltsn_manifest_v51b.csv" ]] || {
      echo "Run prepare-v51b first; missing ${rung_name}" >&2
      return 3
    }
    LTSN_MANIFEST="${V51B_LADDER_DIR}/${rung_name}/ltsn_manifest_v51b.csv" \
    LTSN_SPLIT_MANIFEST="${V51B_LADDER_DIR}/${rung_name}/split_manifest_v51b.json" \
    CONFIG="${V51B_CONFIG}" \
    MODEL_DIR="${V51B_MODEL_ROOT}/${rung_name}" \
    TRAIN_ENGINEERING_SMOKE=1 \
    TRAIN_DEVICES= \
    TRAIN_DEVICE="${V51B_DEVICE:-cuda:1}" \
      train
  done
}

report_v51b() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_ltsn_v51b_memorization.py" \
    --ladder-summary "${V51B_LADDER_DIR}/memorization_ladder.json" \
    --models-root "${V51B_MODEL_ROOT}" \
    --output "${V51B_REPORT}" \
    --maximum-train-loss "${V51B_MAXIMUM_TRAIN_LOSS:-0.1}"
}

prepare_v51c() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_ltsn_v51c_ablation_suite.py" \
    --source-manifest "${V51A_DIAGNOSTIC_VIEW_DIR}/ltsn_manifest_v51a.csv" \
    --source-split-manifest "${V51A_DIAGNOSTIC_VIEW_DIR}/split_manifest_v51a.json" \
    --central-evidence "${V51A_DIAGNOSTIC_VIEW_DIR}/central_direction_exact_evidence_v51a.csv" \
    --config-root "${PROJECT_ROOT}/configs" \
    --output-root "${V51C_SUITE_DIR}"
}

train_v51c() {
  local -a variants
  IFS=',' read -r -a variants <<< "${V51C_VARIANTS:-baseline,restore_dropout,restore_weight_decay,restore_lr_schedule,restore_batch32,restore_short_early_stop,permuted_pair_control}"
  (( ${#variants[@]} > 0 )) || { echo "V51C_VARIANTS is empty" >&2; return 2; }
  local variant config_name view_name
  for variant in "${variants[@]}"; do
    case "${variant}" in
      baseline)
        config_name="ltsn_training_v51c_baseline.toml"
        view_name="true_pairs"
        ;;
      restore_dropout)
        config_name="ltsn_training_v51c_restore_dropout.toml"
        view_name="true_pairs"
        ;;
      restore_weight_decay)
        config_name="ltsn_training_v51c_restore_weight_decay.toml"
        view_name="true_pairs"
        ;;
      restore_lr_schedule)
        config_name="ltsn_training_v51c_restore_lr_schedule.toml"
        view_name="true_pairs"
        ;;
      restore_batch32)
        config_name="ltsn_training_v51c_restore_batch32.toml"
        view_name="true_pairs"
        ;;
      restore_short_early_stop)
        config_name="ltsn_training_v51c_restore_short_early_stop.toml"
        view_name="true_pairs"
        ;;
      permuted_pair_control)
        config_name="ltsn_training_v51c_permuted_pair_control.toml"
        view_name="permuted_pair_control"
        ;;
      *) echo "Unknown V5.1c variant: ${variant}" >&2; return 2 ;;
    esac
    [[ -f "${V51C_SUITE_DIR}/${view_name}/ltsn_manifest_v51c.csv" ]] || {
      echo "Run prepare-v51c first; missing ${view_name} view" >&2
      return 3
    }
    LTSN_MANIFEST="${V51C_SUITE_DIR}/${view_name}/ltsn_manifest_v51c.csv" \
    LTSN_SPLIT_MANIFEST="${V51C_SUITE_DIR}/${view_name}/split_manifest_v51c.json" \
    CONFIG="${PROJECT_ROOT}/configs/${config_name}" \
    MODEL_DIR="${V51C_MODEL_ROOT}/${variant}" \
    TRAIN_ENGINEERING_SMOKE=1 \
    TRAIN_DEVICES= \
    TRAIN_DEVICE="${V51C_DEVICE:-cuda:1}" \
      train
  done
}

report_v51c() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_ltsn_v51c_ablation.py" \
    --suite-summary "${V51C_SUITE_DIR}/ablation_suite.json" \
    --models-root "${V51C_MODEL_ROOT}" \
    --output "${V51C_REPORT}"
}

collect_v52a() {
  : "${ACE_MODEL_SHA256:?Set ACE_MODEL_SHA256 to the 64-hex model tree digest}"
  : "${VAE_SHA256:?Set VAE_SHA256 to the 64-hex VAE tree digest}"
  ACESTEP_DEVICE="${V52A_COLLECT_DEVICE:-cuda:0}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/collect_ltsn_v52a_multidirection.py" \
    --root "${PROJECT_ROOT}" \
    --source-manifest "${V51A_DIAGNOSTIC_VIEW_DIR}/ltsn_manifest_v51a.csv" \
    --source-split-manifest "${V51A_DIAGNOSTIC_VIEW_DIR}/split_manifest_v51a.json" \
    --source-evidence "${V51A_DIAGNOSTIC_VIEW_DIR}/central_direction_exact_evidence_v51a.csv" \
    --ace-config "${PROJECT_ROOT}/configs/ace_rerank_180s.toml" \
    --fingerprint "${FINGERPRINT}" \
    --output-dir "${V52A_DIR}" \
    --ace-model-sha256 "${ACE_MODEL_SHA256}" \
    --vae-sha256 "${VAE_SHA256}" \
    --train-anchors "${V52A_TRAIN_ANCHORS:-32}" \
    --unseen-anchors "${V52A_UNSEEN_ANCHORS:-16}" \
    --directions 8 \
    --train-directions 6 \
    --workers "${V52A_EXACT_WORKERS:-8}" \
    --exact-batch-size "${V52A_EXACT_BATCH_SIZE:-256}" \
    --materialize-mode "${MATERIALIZE_MODE:-auto}" \
    --device "${V52A_COLLECT_DEVICE:-cuda:0}" \
    --resume
}

train_v52a_one() {
  local view_name="$1"
  local model_name="$2"
  TRAIN_DEVICES="${V52A_DEVICES:-}" \
  TRAIN_DEVICE="${V52A_DEVICE:-cuda:0}" \
  LTSN_MANIFEST="${V52A_DIR}/views/${view_name}/ltsn_manifest_v52a.csv" \
  LTSN_SPLIT_MANIFEST="${V52A_DIR}/views/${view_name}/split_manifest_v52a.json" \
  CONFIG="${V52A_CONFIG}" \
  MODEL_DIR="${V52A_MODEL_ROOT}/${model_name}" \
  TRAIN_ENGINEERING_SMOKE=1 \
    train
}

train_v52a() {
  [[ -f "${V52A_DIR}/views/v52a_views.json" ]] || {
    echo "Run collect-v52a first" >&2
    return 3
  }
  train_v52a_one true_pairs true_pairs
  train_v52a_one permuted_pair_control permuted_pair_control
}

report_v52a() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_ltsn_v52a_identifiability.py" \
    --fingerprint "${FINGERPRINT}" \
    --views-summary "${V52A_DIR}/views/v52a_views.json" \
    --true-ensemble "${V52A_MODEL_ROOT}/true_pairs/ensemble_manifest.json" \
    --control-ensemble "${V52A_MODEL_ROOT}/permuted_pair_control/ensemble_manifest.json" \
    --config "${V52A_CONFIG}" \
    --output "${V52A_REPORT}" \
    --device "${V52A_EVAL_DEVICE:-cuda:0}"
}

prepare_v52b() {
  [[ -f "${V52A_REPORT}" ]] || {
    echo "Run report-v52a first" >&2
    return 3
  }
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/prepare_ltsn_v52b.py" \
    --master-manifest "${V52A_DIR}/ltsn_manifest_v52a_master.csv" \
    --evidence "${V52A_DIR}/central_direction_exact_evidence_v52a.csv" \
    --views-summary "${V52A_DIR}/views/v52a_views.json" \
    --v52a-report "${V52A_REPORT}" \
    --output-dir "${V52B_DIR}" \
    --operational-rms 0.005 \
    --control-seed 20260911
}

train_v52b_one() {
  local variant="$1"
  local target_mode="$2"
  if [[ -f "${V52B_MODEL_ROOT}/${variant}/${target_mode}/ensemble_manifest.json" ]]; then
    echo "V5.2b ${variant}/${target_mode} ensemble already exists; skipping"
    return 0
  fi
  local -a device_args
  if [[ -n "${V52B_DEVICES:-}" ]]; then
    local -a devices
    IFS=',' read -r -a devices <<< "${V52B_DEVICES}"
    device_args=(--devices "${devices[@]}")
  else
    device_args=(--device "${V52B_DEVICE:-cuda:0}")
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_ltsn_v52b_probe.py" \
    --fingerprint "${FINGERPRINT}" \
    --pair-manifest "${V52B_DIR}/v52b_pair_manifest.csv" \
    --preparation "${V52B_DIR}/v52b_preparation.json" \
    --config "${V52B_CONFIG}" \
    --output-dir "${V52B_MODEL_ROOT}/${variant}/${target_mode}" \
    --variant "${variant}" \
    --target-mode "${target_mode}" \
    "${device_args[@]}"
}

train_v52b() {
  [[ -f "${V52B_DIR}/v52b_preparation.json" ]] || {
    echo "Run prepare-v52b first" >&2
    return 3
  }
  local variant target_mode
  for variant in scalar pair direction_field; do
    for target_mode in true matched_control; do
      train_v52b_one "${variant}" "${target_mode}"
    done
  done
}

report_v52b() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_ltsn_v52b.py" \
    --fingerprint "${FINGERPRINT}" \
    --pair-manifest "${V52B_DIR}/v52b_pair_manifest.csv" \
    --preparation "${V52B_DIR}/v52b_preparation.json" \
    --config "${V52B_CONFIG}" \
    --models-root "${V52B_MODEL_ROOT}" \
    --output "${V52B_REPORT}" \
    --device "${V52B_EVAL_DEVICE:-cuda:0}"
}

build_tac_target() {
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_tac_topology_target.py" \
    --root "${PROJECT_ROOT}" \
    --config "${PROJECT_ROOT}/configs/tac_topology_target_v1.toml" \
    --output "${TAC_TARGET}"
}

collect_v52c() {
  : "${ACE_MODEL_SHA256:?Set ACE_MODEL_SHA256 to the 64-hex model tree digest}"
  : "${VAE_SHA256:?Set VAE_SHA256 to the 64-hex VAE tree digest}"
  [[ -f "${TAC_TARGET}" ]] || {
    echo "Run build-tac-target first" >&2
    return 3
  }
  [[ -f "${V52A_DIR}/v52a_generation_plan.json" ]] || {
    echo "V5.2a plan is required: ${V52A_DIR}/v52a_generation_plan.json" >&2
    return 3
  }
  ACESTEP_DEVICE="${V52C_DEVICE:-cuda:0}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/collect_tac_v52c.py" \
    --root "${PROJECT_ROOT}" \
    --source-manifest "${V51A_DIAGNOSTIC_VIEW_DIR}/ltsn_manifest_v51a.csv" \
    --v52a-plan "${V52A_DIR}/v52a_generation_plan.json" \
    --v52a-master "${V52A_DIR}/ltsn_manifest_v52a_master.csv" \
    --ace-config "${PROJECT_ROOT}/configs/ace_rerank_180s.toml" \
    --fingerprint "${FINGERPRINT}" \
    --target "${TAC_TARGET}" \
    --output-dir "${V52C_DIR}" \
    --ace-model-sha256 "${ACE_MODEL_SHA256}" \
    --vae-sha256 "${VAE_SHA256}" \
    --workers "${V52C_EXACT_WORKERS:-8}" \
    --batch-size "${V52C_BATCH_SIZE:-64}" \
    --materialize-mode "${MATERIALIZE_MODE:-auto}" \
    --device "${V52C_DEVICE:-cuda:0}"
}

report_v52c() {
  [[ -f "${V52C_DIR}/v52c_response_points.csv" ]] || {
    echo "Run collect-v52c first" >&2
    return 3
  }
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/report_tac_v52c.py" \
    --response-points "${V52C_DIR}/v52c_response_points.csv" \
    --output "${V52C_REPORT}"
}

collect_v52d_repeatability() {
  : "${ACE_MODEL_SHA256:?Set ACE_MODEL_SHA256 to the 64-hex model tree digest}"
  : "${VAE_SHA256:?Set VAE_SHA256 to the 64-hex VAE tree digest}"
  [[ -f "${V52C_DIR}/v52c_generation_plan.json" ]] || {
    echo "Run collect-v52c first" >&2
    return 3
  }
  ACESTEP_DEVICE="${V52D_DEVICE:-cuda:0}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/collect_tac_v52d_repeatability.py" \
    --root "${PROJECT_ROOT}" \
    --source-manifest "${V51A_DIAGNOSTIC_VIEW_DIR}/ltsn_manifest_v51a.csv" \
    --v52c-plan "${V52C_DIR}/v52c_generation_plan.json" \
    --prompt-manifest "${PROMPT_MANIFEST}" \
    --ace-config "${PROJECT_ROOT}/configs/ace_rerank_180s.toml" \
    --fingerprint "${FINGERPRINT}" \
    --target "${TAC_TARGET}" \
    --output-dir "${V52D_REPEAT_DIR}" \
    --ace-model-sha256 "${ACE_MODEL_SHA256}" \
    --vae-sha256 "${VAE_SHA256}" \
    --device "${V52D_DEVICE:-cuda:0}" \
    --workers "${V52D_EXACT_WORKERS:-8}" \
    --batch-size "${V52D_REPEAT_BATCH_SIZE:-4}" \
    --materialize-mode "${MATERIALIZE_MODE:-auto}"
}

collect_v52d_actions() {
  : "${ACE_MODEL_SHA256:?Set ACE_MODEL_SHA256 to the 64-hex model tree digest}"
  : "${VAE_SHA256:?Set VAE_SHA256 to the 64-hex VAE tree digest}"
  [[ -f "${V52D_REPEAT_REPORT}" ]] || {
    echo "Run collect-v52d-repeatability first" >&2
    return 3
  }
  ACESTEP_DEVICE="${V52D_DEVICE:-cuda:0}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/collect_tac_v52d_actions.py" \
    --root "${PROJECT_ROOT}" \
    --source-manifest "${V51A_DIAGNOSTIC_VIEW_DIR}/ltsn_manifest_v51a.csv" \
    --v52c-plan "${V52C_DIR}/v52c_generation_plan.json" \
    --prompt-manifest "${PROMPT_MANIFEST}" \
    --ace-config "${PROJECT_ROOT}/configs/ace_rerank_180s.toml" \
    --fingerprint "${FINGERPRINT}" \
    --target "${TAC_TARGET}" \
    --repeatability-report "${V52D_REPEAT_REPORT}" \
    --output-dir "${V52D_ACTION_DIR}" \
    --ace-model-sha256 "${ACE_MODEL_SHA256}" \
    --vae-sha256 "${VAE_SHA256}" \
    --device "${V52D_DEVICE:-cuda:0}" \
    --workers "${V52D_EXACT_WORKERS:-8}" \
    --batch-size "${V52D_ACTION_BATCH_SIZE:-12}" \
    --materialize-mode "${MATERIALIZE_MODE:-auto}"
}

report_v52d() {
  [[ -f "${V52D_ACTION_DIR}/v52d_action_points.csv" ]] || {
    echo "Run collect-v52d-actions first" >&2
    return 3
  }
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/report_tac_v52d.py" \
    --repeatability-report "${V52D_REPEAT_REPORT}" \
    --action-points "${V52D_ACTION_DIR}/v52d_action_points.csv" \
    --output "${V52D_ACTION_REPORT}"
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
  local -a engineering_args=()
  if [[ "${TRAIN_ENGINEERING_SMOKE:-0}" == "1" ]]; then
    engineering_args=(--engineering-smoke)
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_path_homology_surrogate.py" \
    --fingerprint "${FINGERPRINT}" \
    --manifest "${LTSN_MANIFEST}" \
    --split-manifest "${LTSN_SPLIT_MANIFEST}" \
    --config "${CONFIG}" \
    --output-dir "${MODEL_DIR}" \
    --surrogate-training-gate "${SURROGATE_TRAINING_GATE}" \
    "${engineering_args[@]}" \
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

calibrate_v5() {
  LTSN_MANIFEST="${V5_AUGMENTATION_DIR}/ltsn_manifest_v5.csv" \
  MODEL_DIR="${V5_MODEL_DIR}" \
  CALIBRATION_PATH="${V5_CALIBRATION_PATH}" \
    calibrate
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

calibrate_v5_ood_ablation() {
  CALIBRATION_PATH="${V5_CALIBRATION_PATH}" \
  OOD_ABLATION_CALIBRATION_PATH="${V5_OOD_ABLATION_CALIBRATION_PATH}" \
    calibrate_ood_ablation
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

development_generate_v5() {
  LTSN_MANIFEST="${V5_AUGMENTATION_DIR}/ltsn_manifest_v5.csv" \
  MODEL_DIR="${V5_MODEL_DIR}" \
  CALIBRATION_PATH="${V5_CALIBRATION_PATH}" \
  DEVELOPMENT_DIR="${V5_DEVELOPMENT_DIR}" \
    development_generate
}

development_generate_v5_ood_ablation() {
  LTSN_MANIFEST="${V5_AUGMENTATION_DIR}/ltsn_manifest_v5.csv" \
  MODEL_DIR="${V5_MODEL_DIR}" \
  CALIBRATION_PATH="${V5_CALIBRATION_PATH}" \
  OOD_ABLATION_CALIBRATION_PATH="${V5_OOD_ABLATION_CALIBRATION_PATH}" \
  DEVELOPMENT_DIR="${V5_DEVELOPMENT_DIR}" \
  DEVELOPMENT_OOD_ABLATION=1 \
    development_generate
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

development_score_v5() {
  DEVELOPMENT_DIR="${V5_DEVELOPMENT_DIR}" development_score
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

development_evidence_v5() {
  DEVELOPMENT_DIR="${V5_DEVELOPMENT_DIR}" development_evidence
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

development_finalize_v5() {
  DEVELOPMENT_DIR="${V5_DEVELOPMENT_DIR}" development_finalize
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

qualify_v5() {
  LTSN_MANIFEST="${V5_AUGMENTATION_DIR}/ltsn_manifest_v5.csv" \
  MODEL_DIR="${V5_MODEL_DIR}" \
  CALIBRATION_PATH="${V5_CALIBRATION_PATH}" \
  GUIDANCE_DEVELOPMENT_PATH="${V5_GUIDANCE_DEVELOPMENT_PATH}" \
  QUALIFICATION_PATH="${V5_QUALIFICATION_PATH}" \
    qualify
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

guidance_development_v5() {
  DEVELOPMENT_DIR="${V5_DEVELOPMENT_DIR}" \
  GUIDANCE_DEVELOPMENT_PATH="${V5_GUIDANCE_DEVELOPMENT_PATH}" \
    guidance_development
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
  augment-v5) augment_v5 ;;
  train) train ;;
  train-v5) train_v5 ;;
  train-v5-screen) train_v5_screen ;;
  prepare-v51) prepare_v51 ;;
  train-v51-screen) train_v51_screen ;;
  report-v51-screen) report_v51_screen ;;
  train-v51) train_v51 ;;
  prepare-v51a) prepare_v51a ;;
  train-v51a) train_v51a ;;
  report-v51a) report_v51a ;;
  prepare-v51b) prepare_v51b ;;
  train-v51b) train_v51b ;;
  report-v51b) report_v51b ;;
  prepare-v51c) prepare_v51c ;;
  train-v51c) train_v51c ;;
  report-v51c) report_v51c ;;
  collect-v52a) collect_v52a ;;
  train-v52a) train_v52a ;;
  report-v52a) report_v52a ;;
  prepare-v52b) prepare_v52b ;;
  train-v52b) train_v52b ;;
  report-v52b) report_v52b ;;
  build-tac-target) build_tac_target ;;
  collect-v52c) collect_v52c ;;
  report-v52c) report_v52c ;;
  collect-v52d-repeatability) collect_v52d_repeatability ;;
  collect-v52d-actions) collect_v52d_actions ;;
  report-v52d) report_v52d ;;
  calibrate) calibrate ;;
  calibrate-v5) calibrate_v5 ;;
  calibrate-ood-ablation) calibrate_ood_ablation ;;
  calibrate-v5-ood-ablation) calibrate_v5_ood_ablation ;;
  development-generate) development_generate ;;
  development-generate-v5) development_generate_v5 ;;
  development-generate-v5-ood-ablation) development_generate_v5_ood_ablation ;;
  development-step4-diagnostic) development_step4_diagnostic ;;
  development-step4-score) development_step4_score ;;
  development-step4-report) development_step4_report ;;
  development-score) development_score ;;
  development-score-v5) development_score_v5 ;;
  development-evidence) development_evidence ;;
  development-evidence-v5) development_evidence_v5 ;;
  development-finalize) development_finalize ;;
  development-finalize-v5) development_finalize_v5 ;;
  guidance-development) guidance_development ;;
  guidance-development-v5) guidance_development_v5 ;;
  qualify) qualify ;;
  qualify-v5) qualify_v5 ;;
  guidance-confirmation) guidance_confirmation ;;
  *) echo "Usage: $0 {collect|labels|augment-training|augment-on-policy|augment-v5|train|train-v5|train-v5-screen|prepare-v51|train-v51-screen|report-v51-screen|train-v51|prepare-v51a|train-v51a|report-v51a|prepare-v51b|train-v51b|report-v51b|prepare-v51c|train-v51c|report-v51c|collect-v52a|train-v52a|report-v52a|prepare-v52b|train-v52b|report-v52b|build-tac-target|collect-v52c|report-v52c|collect-v52d-repeatability|collect-v52d-actions|report-v52d|calibrate|calibrate-v5|calibrate-ood-ablation|calibrate-v5-ood-ablation|development-generate|development-generate-v5|development-generate-v5-ood-ablation|development-step4-diagnostic|development-step4-score|development-step4-report|development-score|development-score-v5|development-evidence|development-evidence-v5|development-finalize|development-finalize-v5|guidance-development|guidance-development-v5|qualify|qualify-v5|guidance-confirmation}" >&2; exit 2 ;;
esac
