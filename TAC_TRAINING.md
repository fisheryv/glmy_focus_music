# TAC 新线路：分阶段训练与验证协议

本协议替代继续扩展旧 LTSN surrogate 的做法。每个阶段必须通过冻结门禁，才能生成下一阶段
的训练代码或数据。当前仓库实现 Stage 0 和 Stage 1；TAC critic、候选控制器和 actor 尚未被
授权训练。

## Stage 0：冻结 18D 目标

目标工件不修改 `focus_path_homology_fingerprint_v2.json`。它使用 discovery/focus 的 195 个
冻结 18D 坐标拟合稳健中心、Pitch 13 个有效维度的固定收缩协方差，以及两个 phase 坐标的
稳健尺度。目标距离越小越接近 Focus 参考拓扑：

```text
D(x) = 0.5 D_pitch(x) + 0.25 D_acoustic(x) + 0.25 D_chroma(x)
reward(x_before, x_after) = D(x_before) - D(x_after)
```

构建并验证：

```bash
export RUN_ROOT=$PWD/runs/ltsn_turbo
bash scripts/run_ltsn_pipeline.sh build-tac-target

PYTHONPATH=src python -m pytest \
  tests/test_tac_target.py \
  tests/test_path_homology_fingerprint_v2.py
```

预期工件为 `metadata/tac_topology_target_v1.json`。它哈希绑定原始 Pitch/phase 表、冻结指纹、
配置和构建代码，并明确标记为 diagnostic-only、不可用于 qualification 或 production
guidance。

## Stage 1：V5.2c 局部响应曲线

本阶段使用 V5.2a 的 16 个 unseen anchors 和 8 个冻结正交方向。复用已经 exact-scored 的
`0.0025`、`0.005` 两个半径，只新增 `0.00125`、`0.00375`、`0.0075` 三个半径：

```text
16 anchors × 8 directions × 3 new radii × 2 signs = 768 new decodes
```

先确认以下旧产物仍存在：

```bash
test -f "$RUN_ROOT/training_augmentation_v51a/ltsn_manifest_v51a.csv"
test -f "$RUN_ROOT/training_augmentation_v52a/v52a_generation_plan.json"
test -f "$RUN_ROOT/training_augmentation_v52a/ltsn_manifest_v52a_master.csv"
```

然后使用与 V5.2a 完全相同的 XL-Turbo 模型和 VAE 哈希运行：

```bash
export ACE_MODEL_SHA256=<与-v52a-一致的64位sha256>
export VAE_SHA256=<与-v52a-一致的64位sha256>
export V52C_DEVICE=cuda:0
export V52C_EXACT_WORKERS=8
export V52C_BATCH_SIZE=64

bash scripts/run_ltsn_pipeline.sh collect-v52c
bash scripts/run_ltsn_pipeline.sh report-v52c
```

每批的处理顺序为：生成并保留 latent、临时解码 WAV、exact descriptor/18D 评分、写入哈希
收据、删除该批 WAV。默认批量为 64，因此 180 秒 FLOAT WAV 的瞬时主体约 4.1 GiB；还需为
预处理副本和特征留余量，建议至少保留 15--25 GiB 空闲空间。中断后直接重跑
`collect-v52c`，已完成批次从 descriptor 和收据恢复，不需要 WAV。

主要输出：

- `runs/ltsn_turbo/tac_v52c/v52c_response_report.json`
- `runs/ltsn_turbo/tac_v52c/v52c_response_points.csv`
- `runs/ltsn_turbo/tac_v52c/v52c_response_outcomes.csv`
- `runs/ltsn_turbo/tac_v52c/latents/*.npy`
- `runs/ltsn_turbo/tac_v52c/batches/*/descriptors.csv`

成功结束时 `retained_wav_files` 必须为 0。Stage 1 主门禁固定为：

```text
cross_radius_sign.agreement >= 0.80
cross_radius_sign.wilson95[0] > 0.50
```

只有 `stage1_status == "supported_for_critic_collection"` 且
`next_stage_authorized == true` 才进入 Stage 2。不要因为失败而降低阈值。

## Stage 2：TAC critic（等待 Stage 1 授权）

Stage 1 通过后再实现并采集 on-manifold action 数据。首轮 64 anchors，每个 anchor 约 48 个
动作端点；训练 latent-only、audio-preview-only 和 fused 三个 critic，并使用全新 anchors、
matched-action permutation control 和 no-op 校准验证。critic 输出 18D delta 分布、目标奖励
分位数、质量风险和 OOD 风险。

进入候选控制器前，至少要求：

1. fresh-anchor 动作排序显著优于 matched control；
2. reward 符号 Wilson 下界高于随机水平；
3. 保守下置信界选出的动作，经 exact scorer 闭环确认优于 no-op；
4. 音质、prompt adherence 和多样性不劣化门禁同时通过。

Stage 2 未通过时停止，不实现 actor。

## Stage 3/4：有限候选控制器与 actor

Stage 3 先在每个去噪状态枚举少量 on-manifold 动作，以 critic 的 conservative lower
confidence bound 排序，并保留 no-op fallback。只有 fresh prompt/seed 的 exact 闭环试验通过，
才进入 Stage 4 的 direction-aware actor--critic。actor 只模仿已验证的候选选择并接受 critic
约束；最终授权仍由独立 exact scorer、质量门禁和新鲜确认集决定。


