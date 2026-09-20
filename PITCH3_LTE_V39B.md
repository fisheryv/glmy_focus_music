# V3.9-B：先检查可拟合性，再做冻结转移教师蒸馏

实现日期：2026-09-20。本阶段依据 `reports/pitch3_lte_v39a_audit_20260919/REPORT.md`。
真实效果必须由服务器运行决定。原 V3.9-A 配置、训练实现与生成物保持原样。

## 已实现的实验

| stage | 对照 | 固定预算 | 数据与用途 |
|---|---|---:|---|
| `check` | 合成张量检查 | 无训练实验 | 原 A 检查＋新教师梯度、优化器和 head 保存/加载 |
| `budget` | baseline / transition | 各 48 epochs | 每折两个 fit 家族，排除预算不足；不导出 checkpoint |
| `teacher` | 冻结状态序列→联合计数 | 1280 个 train base 样本 | 重构 q2/q3、绑定文件哈希；不使用 development |
| `distill-diagnose` | transition / distill | 各 48 epochs | 相同小集合上的蒸馏可拟合性；不导出 checkpoint |
| `cv` | transition / distill | 各 12 epochs | 原五折，固定最后一轮；外折仅评估一次 |

`budget-summary`、`distill-diagnose-summary` 和 `cv-summary` 分别复算相应阶段。
各阶段独立启动，没有自动进入下一阶段、自动选模型、development screen 或 guidance。

## 固定协议与网络变化

- 全局仍采用原 G0 的 value、prompt rank、family stratified rank、coordinate .25；ordinal 关闭，local 始终为零。
- learning rate=0.0002、weight decay=0.0001、clip=1、FP32；原教师、Band、划分和门禁不变。
- 新配置 `configs/pitch3_lte_v39b.toml` 保留原 A 的 model/training/v39a 段，实际新轮数由 `[v39b]` 指定。不要修改原 A 配置来运行本阶段。
- 配对任务在同一 GPU 顺序运行。共享主干、transition 分支和初始统计都可核验；蒸馏两组还共享教师 head 初始状态。
- `distill` 从原 128 维 transition summary 接一个训练用 `Linear(128,256)`，新增 33,024 参数。softmax 得到 16×16 联合分布，不扩大主干或原时序分支。
- 蒸馏项为 `KL(P_teacher || P_student)` 和由 P 推出 q2/q3 的带宽归一化 Smooth-L1。二者分别除以同一初始模型在 fit 上的平均损失，再各乘 **0.25**。首轮已预先固定，不根据 outer 成绩调权重。
- 控制组也建立相同辅助 head 以计算一致的初始归一化，两个蒸馏权重为 0，head 不被优化。全局能量始终来自原能量头；不以预测坐标的硬 Band 替换它。
- 不预训练、不使用 EMA、不 early-stop、不选最佳 epoch。CV checkpoint 标注 internal CV only，不能送入原 ensemble/promotion 流程冒充正式模型。

## 五组小集合

组合依据 A 的审查预先固定；运行时不根据预测指标选样。每组两个家族均位于对应折的 fit 内，包含该家族全部 16 prompts ×4 base；local 样本只用于拟合诊断的方向评估。

| 外折编号 | 小集合 fit 家族 |
|---|---|
| 0 | p02_felt_piano、p10_nylon_guitar |
| 1 | p01_soft_piano、p21_lofi_keys |
| 2 | p11_guitar_pad、p18_muted_strings |
| 3 | p06_analog_pad、p17_cello_pad |
| 4 | p03_piano_marimba、p22_soft_rhodes |

共 10 个不同家族，不再重复旧诊断的 p01+p02。每轮两次 optimizer 更新，48 轮共 96 次。
五组数据来自已有 train，不是新增独立验证数据。

## 1. 同步代码并执行自检

在服务器现有 PyTorch 环境执行，沿用 `/home/pc/glmy_focus_music`。同步所有新增模块、scripts、config 和 tests；不要只同步 `.sh`。

```bash
cd /home/pc/glmy_focus_music
export PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" -c "import sys, torch; print(sys.executable, torch.__version__, torch.cuda.is_available())"
nvidia-smi

"$PYTHON_BIN" -m pytest tests/test_pitch3_lte_v39b.py tests/test_pitch3_lte_v39a.py -q -ra

bash scripts/run_pitch3_lte_v39b.sh check \
  --devices cuda:1 cuda:2 cuda:3 --skip-busy-gpus
```

自检输出 `checks/device_*/pitch3_lte_v39b_checks.json`，应有 `all_checks_passed=true`。
服务器测试不能因缺少 Torch 跳过张量和真实优化器用例。这里的 synthetic tests 仅验证实现，不代表真实准确性。

## 2. 现在先运行 budget

可先查看任务，不启动训练：

```bash
bash scripts/run_pitch3_lte_v39b.sh budget --dry-run
```

推荐先第 0 折，完成后补其余四折：

```bash
bash scripts/run_pitch3_lte_v39b.sh budget --folds 0 --devices cuda:1
bash scripts/run_pitch3_lte_v39b.sh budget-summary --folds 0

bash scripts/run_pitch3_lte_v39b.sh budget \
  --folds 1 2 3 4 --devices cuda:1 cuda:2 cuda:3 --skip-busy-gpus
bash scripts/run_pitch3_lte_v39b.sh budget-summary
```

也可第一次就运行全部五折；不要执行全五折命令覆盖已完成的第 0 折。训练拒绝覆盖非空目录。

默认目录：

```text
runs/pitch3_lte_v39b/budget/{baseline,transition}/fold_*/seed_20260941/
  pitch3_lte_run_protocol.json
  pitch3_lte_initialization.json
  pitch3_lte_training_statistics.json
  pitch3_lte_fit_epoch_zero.json
  pitch3_lte_training_history.json
  pitch3_lte_family_gradient_epoch_000.json
  pitch3_lte_family_gradient_epoch_012.json
  pitch3_lte_family_gradient_epoch_024.json
  pitch3_lte_family_gradient_epoch_048.json
  pitch3_lte_train_predictions.csv
  pitch3_lte_train_diagnostics.json
  pitch3_lte_v39b_complete.json
```

每轮 history 记录：家族 fit 指标、训练定义的低值桶、零/正 AUC、各项损失、每家族裁剪前后分块梯度、epoch 净参数更新。
0/12/24/48 轮另做 eval 模式、逐家族梯度探针，记录实际优化目标以及 **低值 value / 低值 rank** 的梯度和低值 q1/q2/q3 rho。这些 low probe 不进入优化器；它们不能被误报为新的训练目标。所有额外评估都隔离 RNG，不改变后续 dropout。

汇总对 12/24/48 固定观察点做记录，不从中选 checkpoint。判断方向：

- 长预算下低值与 q2 明显改善：先考虑优化预算/目标干扰的后续独立实验，暂不扩大蒸馏。
- 全局排序改善但低值仍接近零，新分支梯度和更新正常：继续下面的冻结教师蒸馏。
- 梯度非有限、分支长期无更新或数据校验失败：先排查实现/数据；不要提高 clip、降低门禁或重写 hash 获得通过。

这些是机制判断，不新增一个要求小集必须通过全部正式门禁的规则。

## 3. 需要蒸馏时准备冻结 teacher

目标采用所有有效相邻状态的联合计数 `C`，保留对角线：

```text
P = C / sum(C)
raw_self_transition_ratio = trace(P)
raw_directed_recurrence = sum(P**2)
```

必须是全局归一化，不能逐行归一化，也不能使用删除自环、top-k 后的图。无效帧会打断邻接，不能把缺失片段两端连起来。

准备器复用原 `preprocess_candidates`、`extract_candidate_features`、Tonnetz/codebook 分配规则，限定原 180 秒单分析窗口。每个样本重构 q2/q3 并与原 exact 标签比较（absolute tolerance=1e-7）；不匹配立即停止，保留该样本 scratch 供检查。不会调整教师标签适配新音频。H0 不由这两个公式替代。

若原 signed trajectory WAV 仍存在，可以直接使用：

```bash
bash scripts/run_pitch3_lte_v39b.sh teacher \
  --trajectory-manifest runs/ltsn_turbo/trajectories/trajectory_manifest.csv \
  --devices cuda:1
```

原数据流程可能已清理 WAV。这种情况下，需要原始构建数据时的 ACE 配置来重解码**同一个记录的 x0**，不运行新的 diffusion sampling：

```bash
# 指向原 build-dataset 使用的配置原件，不能凭文件名换成当前修改版。
export ORIGINAL_ACE_CONFIG=/absolute/path/to/original_ace_config.toml

bash scripts/run_pitch3_lte_v39b.sh teacher \
  --ace-config "$ORIGINAL_ACE_CONFIG" \
  --trajectory-manifest runs/ltsn_turbo/trajectories/trajectory_manifest.csv \
  --devices cuda:1 --workers 4
```

ACE 配置必须匹配 `runs/pitch3_lte_v3/exact_local_dataset/pitch3_lte_dataset_plan.json` 内的 `ace_config_sha256`。原始配置 hash 可这样查看：

```bash
"$PYTHON_BIN" -c "import json; print(json.load(open('runs/pitch3_lte_v3/exact_local_dataset/pitch3_lte_dataset_plan.json'))['ace_config_sha256'])"
```

teacher 默认只用第一个指定 GPU，VAE/ACE 的显存开销与 LTE 训练不同，应选择足够空闲的设备。缓存可用时不加载 ACE；缓存缺失且未提供原配置时给出明确错误。重解码路线仍必须通过逐样本 q2/q3 重构。

teacher 支持同一命令中断续跑：已完成样本有签名计划绑定的 receipts，不重做；每个样本成功后清理新建 scratch 音频，保留 counts 与 provenance。中断后务必使用同一 source/config 参数；参数改变使用新的 `--run-root`，不修改旧计划。

输出：

```text
runs/pitch3_lte_v39b/teacher/
  teacher_plan.json
  receipts/*.json
  transition_targets.npz
  teacher_manifest.json
```

完整 teacher 包含原 1280 个 train base；训练仅接收对应折 fit IDs，outer teacher 不进入损失或 normalizer，development 不在 archive 中。loader 复核数据、fingerprint、targets 哈希和 q2/q3 重构。
准备器同时校验原数据 summary/plan；loader 交叉校验恢复计划与逐样本 receipt，拒绝改变分析时长、重构容差或数据使用范围的产物。

## 4. 蒸馏小集对照，再做 CV

teacher 完成且预算实验支持进入蒸馏后：

```bash
bash scripts/run_pitch3_lte_v39b.sh distill-diagnose \
  --devices cuda:1 cuda:2 cuda:3 --skip-busy-gpus
bash scripts/run_pitch3_lte_v39b.sh distill-diagnose-summary
```

小集合复核后启动原五折单种子匹配实验：

```bash
bash scripts/run_pitch3_lte_v39b.sh cv \
  --devices cuda:1 cuda:2 cuda:3 --skip-busy-gpus
bash scripts/run_pitch3_lte_v39b.sh cv-summary
```

只有一致收益值得复验时，再补种子：

```bash
bash scripts/run_pitch3_lte_v39b.sh cv --seeds 20260942 20260943 \
  --devices cuda:1 cuda:2 cuda:3 --skip-busy-gpus
bash scripts/run_pitch3_lte_v39b.sh cv-summary --seeds 20260941 20260942 20260943
```

汇总优先看通过家族数、最差家族、低值 rho 和 q2/q3，同时检查同 prompt 排序及局部方向是否下降。使用等权成员能量平均，不混合 logits 或方向多数投票。CV 权重不会自动用于正式 inference；全局与局部问题仍需各自原门禁验证。

## 同步与验证边界

新入口：`scripts/run_pitch3_lte_v39b.sh`；新协议、trainer、teacher builder、synthetic checks、summary 和 tests 均已实现。
本机 NumPy 协议/数据检查可以运行，但无 PyTorch，未执行真实 CUDA 教师重解码或训练。服务器合成检查与真实小集结果是下一步证据，不能把本地测试通过当成准确性提高。

2026-09-20 本地验证：V3.9-B/A 与 V3.8-B 的定向回归共 **55 passed、11 skipped**，跳过项均依赖 PyTorch。新增 Python 文件通过 Ruff 与编译检查；五个 CLI 的 `--help` 和全部八个阶段的 `--dry-run` 均通过。服务器应补跑文档第 1 步的张量/优化器测试和 GPU 自检。

完成 budget 后同步整个 `runs/pitch3_lte_v39b/budget` 和 `summary`；若继续蒸馏，同时保留 teacher manifest/targets/receipts、checks、完整 CV checkpoint 及所有诊断 JSON，方便复核来源与模型行为。
