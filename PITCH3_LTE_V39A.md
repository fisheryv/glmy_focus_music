# V3.9-A：高分辨率时序转移分支实验操作

实现日期：2026-09-18。本版只实施 V3.9-A，比较 G0/L0 等价的 `baseline` 与 `transition`；不包含 V3.9-B 转移矩阵蒸馏或 V3.9-C 局部网络。效果必须由服务器实验决定。

## 实现范围与固定条件

- 新分支在 stride-4 之前读取 latent，采用 normalized/raw-RMS 双输入、64 通道、3 个 dilation=1/2/4 的 separable convolution 残差块。
- lag=1/2/4/8，配对输入为 `[Ht,Ht+l,Ht+l−Ht,Ht⊙Ht+l]`；共享 32 维配对 MLP，汇总 mean/std，再附加 4 个 lag 有效性标志，共 260 维。
- 末层输出投影零初始化后加到原 128 维表示；新增分支没有 dropout。按层尺寸计算新增 80,160 个参数，`check` 会核对实际参数数和 150,000 上限。
- 两组显式复制同一套共享初始张量，核对初始表示/能量/坐标/有效损失相等。两组损失 normalizer 均来自相同的 baseline 初始化、各自 fit 数据；摘要用于汇总时核验。
- 使用原 G0 的 value、logit prompt rank、family 分层 rank、coordinate .25；ordinal 关闭，local 始终为零。教师、Band、分割、5% 扰动半径和原门禁不变。
- CV/global 固定 **12 epochs，online final checkpoint**，每个 epoch 每个 fit family 一次完整 64-base 更新。没有 EMA/early stopping/outer checkpoint selection。
- `diagnose` 是每折 fit 内按家族名称排序的前两个 family，固定 **24 epochs**；它只用于可拟合性诊断，不产生正式 checkpoint。
- 每个 epoch 记录 eval-train 指标、训练分桶诊断、分块裁剪前后梯度、epoch 净参数更新；这些诊断不会改变后续 dropout RNG。CV 外折只在训练完成后评估一次。

原配置中的 `[training]` 部分保留共享损失/优化器配置；**实际轮数只由新配置 `[v39a]` 的 `global_epochs` / `diagnostic_epochs` 控制**。首轮保持 12/24，不在看到外折结果后追加某一组训练。

## 1. 同步源码并确认 Python

服务器项目路径沿用 `/home/pc/glmy_focus_music`。同步全部新增文件及共同模块改动，不能只同步 `.sh` 或主训练文件。新增模块的具体路径可由 `git status --short` 核对。

```bash
cd /home/pc/glmy_focus_music
export PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
nvidia-smi
```

若当前 Python 不是服务器训练环境，直接将 `PYTHON_BIN` 设为正确的解释器入口，例如现有 ACE-Step venv 的 `.../.venv/bin/python`。不要对这个入口使用 `readlink -f`，以免丢失 venv 上下文。

默认输入为：

```text
configs/pitch3_lte_v39a.toml
runs/pitch3_lte_v3/exact_local_dataset/pitch3_lte_examples.csv
metadata/focus_pitch3_fingerprint_v1.json
```

继续使用原始 train/development 数据与对应训练 lineage 的 fingerprint。若服务器原始 fingerprint 位于其他位置，给所有依赖数据的命令传相同的 `--fingerprint /absolute/path/to/original.json`。哈希不匹配需要找回对应原件，不重写 manifest 绕过检查。

本版从共享随机初始化重新训练，不加载旧 V3.8-B checkpoint。旧 checkpoint 的同步截断问题不会阻止这套新实验；原 latent、text embedding 和 dataset 的文件/哈希校验仍须通过。

## 2. 先查看任务列表

```bash
bash scripts/run_pitch3_lte_v39a.sh cv-global --dry-run
```

应显示 10 个任务：2 个 variant × 5 折 × seed 20260941。默认设备为 `cuda:1 cuda:2 cuda:3`，可用 `--devices` 指定实际空闲设备；若设置了 `CUDA_VISIBLE_DEVICES`，这些编号是进程内的逻辑编号。

同一 fold/seed 的两组模型分配到同一 GPU 的顺序队列。`--skip-busy-gpus` 会将全部任务重新排到其余指定 GPU，不减少实验任务。默认 8 GiB 只是启动前的空闲显存下限，不是峰值显存承诺。

所有阶段默认写入 `runs/pitch3_lte_v39a`。训练、diagnose 和 check 拒绝覆盖已有输出；需要重跑时，使用新的 `--run-root`，或只指定尚未运行的 `--folds` / `--seeds`。日志不会覆盖，失败时控制台会显示子进程 traceback。

## 3. 执行 PyTorch 网络自检

先在实际将使用的 GPU 上检查：

```bash
bash scripts/run_pitch3_lte_v39a.sh check \
  --devices cuda:1 cuda:2 cuda:3 --skip-busy-gpus
```

输出 `checks/device_*/pitch3_lte_v39a_checks.json`，需有 `all_checks_passed=true`。自检包括：共享初始张量/输出/损失、训练模式 dropout 一致性、转移分支开始接收梯度、padding 值及长度不变性、无效帧零梯度、缺失 lag 标志、标量输入导数、checkpoint 保存/加载、local 严格为零。

这些是合成张量测试，不代表真实音频有效。自检失败应先定位原因，不直接跳到全量 CV。

如果训练环境已有 pytest，建议同时运行真实优化器的合成流程回归：

```bash
"$PYTHON_BIN" -m pytest tests/test_pitch3_lte_v39a.py -q -ra
```

其中 CPU 集成测试在合成数据上覆盖完整 trainer、固定轮次、初始统计匹配、外折只评一次、fit-only diagnose，以及拒绝覆盖。服务器运行不应因缺少 Torch 跳过张量测试。

## 4. 先做 fit-only 拟合诊断

先跑第 0 折，观察两组模型是否正常学习：

```bash
bash scripts/run_pitch3_lte_v39a.sh diagnose \
  --folds 0 --devices cuda:1
```

完成后检查：

```text
runs/pitch3_lte_v39a/diagnostics/{baseline,transition}/fold_0/seed_20260941/diagnostic/
  pitch3_lte_initialization.json
  pitch3_lte_fit_epoch_zero.json
  pitch3_lte_training_history.json
  pitch3_lte_gradient_initial.json
  pitch3_lte_gradient_final.json
  pitch3_lte_train_predictions.csv
  pitch3_lte_train_diagnostics.json
  pitch3_lte_diagnostic_complete.json
```

主要看 epoch 0 至 24 的 fit 家族 ρ、低值桶排序、零/正 AUC、坐标误差、各块梯度和参数更新。转移末层零初始化使特征层在**第一次 backward** 的梯度为零，这是预期；后续应观察到 `transition_features` 的非零梯度，不能仅凭 weight decay 引起的参数移动判断其已学习。

这不是新的准确性门槛，也不能保证 24 epochs 足以完成小集拟合。若无进展，先审查 `before_clip` / `after_clip` 与实际更新；不通过提高 clip 或反复查看 development 寻找通过方式。

第 0 折正常后补其余四折，避免覆盖前一折：

```bash
bash scripts/run_pitch3_lte_v39a.sh diagnose \
  --folds 1 2 3 4 --devices cuda:1 cuda:2 cuda:3 --skip-busy-gpus
```

## 5. 首轮五折单种子匹配实验

```bash
bash scripts/run_pitch3_lte_v39a.sh cv-global \
  --devices cuda:1 cuda:2 cuda:3 --skip-busy-gpus

bash scripts/run_pitch3_lte_v39a.sh summary
```

训练目录：

```text
runs/pitch3_lte_v39a/cv/{baseline,transition}/fold_{0..4}/seed_20260941/models/
```

汇总目录：

```text
runs/pitch3_lte_v39a/summary/
  pitch3_lte_v39a_cv_baseline_transition_20260941.json
```

汇总会检查模型 ZIP/哈希、配置和代码来源、训练/留出 ID、标签、原门禁复算，以及两组共享初始化和初始 normalizer 一致。统计采用等权成员能量平均，不平均 logits，不改成方向多数投票。

优先比较 `families_passing_original_rho_gate`，再看 `minimum_family_spearman`，并检查同 prompt 排序、collapse 和方向是否退步。`paired_comparison` 给出逐家族增量、新增/丢失通过家族及方向净正确数。低值与坐标诊断用于解释机制，不代替冻结门禁。

## 6. 有一致收益时，再补两个种子

首轮结果值得复验时执行，已有 20260941 不需要重跑：

```bash
bash scripts/run_pitch3_lte_v39a.sh cv-global \
  --seeds 20260942 20260943 \
  --devices cuda:1 cuda:2 cuda:3 --skip-busy-gpus

bash scripts/run_pitch3_lte_v39a.sh summary \
  --seeds 20260941 20260942 20260943
```

这一步新增 20 个 global 训练任务。单种子和三种子汇总文件均保留。若 A 没有可重复收益，停止扩大 A；根据 fit/留族差异决定后续 teacher 对齐或数据覆盖实验，不自动执行 B/C。

## 7. 选定方案后才全 train 和 model-screen

以下以内部证据选定 `transition` 为例；这不是脚本替用户自动宣布它获胜。baseline 若更合适，也必须经过同样的原门禁。

```bash
bash scripts/run_pitch3_lte_v39a.sh train-global \
  --variants transition \
  --devices cuda:1 cuda:2 cuda:3 --skip-busy-gpus

bash scripts/run_pitch3_lte_v39a.sh ensemble --variants transition

bash scripts/run_pitch3_lte_v39a.sh model-screen \
  --variants transition --devices cuda:1
```

全 train 默认三种子仍为 20260941/42/43。`train-global` 不读取 development 预测进行选择；`ensemble` / `model-screen` 必须显式选择单一 variant，不允许混合 baseline/transition。诊断和 CV checkpoint 不可组成正式引导 ensemble。

最终输出在：

```text
runs/pitch3_lte_v39a/final/transition/
  seed_20260941/models/
  seed_20260942/models/
  seed_20260943/models/
  pitch3_lte_ensemble.json
  development_screen_model_only_fp32/
```

development 已被反复用于模型开发，其通过不能称为全新独立确认。五项原模型门禁全部通过后，才评估原 Step-4 exact 改善率和质量非劣；本版本不自动启动音频 guidance 或授予 production authorization。

## 本地验证边界与同步结果

本地执行语法、Ruff、Torch-free 协议/runner/summary 回归。当前本地 Python 不含 PyTorch，因此未执行真实网络 forward/backward、训练和显存测量；上面的 `check` 和 CPU 合成 trainer 回归是服务器执行步骤。真实准确性结果仍待本轮运行。

分析时请同步完整 `runs/pitch3_lte_v39a`，尤其保留配置、protocol、statistics、initialization、逐 epoch history、train/selection 预测、diagnostics、manifest 和完整 `.pt`。只同步部分 checkpoint 字节会被 ZIP/哈希验证拒绝。
