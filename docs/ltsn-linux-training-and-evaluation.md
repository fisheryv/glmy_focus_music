# ACE-Step 18-D LTSN：Linux/NVIDIA 训练与评估指引

更新日期：2026-08-29

本指引对应 `focus_path_homology_fingerprint_v2`，即 Pitch 16 维、Acoustic phase
`loop_score` 1 维和 Chroma phase `loop_score` 1 维。Exact Path Homology 是教师和
最终裁判；LTSN 只是可微代理。

当前仓库已经补齐轨迹采集、逐快照 exact 标注、分组切分校验、三 seed 训练、校准、
独立资格检验与 decoded exact 终评接口。真实 LTSN 标注和训练仍受 Stage 1 exact
reranking 效果门禁阻断；`--engineering-smoke` 只用于验证软件流水线，产物永远不能
通过资格签发。

## 1. Linux/NVIDIA 环境

目标服务器为 2× Intel Xeon Silver 4514Y（合计 32 核/64 线程）、256 GiB 内存和
2× NVIDIA L40S。建议 Ubuntu 22.04/24.04、Python 3.11、支持 CUDA 12.8 的 NVIDIA
驱动，并将 trajectories、exact work 和 labels 放在同一块本地 NVMe 文件系统。

推荐从发布仓库执行冻结安装：

```bash
git clone https://github.com/fisheryv/glmy_focus_music.git
cd glmy_focus_music
bash scripts/bootstrap_linux_l40s.sh
source ACE-Step-1.5/.venv/bin/activate
```

脚本会签出 `pyglmy` 的
`49bd5ea7617906f09940dcc9b9718bbfc1482d6f` 和 ACE-Step 的
`a5632cda3084f1088e69b2057dde7047e1bb4839`，应用 sampler patch，并使用 ACE 的
`uv.lock` 创建共享环境。手工安装时可参考：

```bash
nvidia-smi
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 按 NVIDIA/PyTorch 官方矩阵选择 CUDA wheel；以下仅为 CUDA 12.8 示例。
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[audio,stats,tda,topology-guidance,repro,dev]"
python -m pip install -e ./ACE-Step-1.5
```

核对 GPU 与冻结 ACE checkout：

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(torch.cuda.is_bf16_supported())
PY

git -C ACE-Step-1.5 rev-parse HEAD
# 必须是 a5632cda3084f1088e69b2057dde7047e1bb4839
```

下载并核对公开数据集：

```bash
python scripts/prepare_release_dataset.py
python scripts/verify_linux_l40s.py --root .
```

下载器逐个核对 600 条 `SHA256SUMS`，并验证 `tracks.csv`、`licenses.csv` 与根仓库
`metadata/track_index.csv` 的身份。Hugging Face 数据用于重建语料分析和 exact scorer；
LTSN 的 ACE 轨迹仍来自独立 prompt 清单，不能用语料 split 冒充轨迹 split。

若 Linux checkout 尚未包含 sampler hook，在干净的上述 commit 上执行：

```bash
git -C ACE-Step-1.5 apply ../patches/ace-step-1.5-topology-corrector.patch
git -C ACE-Step-1.5 diff --check
```

该 hook 位于 Euler/Heun 与 DCW 之后、Repaint 注入之前；未安装 corrector 时完全不
改变原采样。

## 2. 哈希与输入清单

训练数据必须绑定 ACE 模型和 VAE 的内容哈希。目录可用以下可复现方法生成摘要：

```bash
find ACE-Step-1.5/checkpoints/acestep-v15-turbo -type f -print0 \
  | sort -z | xargs -0 sha256sum | sha256sum
find ACE-Step-1.5/checkpoints/vae -type f -print0 \
  | sort -z | xargs -0 sha256sum | sha256sum
```

prompt CSV 必须显式包含 `prompt_id,caption,split`，可选 `seed,bpm,keyscale,timesignature`。
`split` 只能是 `train`、`development`、`calibration`、`qualification`；同一 prompt
和 trajectory 不得跨分区。Qualification prompts 必须在设计上独立，不能只把同一
prompt 的另一个 seed 标成 qualification。

`configs/ltsn_prompts.template.csv` 只演示 schema，不是正式实验设计，也不足以训练或
签发模型。正式清单应另存为 `metadata/ltsn_prompts.csv`，在采集前审计 prompt 独立性和
每个 split 的计划数量。

Turbo 与 `acestep-v15-sft` 的轨迹、manifest 和 checkpoint 必须分开。当前已提交的
ACE hook 只覆盖 Turbo PyTorch 路径；SFT 在增加和单独验证相同语义的 hook 前不得
复用 Turbo checkpoint。

## 3. Stage 0：无干预轨迹采集

```bash
export PYTHONPATH="src:packages/pyglmy/src"
python scripts/collect_ltsn_trajectories.py \
  --root . \
  --ace-config configs/ace_rerank_180s.toml \
  --prompt-manifest metadata/ltsn_prompts.csv \
  --output-dir runs/ltsn_turbo/trajectories \
  --backend ace \
  --ace-model-sha256 <64-hex> \
  --vae-sha256 <64-hex> \
  --duration-seconds 180 \
  --decode-snapshots \
  --discard-generator-final-audio
```

采集器通过同一个 sampler hook 保存 step 4、5、6 和 final 的
`x0_hat = x_t - t*v_t`，随后返回原 `xt_next`，因此是 no-op。每个 latent 为
`[T,64]` `.npy`；`--decode-snapshots` 使用当前 ACE VAE 为每个快照单独生成 WAV。
`trajectory_manifest.csv` 记录 latent/audio、ACE、VAE、model family 与 split 哈希。
`--discard-generator-final-audio` 只删除 ACE 生成器额外保存、且未被 manifest 引用的
最终 WAV；它必须与 `--decode-snapshots` 同时使用，并且不会删除 step 4/5/6/final
快照。若需要人工检查生成器的原始最终 WAV，可以省略该参数，但 2048 条轨迹约增加
142 GB 占用。

## 4. Stage 1：单独签发 reranking 效果门禁

LTSN 标签构建不接受 scorer release manifest 代替效果证据。需要另行生成 JSON：

```json
{
  "gate": "exact_reranking_effect_v1",
  "status": "passed",
  "fingerprint_json_sha256": "<signed 18-D scorer SHA-256>",
  "median_loss_improvement_fraction": 0.10,
  "bootstrap_ci95_low": 0.001,
  "target_band_hit_rate_improved": true,
  "quality_noninferior": true,
  "prompt_noninferior": true,
  "diversity_preserved": true
}
```

代码会重新检查改善至少 10%、按 prompt 聚类 bootstrap CI 下界大于 0，以及命中率、
质量、prompt 一致性和多样性门槛。当前仓库尚无这份通过产物，因此真实标注/训练按
设计仍会停止。

## 5. Stage 2：逐快照 exact 标签

```bash
python scripts/build_ltsn_labels.py \
  --root . \
  --fingerprint metadata/focus_path_homology_fingerprint_v2.json \
  --trajectory-manifest runs/ltsn_turbo/trajectories/trajectory_manifest.csv \
  --work-dir runs/ltsn_turbo/exact_work \
  --output-dir runs/ltsn_turbo/labels \
  --reranking-gate metadata/ace_reranking_effect_gate.json \
  --workers 4 \
  --batch-size 256 \
  --materialize-mode auto
```

真实路径对每个快照执行：VAE WAV -> 冻结预处理 -> Pitch-v2 codebook -> exact Path
Homology 20 descriptors -> 双 phase exact `loop_score` -> 签发 18-D scorer 变换与 Focus
logit。它不会把 final 标签复制给中间步。

输出包括：

- `exact_snapshot_descriptors.csv`：scorer 之前的 exact descriptors；
- `exact_labels.csv`：18-D 坐标、Focus logit、band loss 与块距离；
- `split_manifest.json`：prompt 分组分区；
- `ltsn_manifest.csv`：训练输入，绑定 latent、标签表、scorer、ACE/VAE 和 gate 哈希。
- `exact_snapshot_descriptors_storage.json`：批大小、链接/复制方式、断点续跑批次数、
  manifest 与 descriptor 哈希。

### 5.1 存储有界的 exact 标注

正式标注默认每 256 个快照为一批。每批完成后先原子写入 descriptor CSV 和哈希收据，
再删除该批 `data_raw`、预处理 WAV、特征和 phase 中间文件；源目录中的 snapshot WAV、
latent、trajectory manifest 不会被删除。进程中断后再次运行同一命令，会验证 manifest、
批输入和 descriptor 哈希并从下一个未完成批次继续。

`--materialize-mode auto` 按以下顺序物化只读快照：

1. Linux 文件系统支持时使用 reflink（copy-on-write）；
2. 同一文件系统时使用 hardlink；
3. 前两者均不可用时执行物理复制，并在安装前复核 SHA-256。

因此建议 trajectories、labels 和 `exact_work` 位于同一 NVMe 文件系统。显式使用
`--materialize-mode reflink` 或 `hardlink` 时，若文件系统不支持会直接失败，不会静默
退回复制。调试时可增加 `--keep-batch-work` 保留中间文件；需要强制重算所有批次时使用
`--no-resume`。不要在批处理期间修改源 snapshot WAV；hardlink 依赖其不可变性。

对于 8192 个 180 s FLOAT WAV，旧的一次性管线峰值约 1.5--1.8 TB。采用
`--discard-generator-final-audio`、链接物化和 256 快照分批清理后，典型峰值约为
0.7--1.0 TB，另需保留失败重跑与文件系统余量。2 TB NVMe 可以运行，4 TB NVMe
仍是保留全部审计音频时的推荐配置。磁盘紧张时可把 `--batch-size` 降至 128；提高到
512 会增加单批预处理 WAV 和特征的峰值，但不改变标签。

仅检查软件时可加 `--engineering-smoke`。该模式用 deterministic synthetic descriptors，
明确写入 `qualification_eligible=false`，不能加载到正式资格流程。

## 6. 三 seed 训练

```bash
python scripts/train_path_homology_surrogate.py \
  --fingerprint metadata/focus_path_homology_fingerprint_v2.json \
  --manifest runs/ltsn_turbo/labels/ltsn_manifest.csv \
  --split-manifest runs/ltsn_turbo/labels/split_manifest.json \
  --config configs/ltsn_training.toml \
  --output-dir runs/ltsn_turbo/models \
  --reranking-gate metadata/ace_reranking_effect_gate.json \
  --device cuda
```

冻结起点是 AdamW `3e-4`、weight decay `1e-2`、5% warmup + cosine、bf16 forward +
fp32 loss、effective batch 64、global grad clip 1.0 和三个 seed。Batch sampler 尽量把
同 trajectory 快照放在一起，以启用同 prompt ranking 与 trajectory delta loss。
Early stopping 使用 development Focus score error、18-D 坐标 Spearman 和 Pitch/phase
块 Spearman 的联合目标。

每个 checkpoint 保存 state dict、完整网络/损失/训练配置、开发历史和所有上游哈希；
`ensemble_manifest.json` 再记录 checkpoint SHA-256。上游任一哈希改变都必须建立新
run 目录，不能覆盖旧模型。

当前 `train_ensemble()` 在一张卡上顺序训练三个 seed；`--device cuda:0` 不会自动使用
第二张 L40S，也没有宣称 DDP。正式首轮建议保持该冻结语义，用第二张卡做独立推理/
开发评估或空闲备用；若以后实现按 seed 并行，必须合并并复核三个 checkpoint 哈希后
重新签发 ensemble manifest。`scripts/run_ltsn_pipeline.sh` 明确用 `TRAIN_DEVICE` 选择卡。

## 7. Calibration 与独立 qualification

```bash
python scripts/evaluate_ltsn_qualification.py calibrate \
  --fingerprint metadata/focus_path_homology_fingerprint_v2.json \
  --manifest runs/ltsn_turbo/labels/ltsn_manifest.csv \
  --ensemble-manifest runs/ltsn_turbo/models/ensemble_manifest.json \
  --output runs/ltsn_turbo/models/calibration.json \
  --device cuda
```

Calibration 只使用 calibration split，冻结 90% 区间缩放、OOD probability、aleatoric、
epistemic 和 interval-width no-op 阈值。

代理优化后的 latent 必须先解码并形成 paired table，字段至少为：
`prompt_id,fingerprint_json_sha256,exact_focus_band_loss_before,exact_focus_band_loss_after,`
`proxy_focus_band_loss_before,proxy_focus_band_loss_after,quality_noninferior,`
`prompt_noninferior`。然后运行：

```bash
python scripts/evaluate_path_homology_guidance.py \
  --fingerprint metadata/focus_path_homology_fingerprint_v2.json \
  --pair-table runs/ltsn_turbo/development_pairs.csv \
  --output runs/ltsn_turbo/development_exact_direction.json \
  --mode development

python scripts/evaluate_ltsn_qualification.py qualify \
  --fingerprint metadata/focus_path_homology_fingerprint_v2.json \
  --manifest runs/ltsn_turbo/labels/ltsn_manifest.csv \
  --ensemble-manifest runs/ltsn_turbo/models/ensemble_manifest.json \
  --calibration runs/ltsn_turbo/models/calibration.json \
  --guidance-development-report runs/ltsn_turbo/development_exact_direction.json \
  --output runs/ltsn_turbo/models/qualification.json \
  --device cuda
```

资格报告同时给出 overall 与 step 4/5/6/final 分层结果。硬门槛是 Focus logit
Spearman >= 0.70、坐标中位 Spearman >= 0.50、Pitch/phase 块和两个 phase 坐标
Spearman 均 >= 0.50、四分位排序准确率 >= 0.65、90% coverage 在 0.85--0.95、
OOD AUROC >= 0.80，以及 decoded exact/proxy 同向比例 >= 0.65 且中位 exact 改善为正。
任一失败即不签发采样资格。

## 8. 采样引导终评边界

只有 `qualification.json` 为 `qualification_passed=true` 时，才可构造
`TopologyCorrectorConfig(enabled=true, qualification_passed=true, ...)` 并使用
calibration 中的四个阈值。Corrector 只在 step 4--6 运行，RMS clip 只允许
0.25%、0.5%、1.0%，OOD、高不确定、NaN/Inf 或 mask 无效时逐样本 no-op。

最终 confirmation 使用全新 32 prompt x 8 seed 的 baseline/guided 配对，所有音频
仍需 decoded exact 18-D scorer、质量与 prompt 非劣检查。代理分数改善而 exact loss
不改善时属于 proxy gaming，结论必须判失败并退回 exact reranking。

## 9. 快速工程 smoke

在没有 GPU/ACE 权重时，可验证 gate、manifest、哈希和标签拼接；安装 CPU torch 后还
可跑小模型训练：

```bash
python scripts/collect_ltsn_trajectories.py \
  --root . --ace-config configs/ace_rerank_180s.toml \
  --prompt-manifest configs/ltsn_smoke_prompts.csv \
  --output-dir runs/ltsn_smoke/trajectories --backend synthetic \
  --ace-model-sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --vae-sha256 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
  --engineering-smoke

python scripts/build_ltsn_labels.py \
  --trajectory-manifest runs/ltsn_smoke/trajectories/trajectory_manifest.csv \
  --output-dir runs/ltsn_smoke/labels --engineering-smoke
```

Smoke 成功只说明接口和审计约束可运行，不提供 LTSN 拟合、拓扑引导或音乐效果证据。
