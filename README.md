# Focus Music GLMY：18-D LTSN 可复现训练仓库

本仓库是 `open-focus-classical-600` 研究分析、冻结 18-D exact scorer、ACE-Step
轨迹采集、逐快照 exact 标签、LTSN 训练/校准/资格检验，以及默认关闭的潜空间拓扑
corrector 的统一发布仓库。生产目标为 Linux x86_64、Python 3.11、256 GiB 内存和
2× NVIDIA L40S。

当前证据边界：代码与本地工程测试可用；真实 Linux/L40S 轨迹、正式标签、三 seed
训练和资格签发尚未在本仓库产出。`--engineering-smoke` 永远不构成资格证据；正式标签
必须先通过独立的 `exact_reranking_effect_v1` 门禁。

## Linux / 2×L40S 快速部署

服务器建议准备 2 TiB 以上同一文件系统的本地 NVMe（保留完整审计音频建议 4 TiB）。
系统需预装 NVIDIA 驱动、Git、FFmpeg 和 Python 3.11；CUDA 用户态由 ACE-Step 的
冻结 `uv.lock` 安装。

```bash
git clone https://github.com/fisheryv/glmy_focus_music.git
cd glmy_focus_music

bash scripts/bootstrap_linux_l40s.sh
source ACE-Step-1.5/.venv/bin/activate

python scripts/prepare_release_dataset.py
python scripts/verify_linux_l40s.py --root .
pytest -q
```

`prepare_release_dataset.py` 从
[`fisheryv/open-focus-classical-600`](https://huggingface.co/datasets/fisheryv/open-focus-classical-600)
下载 600 个原始音频，逐个核对发布的 `SHA256SUMS`，再按项目约定硬链接（跨文件系统
时校验后复制）到 `data_raw/focus_music` 和 `data_raw/classical_music`。它不会把音频
提交到 Git。

外部依赖均在 `reproducibility/release_manifest.toml` 冻结：

- `pyglmy`：`49bd5ea7617906f09940dcc9b9718bbfc1482d6f`；
- ACE-Step 1.5：`a5632cda3084f1088e69b2057dde7047e1bb4839`，随后应用仓库内 sampler patch；
- exact scorer：Pitch 16 + Acoustic/Chroma phase `loop_score`，共 18 维；旧 51-D
  运行时输入明确拒绝。

## 正式 LTSN 流水线

先准备 `metadata/ltsn_prompts.csv`（字段与分区规则见
`docs/ltsn-linux-training-and-evaluation.md`），计算 ACE 模型/VAE 目录内容哈希，然后按
阶段运行：

```bash
export ACE_MODEL_SHA256=<64-hex>
export VAE_SHA256=<64-hex>

bash scripts/run_ltsn_pipeline.sh collect
# 完成独立 reranking、质量、prompt 和多样性评价并签发通过的 gate 后：
export RERANKING_GATE=$PWD/metadata/ace_reranking_effect_gate.json
bash scripts/run_ltsn_pipeline.sh labels
bash scripts/run_ltsn_pipeline.sh train
bash scripts/run_ltsn_pipeline.sh calibrate
# 生成并 exact 解码 development 引导对后：
PAIR_TABLE=<development-pairs.csv> bash scripts/run_ltsn_pipeline.sh guidance-development
bash scripts/run_ltsn_pipeline.sh qualify
# 资格通过并完成全新 32 prompt × 8 seed 配对实验后：
PAIR_TABLE=<confirmation-pairs.csv> bash scripts/run_ltsn_pipeline.sh guidance-confirmation
```

单个 LTSN ensemble 目前按三个 seed 顺序训练，并只占用 `TRAIN_DEVICE`（默认
`cuda:0`）；第二张 L40S 可用于独立预计算或后续并行 seed 调度，但不要把“有两张卡”
误报为当前代码已经执行 DDP。完整门禁、存储峰值、断点续跑和最终 32 prompt × 8 seed
配对评估见 [Linux/NVIDIA 指引](docs/ltsn-linux-training-and-evaluation.md)。

## 发布内容与排除项

Git 发布包含源代码、冻结配置/哈希、18-D scorer、ACE patch、测试、论文与复现文档；
不包含 `.env`、原始音频、Hugging Face 下载缓存、ACE checkpoint、运行日志、模型权重
或资格结果。每条音频继续适用数据集 `metadata/licenses.csv` 中各自的许可，不存在单一
数据集总许可。仓库软件代码按根目录 [MIT License](LICENSE) 发布；该软件许可不覆盖
数据集、第三方模型、论文素材或各音频作品。

---

# Focus Topology API 与研究分析

> **Canonical dataset (2026-08-02):** the study now uses a two-group corpus:
> Jamendo Open Focus 300 + Classical 300, with auditable per-track licenses.
> The former Pop comparison group is preserved under
> `dataset_archive/pop_music_legacy_2026-08-02/`; the retired Brain.fm baseline remains under
> `restricted_archive/brainfm_legacy_2026-08-02/` and is not part of current
> metadata, preprocessing, feature fitting, or future confirmatory claims. See
> [the two-group migration report](docs/two-group-dataset-migration.md) and
> [results status](metadata/RESULTS_STATUS.md).

本仓库分为两个层次：

- [`fisheryv/pyglmy`](https://github.com/fisheryv/pyglmy)：领域无关的底层数学库，提供 GLMY Path Homology、Persistent Path Homology 和 Vietoris–Rips TDA；仓库内 `packages/pyglmy` 仅保留发布时的审计镜像；
- `focus_topology`：面向音乐状态序列和音频文件的应用层，负责构图、特征解释和 JSON 结果。

音乐层现在通过兼容适配器调用 `pyglmy`，不再自行维护 Path Homology 或 Ripser 包装算法。

## 独立底层库

只安装 Path Homology：

```bash
python -m pip install "pyglmy @ git+https://github.com/fisheryv/pyglmy.git@49bd5ea7617906f09940dcc9b9718bbfc1482d6f"
```

同时安装点云 TDA 后端：

```bash
python -m pip install "pyglmy[tda] @ git+https://github.com/fisheryv/pyglmy.git@49bd5ea7617906f09940dcc9b9718bbfc1482d6f"
```

底层库的独立 API、CLI、数学边界和示例见
[packages/pyglmy/README.md](packages/pyglmy/README.md)。

## 安装

从当前仓库以可编辑模式安装核心库：

```bash
python -m pip install -e .
```

如果需要直接读取 WAV、FLAC 或其他 librosa/soundfile 支持的音频：

```bash
python -m pip install -e ".[audio,tda]"
```

开发和测试环境：

```bash
python -m pip install -e ".[audio,stats,tda,repro,dev]"
pytest
```

## Python 快速开始

### 分析状态序列

状态可以是任意可哈希的 Python 值；`None` 表示缺失观测。

```python
from focus_topology import AnalysisConfig, analyze_states

states = [0, 1, 2, 0, 1, 2, 0]
config = AnalysisConfig(
    thresholds=(0.95, 0.8, 0.5),
    top_k=6,
)

result = analyze_states(
    states,
    config=config,
    metadata={"track_id": "demo-cycle", "view": "pitch"},
)

print(result.betti_curve(0))
print(result.betti_curve(1))
print(result.metrics["directed_recurrence"])
result.write_json("topology-result.json")
```

返回的 `TopologyAnalysis` 包含：

- `graph`：顶点、边、转移次数与归一化转移概率；
- `persistence`：各阈值描述子、H0/H1 rank invariant 和持久区间；
- `metrics`：熵、复现率、互反性、Betti 曲线汇总等指标；
- `metadata`：调用方传入的曲目、视角或数据集信息；
- `to_dict()`、`to_json()`、`write_json()`：稳定的 JSON 导出接口。

### 直接分析音频

```python
from focus_topology import analyze_audio

result = analyze_audio(
    "example.wav",
    view="pitch",
)

print(result.betti_curve(1))
result.write_json("example.pitch-topology.json")
```

`view="pitch"` 使用节拍同步的主导音级状态。`view="modulation"` 使用单曲内部的三档调制频带分位数，适合单曲探索；跨曲目统计比较应使用同一个 discovery 数据拟合并冻结状态模型，避免每首曲目单独拟合造成口径漂移。

音频入口会自动转为单声道，并重采样到 `FeatureExtractionConfig.sample_rate`（默认 22.05 kHz）；原文件不会被修改。

### 低层算法

```python
from focus_topology import (
    build_transition_graph,
    compute_path_homology,
    persistent_path_homology,
)

graph = build_transition_graph([0, 1, 2, 0, 1, 2])
groups = compute_path_homology(
    graph.vertices,
    graph.edge_pairs,
    max_dimension=1,
)
filtration = persistent_path_homology(graph, [0.95, 0.8, 0.5])
```

## 命令行

分析 JSON 状态序列：

```powershell
focus-topology states examples/states_cycle.json `
  --thresholds 0.95,0.8,0.5 `
  --output topology-result.json
```

直接分析音频：

```powershell
focus-topology audio example.wav `
  --view pitch `
  --output example.pitch-topology.json
```

检查安装能力和运行最小示例：

```powershell
focus-topology doctor
focus-topology demo
```

也可以使用模块方式运行：

```powershell
python -m focus_topology --version
```

## API 稳定边界

推荐从 `focus_topology` 顶层导入以下公开接口：

- `AnalysisConfig`
- `TopologyAnalyzer`
- `TopologyAnalysis`
- `analyze_states`
- `analyze_audio`
- `states_from_audio`
- `TransitionGraph`、`WeightedEdge`、`build_transition_graph`
- `compute_path_homology`、`persistent_path_homology`

历史接口 `focus_topology.pipeline.analyze_state_sequence` 继续保留，但只返回固定阈值描述子。新集成建议使用 `analyze_states`。

完整说明见 [docs/library-api.md](docs/library-api.md)，可运行示例见 [examples/library_quickstart.py](examples/library_quickstart.py) 和 [examples/analyze_audio.py](examples/analyze_audio.py)。

## 研究复现层

Jamendo/FMA 开放 Focus 替代集使用 `focus-open` 构建。筛选规则、FMA 回退及 Jamendo
`mp32` 与实测码率的区别见 `docs/open-focus-dataset.md`。在 300 首下载通过所选码率门槛和
内容审计前，它与已授权的 Brain.fm 基线保持分离。

本仓库还包含专注音乐研究的完整流水线：

```text
data_raw/                 本地音频（不进入版本控制）
metadata/                 许可、曲目索引、数据切分和分析汇总
src/features/             MIR 特征提取与共享状态模型
src/topology/             批量拓扑分析和统计
src/tda/                  连续轨迹 Vietoris–Rips TDA
src/repetition/           重复结构与相位提升 Path Homology
src/generation/           ACE-Step 生成、重排和引导实验
```

典型复现命令：

```powershell
focus-features run --root . --workers 2
focus-path-analysis run --root . --workers 2
focus-tda all --root .
focus-repetition all --root .
```

工程关系见 [docs/architecture.md](docs/architecture.md)，拓扑分析结果见 [docs/topology-analysis-results.md](docs/topology-analysis-results.md)，数据公开边界见 [docs/data-governance.md](docs/data-governance.md)。

## 科研与数据边界

- 本库输出拓扑描述，不构成 ADHD 或其他疾病的诊断、治疗或疗效声明。
- `data_raw/focus_music/` 中的受限音频不得提交、公开或再分发。
- 跨数据集比较应复用同一个状态模型、阈值集合和预处理配置。
- 当前项目许可证仍是研究用途边界；公开发布到包索引前，应由项目所有者补充明确的软件许可证。
