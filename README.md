# Focus Topology

> **Canonical dataset (2026-08-02):** the study now uses a two-group corpus:
> Jamendo Open Focus 300 + Classical 300, with auditable per-track licenses.
> The former Pop comparison group is preserved under
> `dataset_archive/pop_music_legacy_2026-08-02/`; the retired Brain.fm baseline remains under
> `restricted_archive/brainfm_legacy_2026-08-02/` and is not part of current
> metadata, preprocessing, feature fitting, or future confirmatory claims. See
> [the two-group migration report](docs/two-group-dataset-migration.md) and
> [results status](metadata/RESULTS_STATUS.md).

本仓库现在分为两个层次：

- `packages/pathhom_tda`：领域无关的底层数学库，提供 GLMY Path Homology、Persistent Path Homology 和 Vietoris–Rips TDA；
- `focus_topology`：面向音乐状态序列和音频文件的应用层，负责构图、特征解释和 JSON 结果。

音乐层现在通过兼容适配器调用 `pathhom_tda`，不再自行维护 Path Homology 或 Ripser 包装算法。

## 独立底层库

只安装 Path Homology：

```powershell
python -m pip install -e packages/pathhom_tda
```

同时安装点云 TDA 后端：

```powershell
python -m pip install -e "packages/pathhom_tda[tda]"
```

底层库的独立 API、CLI、数学边界和示例见
[packages/pathhom_tda/README.md](packages/pathhom_tda/README.md)。

## 安装

从当前仓库以可编辑模式安装核心库：

```powershell
python -m pip install -e packages/pathhom_tda
python -m pip install -e .
```

如果需要直接读取 WAV、FLAC 或其他 librosa/soundfile 支持的音频：

```powershell
python -m pip install -e "packages/pathhom_tda[tda]"
python -m pip install -e ".[audio]"
```

开发和测试环境：

```powershell
python -m pip install -e "packages/pathhom_tda[dev]"
python -m pip install -e ".[audio,stats,dev]"
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
