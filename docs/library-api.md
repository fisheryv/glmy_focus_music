# `focus_topology` 库接口

> `focus_topology` 是音乐应用层。需要直接处理任意有向图、Path Complex、
> Persistent Path Homology 或点云 Vietoris–Rips TDA 时，请使用独立的
> [`pyglmy`](../packages/pyglmy/README.md) 底层包。

## 1. 输入模型

拓扑内核接收一维离散状态序列。相邻有效状态形成一次有向转移，`None` 会切断相邻转移。状态必须可哈希，例如整数、字符串或元组。

默认情况下：

- 边权是同一源状态下的归一化转移概率；
- 每个源状态最多保留权重最高的 6 条出边；
- 自环会参与描述性序列指标，但不会进入 regular GLMY path complex；
- filtration 阈值按从高到低排列，随着阈值降低只增加边。

## 2. 高层接口

### `AnalysisConfig`

```python
AnalysisConfig(
    thresholds=(0.95, 0.9, 0.8, 0.7, 0.6, 0.5),
    top_k=6,
    include_self_loops=False,
    tolerance=1e-9,
)
```

阈值会自动去重并降序保存。`top_k=None` 表示保留所有出边。

### `analyze_states`

```python
result = analyze_states(states, config=config, metadata=metadata)
```

适合单次调用。需要对很多序列复用相同配置时，可以创建分析器：

```python
from focus_topology import AnalysisConfig, TopologyAnalyzer

analyzer = TopologyAnalyzer(AnalysisConfig(thresholds=(0.9, 0.7, 0.5)))
first = analyzer.analyze(first_states, metadata={"track_id": "first"})
second = analyzer.analyze(second_states, metadata={"track_id": "second"})
```

### `analyze_audio`

音频入口是可选扩展，需要安装 `.[audio]`。

```python
result = analyze_audio(
    "track.wav",
    view="pitch",
    track_id="track-001",
    config=AnalysisConfig(),
)
```

支持的独立分析视角：

- `pitch`：节拍同步的主导音级；不确定或静音窗口记为缺失状态；
- `modulation`：7–9 Hz、12–20 Hz、30–34 Hz 三个关键频带的单曲分位数状态；
- `structure`：声学特征的余弦自相似矩阵（SSM）经棋盘核得到 novelty 曲线和段落边界，再将宏观声学块映射为可复现的高阶状态路径。结果元数据包含 `boundary_seconds`。

`modulation` 的分位数和 `structure` 的复现标签在当前音频上拟合，因此只适合单曲探索。正式组间比较应使用仓库批处理流程，仅以 discovery split 拟合共享状态原型，再冻结应用于 validation/test。

输入音频会在临时目录中自动转为单声道并重采样到特征配置的采样率，原文件不会被改写。

## 3. 返回对象

`TopologyAnalysis` 的主要成员：

| 成员 | 内容 |
|---|---|
| `states` | 实际分析的状态元组 |
| `graph` | `TransitionGraph` |
| `persistence` | `PersistentPathResult` |
| `metrics` | 序列、图和拓扑汇总指标 |
| `config` | 本次使用的冻结配置 |
| `metadata` | 调用方附加信息 |

常用方法：

```python
h0 = result.betti_curve(0)
h1 = result.betti_curve(1)
payload = result.to_dict()
text = result.to_json()
path = result.write_json("result.json")
```

JSON schema 当前版本为 `1`。顶点和边保留原状态值；无法直接表示为 JSON 的自定义状态会退化为 `repr()` 字符串。

## 4. 指标说明

序列和图指标包括：

- 有效状态数、有效转移数、自转移比例；
- 顶点数、边数、边密度、互反性；
- 转移熵、条件 Path Entropy；
- Directed Recurrence 及其无偏估计。

每个 H0/H1 维度还包括：

- Betti 曲线均值、最大值和阈值面积；
- 持久区间计数；
- 观测到的总 persistence；
- 在最低阈值仍未死亡的 censored 区间数。

## 5. 低层接口

历史兼容入口仍可从 `focus_topology` 使用；新代码建议直接依赖底层包：

```python
from pyglmy import WeightedDiGraph, path_homology, persistent_path_homology

graph = WeightedDiGraph.from_edges(
    [(0, 1, 0.9), (1, 2, 0.9), (2, 0, 0.9)]
)
groups = path_homology(graph.vertices, graph.edge_pairs, max_dimension=2)
result = persistent_path_homology(
    graph,
    thresholds=(0.95, 0.8, 0.5),
    max_dimension=2,
)
```

当前有限 filtration 的持久计算返回 H0/H1。单阈值 `compute_path_homology` 可通过 `max_dimension` 请求更高维度，但路径枚举实现优先保证可解释性和交叉验证，不适合未经评估的大规模稠密高维图。

## 6. 可复现比较建议

比较多首曲目时应固定以下内容：

1. 音频采样率、分析窗口和特征配置；
2. 由 discovery 数据拟合的状态模型；
3. `AnalysisConfig` 的阈值、`top_k` 和数值容差；
4. 缺失状态处理与曲目切片长度。

这些条件不同会改变转移图和 persistence，结果不应直接混合解释。
