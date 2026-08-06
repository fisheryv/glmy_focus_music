# 强化相位—状态 Path Homology 可行性结果

生成日期：2026-07-30。仅使用 discovery 数据；由于没有候选通过预设门槛，validation 未用于参数选择或确认性检验。

## 预设配置结果（状态相似度 0.55）

| 表示 | 180s 效应 | p | 300s 效应 | 跨尺度ρ | 增量效应 | 增量p | 入选 |
|---|---:|---:|---:|---:|---:|---:|---|
| phase_state_rhythm__h1_dominant_persistence | 0.033 | 0.426 | 0.267 | 0.514 | 0.118 | 0.245 | 否 |
| phase_state_acoustic__h1_dominant_persistence | 0.010 | 0.479 | -0.170 | 0.600 | 0.056 | 0.375 | 否 |
| phase_state_acoustic__h1_max_lifetime | -0.215 | 0.903 | -0.429 | 0.417 | 0.104 | 0.272 | 否 |
| phase_state_acoustic__h1_normalized_auc | -0.208 | 0.894 | -0.358 | 0.514 | -0.031 | 0.578 | 否 |
| phase_state_rhythm__h1_max_lifetime | -0.153 | 0.822 | 0.226 | 0.336 | 0.281 | 0.0485 | 否 |
| phase_state_rhythm__h1_normalized_auc | -0.247 | 0.93 | -0.031 | 0.423 | 0.208 | 0.11 | 否 |

## 状态粒度敏感性

| 阈值 | 表示 | 180s 效应 | p | 300s 效应 | p | 跨尺度ρ |
|---:|---|---:|---:|---:|---:|---:|
| 0.40 | phase_state_acoustic__h1_max_lifetime | 0.139 | 0.207 | 0.220 | 0.0967 | 0.010 |
| 0.40 | phase_state_acoustic__h1_dominant_persistence | 0.184 | 0.139 | 0.238 | 0.0804 | 0.134 |
| 0.40 | phase_state_acoustic__h1_normalized_auc | 0.111 | 0.258 | 0.399 | 0.00911 | 0.294 |
| 0.40 | phase_state_rhythm__h1_max_lifetime | -0.076 | 0.68 | -0.108 | 0.742 | 0.155 |
| 0.40 | phase_state_rhythm__h1_dominant_persistence | 0.019 | 0.459 | -0.080 | 0.686 | 0.150 |
| 0.40 | phase_state_rhythm__h1_normalized_auc | -0.170 | 0.846 | 0.212 | 0.106 | 0.122 |
| 0.55 | phase_state_acoustic__h1_max_lifetime | -0.215 | 0.903 | -0.429 | 0.995 | 0.417 |
| 0.55 | phase_state_acoustic__h1_dominant_persistence | 0.010 | 0.479 | -0.170 | 0.846 | 0.600 |
| 0.55 | phase_state_acoustic__h1_normalized_auc | -0.208 | 0.894 | -0.358 | 0.984 | 0.514 |
| 0.55 | phase_state_rhythm__h1_max_lifetime | -0.153 | 0.822 | 0.226 | 0.0895 | 0.336 |
| 0.55 | phase_state_rhythm__h1_dominant_persistence | 0.033 | 0.426 | 0.267 | 0.0574 | 0.514 |
| 0.55 | phase_state_rhythm__h1_normalized_auc | -0.247 | 0.93 | -0.031 | 0.578 | 0.423 |
| 0.70 | phase_state_acoustic__h1_max_lifetime | -0.306 | 0.967 | -0.512 | 0.999 | 0.478 |
| 0.70 | phase_state_acoustic__h1_dominant_persistence | 0.073 | 0.336 | -0.108 | 0.742 | 0.650 |
| 0.70 | phase_state_acoustic__h1_normalized_auc | -0.302 | 0.964 | -0.420 | 0.994 | 0.568 |
| 0.70 | phase_state_rhythm__h1_max_lifetime | -0.201 | 0.887 | 0.069 | 0.342 | 0.410 |
| 0.70 | phase_state_rhythm__h1_dominant_persistence | -0.024 | 0.561 | 0.156 | 0.179 | 0.420 |
| 0.70 | phase_state_rhythm__h1_normalized_auc | -0.337 | 0.978 | -0.188 | 0.869 | 0.436 |

## 诊断结论

人工循环与时间打乱校准全部通过，但没有强化 H1 端点同时满足 180秒组间效应、300秒方向一致、跨尺度稳定和非拓扑基线之外的增量门槛。

原始同相位复现仍呈 Focus>Pop；将相位拆成数据驱动状态后，Pop 的集中状态转移会产生同样或更长的单个 H1 环。Focus 图出现更多节点、边和环类，但这些差异可由图规模解释，不能归因于独立的 Path Homology 信息。

因此本次可行性试验不支持把强化相位—状态 Path Homology 扩展到 validation。原固定相位环的声学/节奏结果仍可作为复现一致性指标，但不应宣称其优势来自非平凡的图拓扑。
