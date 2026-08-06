# 循环敏感 Homology / Path Homology 结果

生成日期：2026-07-29。先用 discovery 小样本执行人工循环/时间打乱校准，再冻结候选并在 validation 确认。检验方向预先固定为 Focus 的环分数高于对照。

## 人工循环校准

| 表示 | 方法 | 人工循环 | 时间打乱 | 中位差 | 正向比例 | FDR | 通过 |
|---|---|---:|---:|---:|---:|---:|---|
| path_rhythm_phase | phase_lifted_path_homology | 1.000 | 0.330 | 0.670 | 1.00 | 1.57e-13 | 是 |
| path_acoustic_phase | phase_lifted_path_homology | 1.000 | 0.347 | 0.653 | 1.00 | 1.57e-13 | 是 |
| path_chroma_phase | phase_lifted_path_homology | 1.000 | 0.360 | 0.640 | 1.00 | 1.57e-13 | 是 |
| sw_loudness | sliding_window_homology | 0.672 | 0.148 | 0.515 | 1.00 | 1.57e-13 | 是 |
| sw_onset | sliding_window_homology | 0.662 | 0.156 | 0.507 | 0.97 | 1.75e-13 | 是 |
| sw_modulation | sliding_window_homology | 0.434 | 0.139 | 0.297 | 0.83 | 2.59e-11 | 是 |
| sw_acoustic_novelty | sliding_window_homology | 0.396 | 0.144 | 0.258 | 0.99 | 1.57e-13 | 是 |
| sw_tonal_novelty | sliding_window_homology | 0.294 | 0.148 | 0.152 | 0.94 | 2.48e-13 | 是 |

## Discovery 小样本筛选

| 表示 | 180s 效应 | 单侧 p | 300s 效应 | 跨尺度 Spearman | 入选 |
|---|---:|---:|---:|---:|---|
| path_rhythm_phase | 0.441 | 0.00455 | 0.497 | 0.887 | 是 |
| path_acoustic_phase | 0.392 | 0.0102 | 0.358 | 0.829 | 是 |
| path_chroma_phase | 0.111 | 0.258 | 0.243 | 0.726 | 否 |
| sw_acoustic_novelty | -0.260 | 0.94 | -0.566 | 0.360 | 否 |
| sw_loudness | -0.156 | 0.826 | -0.451 | 0.615 | 否 |
| sw_modulation | -0.441 | 0.996 | -0.194 | 0.399 | 否 |
| sw_onset | -0.163 | 0.836 | -0.177 | 0.689 | 否 |
| sw_tonal_novelty | -0.316 | 0.97 | 0.052 | 0.314 | 否 |

## 冻结表示

- `path_rhythm_phase`
- `path_acoustic_phase`

全量清单中 24 个片段未达到时序长度质量门槛，已在统计前排除。

## Focus vs Pop 确认

| 表示 | Focus 180s | Pop 180s | 180s 效应 | 180s FDR | 300s 效应 | 300s FDR |
|---|---:|---:|---:|---:|---:|---:|
| path_acoustic_phase | 0.448 | 0.396 | 0.355 | 0.00194 | 0.440 | 0.000123 |
| path_rhythm_phase | 0.411 | 0.388 | 0.283 | 0.00667 | 0.322 | 0.00249 |

## Classical 特异性复核

| 表示 | Focus | Classical | 效应 | FDR |
|---|---:|---:|---:|---:|
| path_acoustic_phase | 0.448 | 0.340 | 0.817 | 3.88e-13 |
| path_rhythm_phase | 0.411 | 0.348 | 0.723 | 6.75e-11 |

## 分类辅助结果

| 任务 | Macro-F1 | Balanced accuracy | AUROC |
|---|---:|---:|---:|
| three_class | 0.503 | 0.563 | 0.720 |
| focus_vs_pop | 0.637 | 0.661 | 0.682 |

## 解释

phase-lifted 图使用 6 个相位节点形成有向环；边权是相隔一个主周期的同相位复现一致性。`loop_score` 是该环最弱边的权重，也就是 H1 在边权过滤中首次完整出现的临界尺度。普通滑动窗口方法只有在延迟嵌入产生长寿命 H1 时才得到高分。

该结果描述数据集中的可听重复结构，不构成注意力作用的因果证据。
180秒 Focus/Pop 主确认支持：`path_acoustic_phase`、`path_rhythm_phase`。
`path_chroma_phase` 未通过 discovery 冻结门槛，因此没有进入确认性检验。

## 图形输出

- `runs/repetition_homology/loop_scores_validation.png`
- `runs/repetition_homology/loop_scores_validation.svg`
- `runs/repetition_homology/path_h1_filtration_validation.png`
- `runs/repetition_homology/path_h1_filtration_validation.svg`
