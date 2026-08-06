# TDA 分析结果

生成日期：2026-08-01。候选表示只在 discovery 小样本中筛选；validation / 180 秒是确认分析，validation / 300 秒是尺度复核。

## 冻结的 TDA 端点

- `acoustic_novelty_delay / h0_max_persistence`
- `rhythm / h0_total_persistence`
- `acoustic_pca / h0_persistence_entropy`

所有点云均固定为 24 个时间均匀采样点，并以点间距离中位数归一化；因此持久性值主要描述形状，而不是原始特征振幅。使用 Vietoris–Rips filtration，计算 Z/2 上的 H0/H1。
全量特征清单中有 2 个片段未达到至少 24 个时间点的质量门槛，已在统计分析前排除。

## Focus vs Pop 确认结果

validation / 180 秒共有 3 个 FDR q≤0.05 的端点；其中 2 个在 validation / 300 秒也显著且方向一致。

| 表示 | 特征 | Focus 180s | Pop 180s | 180s 效应 | 180s FDR | 300s 效应 | 300s FDR |
|---|---|---:|---:|---:|---:|---:|---:|
| acoustic_pca | h0_persistence_entropy | 0.9837 | 0.9544 | 0.674 | 8.3e-09 | 0.864 | 7.07e-14 |
| acoustic_novelty_delay | h0_max_persistence | 0.9640 | 1.3103 | -0.550 | 1.82e-06 | -0.660 | 8.67e-09 |
| rhythm | h0_total_persistence | 9.6785 | 10.4986 | -0.262 | 0.021 | -0.219 | 0.0534 |

## Classical 特异性复核

在 validation / 180 秒，2 个端点同时显著区分 Focus–Pop 与 Focus–Classical。

| 表示 | 特征 | Focus 180s | Classical 180s | 180s 效应 | 180s FDR | 300s 效应 | 300s FDR |
|---|---|---:|---:|---:|---:|---:|---:|
| acoustic_novelty_delay | h0_max_persistence | 0.9640 | 1.2599 | -0.498 | 1.45e-05 | -0.567 | 1.41e-06 |
| rhythm | h0_total_persistence | 9.6785 | 10.7530 | -0.510 | 1.45e-05 | -0.382 | 0.00105 |
| acoustic_pca | h0_persistence_entropy | 0.9837 | 0.9821 | 0.051 | 0.655 | 0.351 | 0.00181 |

## 拓扑解释

- acoustic novelty 延迟嵌入的 H0 最大持久性更低：在距离尺度归一化后，Focus 的新颖度动态缺少特别孤立、需要很大半径才合并的状态簇。
- rhythm 的 H0 总持久性更低：固定 24 点时它等价于更短的最小生成树总长度，说明 Focus 的节奏状态几何更紧凑、碎片化更少。
- acoustic PCA 的 H0 持久性熵更高：连通分支的合并尺度更均匀；但该端点在主尺度不能区分 Focus 与 Classical。
- 没有 H1 端点进入冻结特征集；当前证据指向连通分支的多尺度几何，而不是稳定环洞。

## TDA-only 分类

| 任务 | Macro-F1 | Balanced accuracy | AUROC |
|---|---:|---:|---:|
| three_class | 0.548 | 0.592 | 0.789 |
| focus_vs_pop | 0.690 | 0.763 | 0.864 |

## 图形输出

- `runs/tda/selected_features_validation.png`
- `runs/tda/selected_features_validation.svg`

## 解释边界

acoustic novelty delay 与 rhythm 的 H0 端点通过了 Pop 和 Classical 双对照复核；acoustic PCA H0 熵在主尺度不能区分 Focus 与 Classical，因此不能视为 Focus 特异。这些结果不等于注意力提升或神经机制的因果证据。筛选和确认使用分离的数据划分；300 秒结果仅用于尺度稳健性复核。
