# Path Homology 结构视角：Focus–Classical 完整分析

生成日期：2026-08-04。切分版本：`symmetric_holdout_v2`。本文使用当前规范数据集 Open Focus 300 与 Classical 300；每组 discovery/validation/holdout 分别为 195/60/45 首。结构状态模型仅在两组 discovery/180 s 上拟合；本专项的主检验固定为 validation/180 s（n=120：Classical 60、Open Focus 60），validation/300 s 仅作同曲目时长敏感性。holdout 是哈希门控后的单次操作性最终确认，但 Classical holdout 在旧切分中曾属于 discovery，并非 pristine 外部复制集。结构视角是原三视角确认性家族之外的扩展，因此其证据层级保持为探索性验证。本版在该层级内同样采用统一的 BH-FDR $q\le0.05$；原 holdout 的 $q\le0.10$ 门控作为历史记录保留。

## 1. 结论摘要

- 1,200/1,200 个结构片段成功完成有向图与持续 Path Homology，失败 0；状态模型 SHA-256 为 `a0dc5f846e8f7121293a9ff7e9386aad9d5c9377b5e76ae48bd67da2d5322b0d`。
- validation/180 s 的 20 个预设结构指标中，6 个通过 BH-FDR $q\le0.05$；validation/300 s 有 4 个通过，其中 4 个在两种时长均显著且方向一致。
- 通过主分析 FDR 的指标为 edge_density、reciprocity、path_entropy、self_transition_ratio、transition_entropy、directed_recurrence；它们描述两组在共享宏观状态码本中的覆盖、转移方向和连通过程，不表示曲式质量或复杂度高低。
- $H_1$ 高度零膨胀：validation/180 s 非零曲目为 Classical 8/60、Open Focus 5/60；不得把扩展阈值个例改写成主分析中的普遍结构环。
- holdout/180 s 的结构块整体表示 pseudo-$F=2.829$、$p=0.005$、跨次级视角 BH $q=0.005$。原门控的 6 个方向指标中，5/6 方向一致；历史 $q\le0.10$ 有 4/6、严格 $q\le0.05$ 有 3/6 复现，属于部分而非完整复制。
- 当前拓扑输入 SHA-256 为 `192647177a7a72e77f74265c54260bb741cba2d961799bda7195366d9242f882`，与 holdout gate **一致**；没有在重跑后改阈值、重选指标或调参。
- 结论属于观察性声学结构比较，不支持注意力、治疗、认知、生成质量或因果结论。

## 2. 结构视角的构建思想

音高、节奏和调制视角描述局部状态，而结构视角把音乐表示为“宏观声学段落的有向演化”。它先比较不同时刻的声学纹理，利用自相似矩阵定位变化边界，再把每个段落映射到共享结构原型。这样得到的路径不是逐帧音色序列，而是 A→B→A、A→B→C 等高阶段落组织。

### 2.1 自相似矩阵

对第 $i$ 个短时声学向量 $\mathbf{x}_i\in\mathbb{R}^d$ 作稳健标准化：

$$
\mathbf{z}_i=\frac{\mathbf{x}_i-\operatorname{med}(\mathbf{x})}{1.4826\,\operatorname{MAD}(\mathbf{x})+\varepsilon},
\qquad
\mathbf{u}_i=\frac{[\mathbf{z}_i,1]}{\lVert[\mathbf{z}_i,1]\rVert_2}.
$$

声学自相似矩阵为

$$
S_{ij}=\frac{1+\mathbf{u}_i^\mathsf{T}\mathbf{u}_j}{2},\qquad S_{ij}\in[0,1].
$$

块状对角结构表示相似段落，块之间的突变提示结构边界。

### 2.2 Foote 棋盘 novelty 与边界

令 $L_t=[t-h,t)$、$R_t=[t,t+h)$，棋盘 novelty 为

$$
\nu(t)=\frac{1}{2h^2}\left(
\sum_{i,j\in L_t}S_{ij}+\sum_{i,j\in R_t}S_{ij}
-\sum_{i\in L_t,j\in R_t}S_{ij}-\sum_{i\in R_t,j\in L_t}S_{ij}
\right).
$$

实现固定使用 8 s 核、$\operatorname{median}(\nu)+1.5\operatorname{MAD}(\nu)$ 峰值阈值，并把段长约束在 8–45 s。边界 $0=b_0<\cdots<b_K=T$ 把音频划分为 $K$ 个宏观块。

### 2.3 高阶状态码本

第 $k$ 块的声学向量为

$$
\mathbf{q}_k=\frac{1}{|I_k|}\sum_{i\in I_k}\mathbf{x}_i.
$$

仅用 discovery/180 s 的 Focus/Classical 平衡数据拟合稳健标准化、32 维 PCA 和 16 个 MiniBatch K-means 原型 $\mathbf{c}_m$。状态分配为

$$
s_k=\arg\min_m\left\lVert\mathbf{P}\mathbf{D}^{-1}(\mathbf{q}_k-\boldsymbol{\mu})-\mathbf{c}_m\right\rVert_2^2.
$$

本轮共得到 10,983 个宏观块和 12,183 个边界；16 个原型均被使用。

## 3. 有向图与 Path Homology

对状态路径 $(s_0,\ldots,s_K)$，定义相邻转移计数和条件概率：

$$
C_{uv}=|\{k:s_k=u,\ s_{k+1}=v\}|,
\qquad p_{uv}=\frac{C_{uv}}{\sum_w C_{uw}}.
$$

自环用于 `self_transition_ratio`，但不进入 Path Homology 图。每个源状态最多保留 top-6 非自环边。过滤图 $G_\tau$ 保留 $p_{uv}\ge\tau$ 的边；阈值从 0.95 降至 0.05 时只增加边：

$$
G_{0.95}\subseteq G_{0.90}\subseteq\cdots\subseteq G_{0.05}.
$$

对允许的 $p$-路径 $e_{v_0\ldots v_p}$，GLMY 边界算子为

$$
\partial e_{v_0\ldots v_p}=\sum_{i=0}^p(-1)^i e_{v_0\ldots\widehat{v_i}\ldots v_p}.
$$

令 $\Omega_p=\{a\in A_p:\partial a\in A_{p-1}\}$，则

$$
H_p^{\mathrm{path}}(G)=
\frac{\ker(\partial_p|_{\Omega_p})}{\operatorname{im}(\partial_{p+1}|_{\Omega_{p+1}})},
\qquad \beta_p=\dim H_p^{\mathrm{path}}(G).
$$

持久图和 barcode 使用递增坐标 $a=1-\tau$。对 $a_i\le a_j$，持久秩不变量为

$$
\rho_p(a_i,a_j)=\operatorname{rank}\operatorname{im}
\left[H_p(G_{a_i})\longrightarrow H_p(G_{a_j})\right].
$$

$H_0$ 表示有向可达结构的连通变化，$H_1$ 表示不能由允许 2-路径边界填充的有向一维类。生产流程只报告 $H_0/H_1$，没有计算 $H_2$。

## 4. 可视化与代表样本

### 4.1 稳健代表样本选择

代表曲目只用于可视化，不参与假设检验。对 validation/180 s 中每一组，在 9 个预设描述子上计算稳健组中心。若 $\mathbf{m}$ 为组中位数、$r_j=1.4826\operatorname{MAD}_j$，并令 $\widetilde r_j=r_j$（当 $r_j>10^{-9}$）否则 $\widetilde r_j=1$，选择

$$
i^*=\arg\min_i\sum_j\left(\frac{x_{ij}-m_j}{\widetilde r_j}\right)^2.
$$

这避免人工挑选“最像预期”的图。选择记录保存在 `metadata/structure_representative_selection.csv`。

| 组别 | 代表片段 | 稳健距离 | 顶点 | 边 | 自转移率 | 路径熵 |
|---|---|---:|---:|---:|---:|---:|
| Classical | `classical_musopen_9e03803778c4__180s` | 0.188 | 2 | 2 | 0.333 | 0.462 |
| Open Focus | `focus_jamendo_1156386__180s` | 0.432 | 3 | 2 | 0.714 | 0.198 |

### 4.2 码本、单曲结构轨迹与有向图

![structure_codebook](../runs/structure_path_homology_open/structure_codebook.png)

[SVG](../runs/structure_path_homology_open/structure_codebook.svg)

![structure_ssm](../runs/structure_path_homology_open/structure_ssm.png)

[SVG](../runs/structure_path_homology_open/structure_ssm.svg)

![structure_directed_state_graph](../runs/structure_path_homology_open/structure_directed_state_graph.png)

[SVG](../runs/structure_path_homology_open/structure_directed_state_graph.svg)

### 4.3 两组稳健代表

![structure_representative_state_graphs](../runs/structure_path_homology_open/structure_representative_state_graphs.png)

[SVG](../runs/structure_path_homology_open/structure_representative_state_graphs.svg)

![structure_representative_ssm](../runs/structure_path_homology_open/structure_representative_ssm.png)

[SVG](../runs/structure_path_homology_open/structure_representative_ssm.svg)

### 4.4 持久 Path Homology 过程

validation/180 s 的扩展阈值中没有有限 H1 区间，因此不挑选循环机制个例。图中使用冻结的 Focus 稳健代表 `focus_jamendo_1156386__180s` 展示 0.95、0.50、0.05 三个阈值和其 persistence/barcode；这是一张负结果说明图，不参与检验。

![structure_filtration_process](../runs/structure_path_homology_open/structure_filtration_process.png)

[SVG](../runs/structure_path_homology_open/structure_filtration_process.svg)

![structure_persistence_diagram](../runs/structure_path_homology_open/structure_persistence_diagram.png)

[SVG](../runs/structure_path_homology_open/structure_persistence_diagram.svg)

![structure_barcode](../runs/structure_path_homology_open/structure_barcode.png)

[SVG](../runs/structure_path_homology_open/structure_barcode.svg)

### 4.5 组间分布、Betti 曲线与尺度敏感性

![structure_group_summary](../runs/structure_path_homology_open/structure_group_summary.png)

[SVG](../runs/structure_path_homology_open/structure_group_summary.svg)

![structure_betti_curves](../runs/structure_path_homology_open/structure_betti_curves.png)

[SVG](../runs/structure_path_homology_open/structure_betti_curves.svg)

![structure_scale_sensitivity](../runs/structure_path_homology_open/structure_scale_sensitivity.png)

[SVG](../runs/structure_path_homology_open/structure_scale_sensitivity.svg)

![structure_effect_sizes](../runs/structure_path_homology_open/structure_effect_sizes.png)

[SVG](../runs/structure_path_homology_open/structure_effect_sizes.svg)

![structure_duration_stability](../runs/structure_path_homology_open/structure_duration_stability.png)

[SVG](../runs/structure_path_homology_open/structure_duration_stability.svg)

## 5. 组间结果

Kruskal–Wallis 检验在 20 个预设结构指标内作 BH-FDR，判定统一要求 $q\le0.05$。若秩和统计量为 $H$、组数为 $k$、总样本为 $N$，效应量为

$$
\epsilon^2=\frac{H-k+1}{N-k}.
$$

两组情况下另报告 Mann–Whitney rank-biserial。holdout 的整体结构块使用 discovery 拟合的秩正态 Mahalanobis 距离与 999 次标签置换，其 pseudo-$F$ 为

$$
F^*=\frac{SS_{between}/(g-1)}{SS_{within}/(N-g)}.
$$

| 指标 | Classical 中位数 | Open Focus 中位数 | $\epsilon^2$ | 180 s FDR | 300 s FDR |
|---|---:|---:|---:|---:|---:|
| edge_density | 1 | 0.375 | 0.228 | 2.51e-06 | 9.22e-07 |
| reciprocity | 1 | 0 | 0.185 | 1.75e-05 | 1.77e-06 |
| path_entropy | 0.481 | 0.375 | 0.063 | 0.0197 | 0.0042 |
| self_transition_ratio | 0.388 | 0.536 | 0.0617 | 0.0197 | 0.00566 |
| transition_entropy | 0.961 | 0.896 | 0.0586 | 0.0197 | 0.103 |
| directed_recurrence | 0.281 | 0.333 | 0.0439 | 0.043 | 0.0946 |
| edge_count | 2 | 2 | 0.0163 | 0.25 | 0.103 |
| h0_censored_count | 1 | 1 | 6.41e-15 | 0.59 | 0.735 |
| h1_betti_auc | 0 | 0 | 0 | 0.59 | 0.0999 |
| h1_betti_max | 0 | 0 | 0 | 0.59 | 0.0999 |
| h1_betti_mean | 0 | 0 | 0 | 0.59 | 0.0999 |
| h1_censored_count | 0 | 0 | 0 | 0.59 | 0.0999 |
| h1_interval_count | 0 | 0 | 0 | 0.59 | 0.0999 |
| h0_betti_auc | 0.45 | 0.45 | 0 | 0.86 | 0.623 |
| h0_betti_max | 1 | 1 | 0 | 0.86 | 0.51 |
| h0_betti_mean | 1 | 1 | 0 | 0.86 | 0.623 |
| h0_interval_count | 1 | 1 | 0 | 0.86 | 0.51 |
| h0_observed_persistence | 0.45 | 0.45 | 0 | 0.86 | 0.623 |
| h1_observed_persistence | 0 | 0 | 0 | 0.86 | 0.623 |
| vertex_count | 2 | 3 | 0 | 0.86 | 0.792 |

Open Focus 与 Classical 的完整独立两两检验：

| 指标 | 比较 | rank-biserial（前者−后者） | FDR |
|---|---|---:|---:|
| edge_density | Classical − Open Focus | 0.546 | 2.55e-06 |
| reciprocity | Classical − Open Focus | 0.48 | 1.77e-05 |
| path_entropy | Classical − Open Focus | 0.307 | 0.0198 |
| self_transition_ratio | Classical − Open Focus | -0.304 | 0.0198 |
| transition_entropy | Classical − Open Focus | 0.297 | 0.0198 |
| directed_recurrence | Classical − Open Focus | -0.263 | 0.0433 |
| edge_count | Classical − Open Focus | 0.176 | 0.252 |
| h0_censored_count | Classical − Open Focus | -0.0167 | 0.594 |
| h1_betti_auc | Classical − Open Focus | 0.0497 | 0.594 |
| h1_betti_max | Classical − Open Focus | 0.05 | 0.594 |
| h1_betti_mean | Classical − Open Focus | 0.0497 | 0.594 |
| h1_censored_count | Classical − Open Focus | 0.05 | 0.594 |
| h1_interval_count | Classical − Open Focus | 0.05 | 0.594 |
| h0_betti_auc | Classical − Open Focus | -0.0208 | 0.863 |
| h0_betti_max | Classical − Open Focus | -0.0128 | 0.863 |
| h0_betti_mean | Classical − Open Focus | -0.0208 | 0.863 |
| h0_interval_count | Classical − Open Focus | -0.0128 | 0.863 |
| h0_observed_persistence | Classical − Open Focus | -0.0189 | 0.863 |
| h1_observed_persistence | Classical − Open Focus | 0.0167 | 0.863 |
| vertex_count | Classical − Open Focus | 0.0475 | 0.863 |

### 5.1 解读

1. **跨时长稳定性。** 只有同时通过 validation/180 s FDR、在 300 s 方向一致并再次显著的指标才视为跨时长稳定；本轮共有 4 项。
2. **状态空间解释。** 边密度、互惠性与自转移率只描述两组在同一 16 原型空间中的组织，不等于曲式标签、复杂度或质量。
3. **$H_1$ 不支持稳定组间差异。** 主尺度非零比例为 Classical 13.3%、Open Focus 8.3%；敏感阈值下为 9/60 和 6/60。holdout/180 s 主阈值为 Classical 7/45、Open Focus 4/45，扩展阈值为 10/45 与 4/45。主分析六个 $H_1$ 汇总量均未通过 FDR；300 s 中若干接近阈值的结果没有主尺度支持，而且所有 validation/180 s $H_1$ 区间均为右删失或不存在，没有有限 birth–death 区间。
4. **观察性边界。** 任何显著声学结构差异都不能推出注意力、治疗、生成质量或因果效应。

## 6. 证据层级与局限

- **探索性验证：** 结构视角并非原三视角确认性家族；本轮在该扩展内部固定 validation/180 s、表示、阈值和 20 指标 FDR，并按统一标准要求 $q\le0.05$。
- **指纹入选：** $q\le0.05$ 是必要条件，但结构还必须对 $L+P$ 提供预定的正融合增量；当前未满足该增量条件，因此结构端点不进入主要冻结指纹，只保留为宏观辅助描述。
- **操作性最终确认：** holdout/180 s 的整体结构块显著，但 6 个原锁定方向指标仅 5 个同方向；历史 $q\le0.10$ 为 4/6、严格 $q\le0.05$ 为 3/6 联合 FDR 复现。Classical holdout 也不是 pristine 外部样本。
- **敏感性：** validation/300 s；4 个主尺度差异再次显著且方向一致。`directed_recurrence` 的 300 s $q=0.0946$ 现降为提示性结果，300 s 新出现的 $H_1$ 边缘结果不能替代主分析。
- **说明性：** 两个稳健组中心代表图、SSM，以及无有限 $H_1$ 区间时的过滤负结果示例。
- **不支持：** 任何稳定或 Focus 特异的 $H_1$；本轮未计算 $H_2$，因此不作 $H_2$ 发现声明；注意力提升、ADHD 疗效、生成效果或因果机制；将已归档三组结论转移到当前两组数据。
- 结构码本描述的是声学段落原型，不等同于音乐学上的主歌、副歌或奏鸣曲式标签。
- 边界检测和 16 状态量化仍可能压缩弱渐变结构；未来应做边界扰动、状态数和 top-k 稳定性分析，但不得据此调参追求显著性。

## 7. 复现入口与产物

```powershell
$env:PYTHONPATH = "packages/pyglmy/src;src"
python scripts/rerun_structure_path_homology.py
python scripts/analyze_structure_results.py
python scripts/render_structure_open_report.py
```

主要数值文件：

- `metadata/structure_topology_segments.csv`
- `metadata/structure_topology_filtration.csv`
- `metadata/structure_topology_filtration_sensitivity.csv`
- `metadata/structure_statistical_tests.csv`
- `metadata/structure_pairwise_tests.csv`
- `metadata/structure_analysis_summary.json`
- `metadata/structure_representative_selection.csv`

## 8. 参考文献

1. Foote, J. (2000). Automatic Audio Segmentation Using a Measure of Audio Novelty. ICME.
2. Müller, M. (2015). *Fundamentals of Music Processing*. Springer.
3. Grigor'yan, A., Lin, Y., Muranov, Y., & Yau, S.-T. (2012). Homologies of path complexes and digraphs.
4. Chowdhury, S., & Mémoli, F. (2018). Persistent path homology of directed networks. SODA.
