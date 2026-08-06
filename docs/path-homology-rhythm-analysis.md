# Path Homology 节奏视角：Focus–Classical 完整分析

生成日期：2026-08-06。本文使用当前规范数据集 Jamendo Open Focus 300 与 Classical 300。两组均分为 discovery 195、validation 60、holdout 45；每首有 180 s 与 300 s 两个片段，共 1,200 个片段。主推断固定为 validation/180 s（n=120：每组 60）；validation/300 s 仅作时长敏感性。holdout 是既有哈希门控后的单次操作性确认，不用于重新选择参数或指标。本版将确认性端点统一为 BH-FDR $q\le0.05$；原 holdout 的 $q\le0.10$ 门控作为历史记录保留。

## 1. 结论摘要

- 1,200/1,200 个节奏片段完成有向图和持续 Path Homology，失败 0。本轮复用开盲前冻结的 discovery-only 状态模型，不在 holdout 后重新拟合；模型 SHA-256 为 `a0dc5f846e8f7121293a9ff7e9386aad9d5c9377b5e76ae48bd67da2d5322b0d`。
- validation/180 s 的 20 个预设 rhythm 指标中，14 个通过 omnibus BH-FDR $q\le0.05$；validation/300 s 有 14 个通过，其中 14 个在两种时长均显著且方向一致。
- 既有单次 holdout/180 s 确认中，rhythm 表示的 permutation pseudo-$F=5.342$、$p=0.001$、次级家族 FDR $q=0.00167$；原门控的 14 个指标中 14 个方向一致、按历史 $q\le0.10$ 有 14 个复现，按统一严格口径 $q\le0.05$ 有 12 个复现。
- 稳定差异集中在状态覆盖、边数、路径熵、有向复现度和 $H_0$ 连通过程；这些量描述量化状态空间，不表示音乐质量高低。
- $H_1$ 高度零膨胀：主阈值非零为 Classical 1/60、Open Focus 1/60；0/6 个预设 $H_1$ 指标通过主分析 FDR。
- 结论属于观察性声学结构比较；不支持注意力、治疗、认知、生成质量或任何因果结论。

## 2. 节奏状态表示

音频为 22,050 Hz 单声道。固定使用 1 s 窗、0.5 s 步长，对第 $n$ 个窗口构造八维向量

$$
\mathbf r_n=[\mu_o,\sigma_o,\max_o,\rho_o,\mu_{IOI},\sigma_{IOI},BPM,\rho_b]^\mathsf T,
$$

依次表示起音包络均值、标准差、最大值、起音率、起音间隔均值与标准差、局部速度和拍点率。事件不足时相应维度记为缺失，不强制设零。

在窗口 $W_n=[t_n,t_n+\Delta)$ 内，若起音时刻为 $a_i$、拍点时刻为 $b_i$，则主要事件统计量为

$$
\rho_o=\frac{N_o(W_n)}{\Delta},\qquad
\mu_{IOI}=\operatorname{mean}(a_{i+1}-a_i),\qquad
BPM=\frac{60}{\operatorname{median}(b_{i+1}-b_i)},\qquad
\rho_b=\frac{N_b(W_n)}{\Delta}.
$$

所有填补、均值、尺度和聚类仅在 discovery/180 s 上拟合。缺失值用 discovery 中位数 $m_j$ 填补，再标准化：

$$
r'_{n,j}=\begin{cases}r_{n,j},&\text{有效},\\m_j,&\text{缺失},\end{cases}
\qquad
\widetilde r_{n,j}=\frac{r'_{n,j}-\mu_j}{\sigma_j}.
$$

仅使用 discovery/180 s，分别从 Classical 与 Focus 平衡抽样 50,000 和 50,000 个窗口，拟合固定 $V_{rhythm}=10$ 的 MiniBatch K-means：

$$
s_n=\arg\min_{v\in\{0,\ldots,9\}}\|\widetilde{\mathbf r}_n-\boldsymbol\mu_v\|_2^2.
$$

单曲顶点集仅包含实际观察到的原型；全局 10 状态不会作为未出现的孤立点补入。

## 3. 有向图与持续 Path Homology

相邻窗口状态定义转移计数与条件概率：

$$
C_{uv}=|\{n:s_n=u,s_{n+1}=v\}|,\qquad
p_{uv}=\frac{C_{uv}}{\sum_w C_{uw}}.
$$

自转移用于描述统计，但不进入 Path Homology 图；每个源状态最多保留 top-6 非自环边。主阈值冻结为 $\tau\in\{0.50,0.60,0.70,0.80,0.90,0.95\}$，扩展至 0.05 仅用于敏感性和机制图：

$$
G_\tau=(V,\{(u,v):u\ne v,\ p_{uv}\ge\tau\}).
$$

对允许路径空间 $\Omega_p$，GLMY 路径同调为

$$
\partial e_{v_0\ldots v_p}=\sum_i(-1)^i e_{v_0\ldots\widehat{v_i}\ldots v_p},\qquad
\Omega_p=A_p\cap\partial^{-1}(A_{p-1}),\qquad
H_p^{path}(G)=\frac{\ker(\partial_p|_{\Omega_p})}{\operatorname{im}(\partial_{p+1}|_{\Omega_{p+1}})}.
$$

其中 $A_p$ 由允许的有向 $p$-路径张成，$\beta_p=\dim H_p^{path}$。阈值下降时边逐步加入，并计算秩不变量

$$
\rho_p(\tau_i,\tau_j)=\operatorname{rank}\operatorname{im}\left[H_p(G_{\tau_i})\to H_p(G_{\tau_j})\right],\qquad \tau_i\ge\tau_j.
$$

条形码和持续图使用单调坐标 $a=1-\tau$。

节奏主分析直接使用冻结状态路径，不构造 SSM。报告中的 SSM 只是轨迹质量诊断，移除它不会改变任何图、barcode 或统计数值。

## 4. 可视化

机制示例 `focus_jamendo_1901339__180s` 按冻结的说明性规则选自当前 Open Focus validation/180 s：优先选择 Focus 中恰有一个有限 $H_1$ 区间的片段，再按 lifetime 最大及 segment ID 确定性排序。其敏感阈值区间在 $\tau=0.4$ 出生、$\tau=0.2$ 死亡；它不代表组中心，也不参与假设检验。

![rhythm_codebook](../runs/rhythm_path_homology_open/rhythm_codebook.png)

[SVG](../runs/rhythm_path_homology_open/rhythm_codebook.svg)

![rhythm_ssm](../runs/rhythm_path_homology_open/rhythm_ssm.png)

[SVG](../runs/rhythm_path_homology_open/rhythm_ssm.svg)

![rhythm_directed_state_graph](../runs/rhythm_path_homology_open/rhythm_directed_state_graph.png)

[SVG](../runs/rhythm_path_homology_open/rhythm_directed_state_graph.svg)

![rhythm_filtration_process](../runs/rhythm_path_homology_open/rhythm_filtration_process.png)

[SVG](../runs/rhythm_path_homology_open/rhythm_filtration_process.svg)

![rhythm_persistence_diagram](../runs/rhythm_path_homology_open/rhythm_persistence_diagram.png)

[SVG](../runs/rhythm_path_homology_open/rhythm_persistence_diagram.svg)

![rhythm_barcode](../runs/rhythm_path_homology_open/rhythm_barcode.png)

[SVG](../runs/rhythm_path_homology_open/rhythm_barcode.svg)

![rhythm_group_summary](../runs/rhythm_path_homology_open/rhythm_group_summary.png)

[SVG](../runs/rhythm_path_homology_open/rhythm_group_summary.svg)

![rhythm_betti_curves](../runs/rhythm_path_homology_open/rhythm_betti_curves.png)

[SVG](../runs/rhythm_path_homology_open/rhythm_betti_curves.svg)

![rhythm_scale_sensitivity](../runs/rhythm_path_homology_open/rhythm_scale_sensitivity.png)

[SVG](../runs/rhythm_path_homology_open/rhythm_scale_sensitivity.svg)

![rhythm_effect_sizes](../runs/rhythm_path_homology_open/rhythm_effect_sizes.png)

[SVG](../runs/rhythm_path_homology_open/rhythm_effect_sizes.svg)

![rhythm_duration_stability](../runs/rhythm_path_homology_open/rhythm_duration_stability.png)

[SVG](../runs/rhythm_path_homology_open/rhythm_duration_stability.svg)

## 5. 组间结果

Kruskal–Wallis 检验在 20 个预设 rhythm 指标内作 BH-FDR，确认性判定统一要求 $q\le0.05$，效应量为 $\epsilon^2$。

$$
\epsilon^2=\frac{H-k+1}{N-k},
$$

其中 $H$ 为 Kruskal–Wallis 统计量、$k=2$ 为组数。独立两两表使用 Mann–Whitney $U$ 与 rank-biserial 效应；300 s 使用同一检验，仅作为同曲目的时长敏感性，不称为独立复制。

| 指标 | Classical 中位数 | Open Focus 中位数 | $\epsilon^2$ | 180 s BH-FDR $q$ | 300 s BH-FDR $q$ |
|---|---:|---:|---:|---:|---:|
| path_entropy | 1.336 | 0.866 | 0.390 | 1.38e-10 | 1.06e-10 |
| edge_count | 36.000 | 21.000 | 0.356 | 5.48e-10 | 3.06e-10 |
| directed_recurrence | 0.092 | 0.265 | 0.311 | 5.47e-09 | 1.78e-08 |
| h0_betti_mean | 6.833 | 4.750 | 0.299 | 8.56e-09 | 4.62e-09 |
| h0_betti_auc | 3.100 | 2.175 | 0.288 | 1.35e-08 | 4.62e-09 |
| h0_observed_persistence | 3.300 | 2.275 | 0.276 | 2.02e-08 | 4.62e-09 |
| transition_entropy | 0.792 | 0.611 | 0.276 | 2.02e-08 | 4.82e-07 |
| h0_betti_max | 8.000 | 6.000 | 0.256 | 5.13e-08 | 2.5e-08 |
| h0_interval_count | 8.000 | 6.000 | 0.256 | 5.13e-08 | 2.5e-08 |
| self_transition_ratio | 0.479 | 0.662 | 0.252 | 5.78e-08 | 3.15e-08 |
| h0_censored_count | 5.000 | 2.000 | 0.245 | 8.46e-08 | 1.24e-05 |
| vertex_count | 8.000 | 7.000 | 0.166 | 9.64e-06 | 1.83e-05 |
| edge_density | 0.577 | 0.493 | 0.075 | 0.00267 | 1.84e-05 |
| reciprocity | 0.809 | 0.769 | 0.039 | 0.0248 | 0.00639 |
| h1_observed_persistence | 0.000 | 0.000 | 0.000 | 0.423 | 1 |
| h1_betti_auc | 0.000 | 0.000 | 0.000 | 1 | 0.334 |
| h1_betti_max | 0.000 | 0.000 | 0.000 | 1 | 0.334 |
| h1_betti_mean | 0.000 | 0.000 | 0.000 | 1 | 0.334 |
| h1_censored_count | 0.000 | 0.000 | 0.000 | 1 | 0.334 |
| h1_interval_count | 0.000 | 0.000 | 0.000 | 1 | 0.334 |

Open Focus 与 Classical 的独立两两检验如下：

| 指标 | rank-biserial（Open Focus−Classical） | bootstrap 95% CI | BH-FDR $q$ |
|---|---:|---:|---:|
| path_entropy | -0.726 | [-0.846, -0.585] | 1.4e-10 |
| edge_count | -0.694 | [-0.829, -0.545] | 5.58e-10 |
| directed_recurrence | 0.650 | [0.487, 0.788] | 5.56e-09 |
| h0_betti_mean | -0.637 | [-0.781, -0.483] | 8.7e-09 |
| h0_betti_auc | -0.626 | [-0.772, -0.467] | 1.37e-08 |
| h0_observed_persistence | -0.613 | [-0.763, -0.459] | 2.05e-08 |
| transition_entropy | -0.613 | [-0.761, -0.445] | 2.05e-08 |
| h0_betti_max | -0.580 | [-0.728, -0.417] | 5.21e-08 |
| h0_interval_count | -0.580 | [-0.733, -0.417] | 5.21e-08 |
| self_transition_ratio | 0.587 | [0.415, 0.748] | 5.86e-08 |
| h0_censored_count | -0.570 | [-0.725, -0.402] | 8.59e-08 |
| vertex_count | -0.467 | [-0.632, -0.287] | 9.76e-06 |
| edge_density | -0.331 | [-0.523, -0.136] | 0.0027 |
| reciprocity | -0.252 | [-0.451, -0.048] | 0.025 |
| h1_observed_persistence | 0.017 | [-0.000, 0.050] | 0.434 |
| h1_betti_auc | 0.000 | [-0.050, 0.050] | 1 |
| h1_betti_max | -0.000 | [-0.050, 0.050] | 1 |
| h1_betti_mean | 0.000 | [-0.050, 0.050] | 1 |
| h1_censored_count | -0.000 | [-0.050, 0.050] | 1 |
| h1_interval_count | -0.000 | [-0.050, 0.050] | 1 |

### 5.1 解读

1. **跨时长稳定差异。** 只有同时通过 validation/180 s FDR、在 300 s 方向一致并再次显著的指标才视为跨时长稳定；本轮共有 14 项。
2. **状态空间解释。** 状态/边覆盖、路径熵和 $H_0$ 描述两组在同一 10 状态码本中的覆盖与连通过程，不等于音乐质量高低。
3. **$H_1$ 不支持组间结论。** 主阈值非零率分别为 Classical 1.7%、Open Focus 1.7%。扩展阈值下为 12/60 和 11/60；不能将敏感过滤发生率当作主结果。
4. **观察性边界。** 更高自转移率或更低路径熵只描述当前量化空间中的节奏重复性，不证明注意力提升、治疗效果或生成质量。

## 6. 已冻结 holdout 的兼容性核验

- 本次重跑产出的 `rhythm_topology_segments.csv` SHA-256 为 `559dc788d227dd2ed7aea3f0d1dd73279e9782ad27ad182f47f2f52b3eb3edb9`；开盲门控记录为 `559dc788d227dd2ed7aea3f0d1dd73279e9782ad27ad182f47f2f52b3eb3edb9`；核验结果：**一致**。
- 因哈希一致，本次重跑没有改变既有单次 holdout 输入：rhythm/180 s pseudo-$F=5.342$，$p=0.001$，$q=0.00167$；14/14 方向一致，历史 $q\le0.10$ 为 14/14、严格 $q\le0.05$ 为 12/14 复现。未通过严格阈值的是 `edge_density`（$q=0.0802$）与 `reciprocity`（$q=0.0986$），仅保留为提示性 holdout 结果。
- 该 holdout 是回顾性对称重切分下的操作性最终确认；Classical holdout 在旧切分中曾属于 discovery，因此不是 pristine 外部确认集，不能提升为外部独立复制或因果证据。

## 7. 证据层级与局限

- **确认性：** validation/180 s、冻结 10 状态、top-6、主阈值 0.50–0.95、20 指标 omnibus 与独立 pairwise FDR，统一要求 $q\le0.05$。
- **指纹入选：** $q\le0.05$ 是必要条件；还必须满足预定方向、300 s 方向一致性、去冗余和融合增量审计。$0.05<q\le0.10$ 的端点只作提示性结果，不进入核心冻结指纹。
- **敏感性：** validation/300 s；报告跨时长显著性和方向一致性，不以敏感性结果替代主检验。
- **探索/说明性：** discovery 占用率、扩展至 0.05 的阈值、SSM 和有限 $H_1$ 个例。
- **操作性最终确认：** 既有哈希门控 holdout/180 s；本轮只核验输入哈希仍相同，不重新开盲或调参。
- **不支持：** 将两组节奏结构差异解释为稳定或 Focus 特异的 $H_1/H_2$；认知、治疗、生成或因果结论；将已归档三组分析的结论转移到当前两组数据。
- 八维局部节奏向量与 10 状态量化可能压缩长程节拍层级；这属于表示局限，不能通过事后调参追求显著性。

## 8. 复现入口与产物

```powershell
$env:PYTHONPATH = "packages/pathhom_tda/src;src"
python scripts/rerun_rhythm_path_homology.py
python scripts/analyze_rhythm_results.py
python scripts/render_rhythm_path_report.py
```

主要数值文件为 `metadata/rhythm_topology_segments.csv`、`metadata/rhythm_topology_filtration.csv`、`metadata/rhythm_topology_filtration_sensitivity.csv`、`metadata/rhythm_statistical_tests.csv`、`metadata/rhythm_pairwise_tests.csv`、`metadata/rhythm_analysis_summary.json` 和 `metadata/rhythm_topology_summary.json`。
