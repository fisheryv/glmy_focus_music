# 四组音乐 节奏 Path Homology 完整分析报告

## 1. 研究范围与证据边界

本报告把新建的 `focus_open_music` 作为独立的 **Focus Open** 组，与原 Focus、Pop、Classical
共同分析，不把它当作原 Focus 的替代品。四组共 2,200 个 180/300 秒片段、
1,100 首曲目；本视角共 2,200 个片段结果，零失败。

- 码本/状态模型拟合：仅 Discovery/180s，四组严格等量抽样；
- 主推断：Validation/180s，共 255 个片段；
- 敏感性：Validation/300s；Discovery/180s 仅探索；
- Holdout 未进入四组 omnibus，因为原三组并不都具有同构 holdout；
- FDR 家族：每个分析集统一校正 3 视角 × 20 指标；六个两两对比另成一族；
- 这是观察性声学结构比较，不能推出注意力、认知收益或因果效应。

## 2. 视角思想、原理与公式


节奏视角把每个分析窗表示为多维节奏描述向量 $r_t$（局部 onset 强度、节拍/速度、
IOI 与 tempogram 形态等）。缺失维使用 Discovery 中位数 $m_d$ 填补，再标准化：

$$
\widetilde r_{td}=\frac{r_{td}-\mu_d}{\sigma_d},\qquad
r_{td}^{\mathrm{fill}}=\begin{cases}r_{td},&\text{valid},\\m_d,&\text{missing}.\end{cases}
$$

四组 Discovery/180s 严格等量抽样后拟合冻结 K-means 节奏码本：

$$
z_t=\arg\min_v\|\widetilde r_t-\mu_v^{(r)}\|_2^2.
$$

节奏图直接由相邻冻结状态建立；SSM 不参与主图构造。

给定状态序列 $z_1,\ldots,z_T$，相邻有效状态产生计数

$$
C_{ij}=\sum_{t=1}^{T-1}\mathbf 1[z_t=i,z_{t+1}=j],\qquad
P_{ij}=\frac{C_{ij}}{\sum_k C_{ik}}.
$$

每个源状态保留至多 `top_k=6` 条边，并按下降阈值构造

$$
G_\tau=(V,\{(i,j):P_{ij}\ge\tau\}),\qquad
\tau\in\{0.95,0.90,\ldots,0.05\}.
$$

允许 $p$-路径空间记为 $\Omega_p(G_\tau)$，边界算子为

$$
\partial(v_0\cdots v_p)=\sum_{q=0}^{p}(-1)^q
v_0\cdots\widehat{v_q}\cdots v_p,
$$

Path Homology 与 Betti 数为

$$
H_p^{\mathrm{path}}(G_\tau)=
\frac{\ker(\partial_p|_{\Omega_p})}
{\operatorname{im}(\partial_{p+1}|_{\Omega_{p+1}})},
\qquad \beta_p(\tau)=\dim H_p^{\mathrm{path}}(G_\tau).
$$

持久图使用递增坐标 $a=1-\tau$。主分析仅使用
$\tau\in\{0.50,0.60,0.70,0.80,0.90,0.95\}$；扩展阈值只用于敏感性和示例。


## 3. 可视化的持续同调过程（示例）

示例自动选自 Focus Open validation/180s：`focus_jamendo_1650016__180s`。选择规则优先最大
$H_1$ 观测持久性，其次为 $H_1$ 峰值、边数和稳定 ID；因此它用于解释机制，不作为“典型曲目”证据。

![input](../runs/four_group_path_homology/rhythm_state_sequence.png)

![directed_graph](../runs/four_group_path_homology/rhythm_directed_state_graph.png)

![filtration](../runs/four_group_path_homology/rhythm_filtration_process.png)

![persistence](../runs/four_group_path_homology/rhythm_persistence_diagram.png)

![barcode](../runs/four_group_path_homology/rhythm_barcode.png)

## 4. 四组结果

Validation/180s 中位数：

| 指标 | Classical | Focus | Focus Open | Pop |
|---|---:|---:|---:|---:|
| `vertex_count` | 8.000 | 7.000 | 6.000 | 7.000 |
| `edge_count` | 33.000 | 22.500 | 22.500 | 19.000 |
| `path_entropy` | 1.274 | 1.075 | 0.960 | 0.986 |
| `directed_recurrence` | 0.103 | 0.150 | 0.239 | 0.158 |
| `h0_betti_mean` | 6.500 | 4.833 | 4.667 | 4.500 |
| `h1_betti_max` | 0.000 | 0.000 | 0.000 | 0.000 |

主 omnibus 中，本视角有 **14/20** 个指标通过统一 FDR $q\le0.10$；
$H_1$ 非零片段为 **7/255**。中位数为零时，即使秩检验显著，也应解释为
零膨胀发生率或尾部分布差异，而不是“普遍存在环”。

Validation/300s 敏感性中 14/20 个指标通过独立 FDR；与主分析共同通过的指标为 `directed_recurrence`, `edge_count`, `edge_density`, `h0_betti_auc`, `h0_betti_max`, `h0_betti_mean`, `h0_censored_count`, `h0_interval_count`, `h0_observed_persistence`, `path_entropy`, `reciprocity`, `self_transition_ratio`, `transition_entropy`, `vertex_count`。

按统一 FDR 排序的前 12 个 omnibus 指标：

| 指标 | $\epsilon^2$ | FDR $q$ |
|---|---:|---:|
| `edge_count` | 0.230 | 2.00e-12 |
| `h0_betti_auc` | 0.185 | 4.12e-10 |
| `h0_betti_mean` | 0.181 | 6.69e-10 |
| `h0_observed_persistence` | 0.170 | 2.34e-09 |
| `path_entropy` | 0.168 | 2.79e-09 |
| `h0_censored_count` | 0.162 | 5.51e-09 |
| `h0_betti_max` | 0.127 | 3.51e-07 |
| `h0_interval_count` | 0.127 | 3.51e-07 |
| `vertex_count` | 0.117 | 1.25e-06 |
| `directed_recurrence` | 0.101 | 8.47e-06 |
| `transition_entropy` | 0.076 | 1.49e-04 |
| `edge_density` | 0.060 | 9.18e-04 |

Focus–Focus Open 与 Focus Open–Pop 的关键两两比较（前 16 项；正 rank-biserial 表示前者更高）：

| 对比 | 指标 | rank-biserial | FDR $q$ |
|---|---|---:|---:|
| Focus vs Focus Open | `self_transition_ratio` | -0.348 | 0.013 |
| Focus Open vs Pop | `edge_density` | 0.276 | 0.022 |
| Focus vs Focus Open | `directed_recurrence` | -0.287 | 0.055 |
| Focus vs Focus Open | `transition_entropy` | 0.261 | 0.093 |
| Focus vs Focus Open | `path_entropy` | 0.224 | 0.171 |
| Focus Open vs Pop | `transition_entropy` | -0.179 | 0.206 |
| Focus Open vs Pop | `h0_censored_count` | 0.166 | 0.223 |
| Focus Open vs Pop | `directed_recurrence` | 0.163 | 0.251 |
| Focus Open vs Pop | `reciprocity` | 0.157 | 0.273 |
| Focus vs Focus Open | `h1_betti_auc` | -0.033 | 0.439 |
| Focus vs Focus Open | `h1_betti_max` | -0.033 | 0.439 |
| Focus vs Focus Open | `h1_betti_mean` | -0.033 | 0.439 |
| Focus vs Focus Open | `h1_censored_count` | -0.033 | 0.439 |
| Focus vs Focus Open | `h1_interval_count` | -0.033 | 0.439 |
| Focus Open vs Pop | `self_transition_ratio` | 0.115 | 0.439 |
| Focus Open vs Pop | `h1_observed_persistence` | 0.017 | 0.455 |

![group_summary](../runs/four_group_path_homology/rhythm_group_summary.png)

![betti_curves](../runs/four_group_path_homology/rhythm_betti_curves.png)

## 5. 解读

1. **Focus Open 必须保持独立。** 它的来源、授权条件、筛选机制和原 Focus 不同；拓扑相似只能支持
   “在本表示下接近”，不能证明数据分布等价。
2. **优先看效应量和方向。** Kruskal–Wallis 的 $\epsilon^2$ 回答四组总体可分程度；
   rank-biserial 回答具体两组方向。统一 FDR 后未通过的差异一律标为不支持。
3. **$H_0$ 与图描述量通常比 $H_1$ 更稳定。** 状态数、边数、路径熵、复现率和连通分支变化反映
   状态组织方式；$H_1$ 只在少量曲目出现时应作为稀有结构现象报告。
4. **阈值敏感性不是主结论。** 扩展过滤用于展示类的出生/死亡；主统计仍冻结在 0.50–0.95。
5. **跨视角结论需查阅三份报告。** 结构描述宏观段落，pitch_v2 描述谐波骨架，节奏描述局部
   时间组织；任何单一视角都不能代表完整音乐结构。

## 6. 复现与产物

```powershell
$env:PYTHONPATH = "packages/pyglmy/src;src"
python scripts/run_four_group_path_homology.py --workers 6
python scripts/render_four_group_path_reports.py
```

数值产物：`metadata/four_group_topology_segments.csv`、
`metadata/four_group_topology_filtration.csv`、
`metadata/four_group_topology_filtration_sensitivity.csv`、
`metadata/four_group_statistical_tests.csv`、`metadata/four_group_pairwise_tests.csv`。
模型、特征、图和同调结果均位于带 `four_group` 的隔离命名空间；PNG 与 SVG 同步输出到
`runs/four_group_path_homology/`，未覆盖原三组验证产物。
> **历史结果声明（2026-08-02）：** 本报告使用旧 Brain.fm + Focus Open 四组设计，已被当前三组开放数据集取代。参见 [迁移报告](open-focus-migration-report.md)。
