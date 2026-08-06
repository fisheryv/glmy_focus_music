# 四组音乐 结构 Path Homology 完整分析报告

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


结构视角先对宏观声学块 $x_t$ 构造余弦自相似矩阵

$$
S_{ij}=\frac{\langle x_i,x_j\rangle}
{\|x_i\|_2\|x_j\|_2},
$$

再用沿对角线移动的棋盘核 $K_L$ 计算新颖度

$$
\nu(t)=\sum_{i=-L}^{L}\sum_{j=-L}^{L}K_L(i,j)S_{t+i,t+j}.
$$

新颖度峰值经最短/最长段约束形成边界 $b_0<\cdots<b_M$；段向量为

$$
s_m=\operatorname{pool}\{x_t:b_m\le t<b_{m+1}\}.
$$

Discovery/180s 四组等量抽样后，对标准化、PCA 投影的 $s_m$ 拟合 K-means，得到
冻结结构原型；Validation 和 300s 只做映射，不参与拟合。SSM 在此视角中是边界构造的一部分。

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

示例自动选自 Focus Open validation/180s：`focus_jamendo_1787506__180s`。选择规则优先最大
$H_1$ 观测持久性，其次为 $H_1$ 峰值、边数和稳定 ID；因此它用于解释机制，不作为“典型曲目”证据。

![input](../runs/four_group_path_homology/structure_ssm_boundaries.png)

![directed_graph](../runs/four_group_path_homology/structure_directed_state_graph.png)

![filtration](../runs/four_group_path_homology/structure_filtration_process.png)

![persistence](../runs/four_group_path_homology/structure_persistence_diagram.png)

![barcode](../runs/four_group_path_homology/structure_barcode.png)

## 4. 四组结果

Validation/180s 中位数：

| 指标 | Classical | Focus | Focus Open | Pop |
|---|---:|---:|---:|---:|
| `vertex_count` | 3.000 | 2.000 | 3.000 | 3.000 |
| `edge_count` | 3.000 | 2.000 | 2.000 | 3.000 |
| `path_entropy` | 0.519 | 0.391 | 0.417 | 0.462 |
| `directed_recurrence` | 0.281 | 0.439 | 0.354 | 0.273 |
| `h0_betti_mean` | 1.000 | 1.000 | 1.000 | 1.000 |
| `h1_betti_max` | 0.000 | 0.000 | 0.000 | 0.000 |

主 omnibus 中，本视角有 **7/20** 个指标通过统一 FDR $q\le0.10$；
$H_1$ 非零片段为 **33/255**。中位数为零时，即使秩检验显著，也应解释为
零膨胀发生率或尾部分布差异，而不是“普遍存在环”。

Validation/300s 敏感性中 13/20 个指标通过独立 FDR；与主分析共同通过的指标为 `directed_recurrence`, `edge_count`, `edge_density`, `reciprocity`, `self_transition_ratio`, `transition_entropy`, `vertex_count`。

按统一 FDR 排序的前 12 个 omnibus 指标：

| 指标 | $\epsilon^2$ | FDR $q$ |
|---|---:|---:|
| `self_transition_ratio` | 0.082 | 7.32e-05 |
| `reciprocity` | 0.058 | 0.001 |
| `edge_count` | 0.054 | 0.002 |
| `edge_density` | 0.037 | 0.013 |
| `vertex_count` | 0.030 | 0.029 |
| `directed_recurrence` | 0.027 | 0.038 |
| `transition_entropy` | 0.022 | 0.064 |
| `path_entropy` | 0.016 | 0.125 |
| `h0_censored_count` | 0.004 | 0.368 |
| `h0_betti_max` | 0.003 | 0.397 |
| `h0_interval_count` | 0.003 | 0.397 |
| `h0_observed_persistence` | 0.003 | 0.401 |

Focus–Focus Open 与 Focus Open–Pop 的关键两两比较（前 16 项；正 rank-biserial 表示前者更高）：

| 对比 | 指标 | rank-biserial | FDR $q$ |
|---|---|---:|---:|
| Focus Open vs Pop | `self_transition_ratio` | 0.301 | 0.010 |
| Focus Open vs Pop | `edge_count` | -0.228 | 0.071 |
| Focus Open vs Pop | `transition_entropy` | -0.166 | 0.234 |
| Focus vs Focus Open | `vertex_count` | -0.189 | 0.234 |
| Focus Open vs Pop | `vertex_count` | -0.159 | 0.234 |
| Focus Open vs Pop | `directed_recurrence` | 0.161 | 0.255 |
| Focus vs Focus Open | `self_transition_ratio` | 0.189 | 0.262 |
| Focus Open vs Pop | `h0_censored_count` | 0.057 | 0.311 |
| Focus vs Focus Open | `h0_betti_auc` | -0.105 | 0.356 |
| Focus vs Focus Open | `h0_betti_mean` | -0.105 | 0.356 |
| Focus vs Focus Open | `h0_betti_max` | -0.102 | 0.374 |
| Focus vs Focus Open | `h0_interval_count` | -0.102 | 0.374 |
| Focus vs Focus Open | `h0_observed_persistence` | -0.102 | 0.374 |
| Focus vs Focus Open | `transition_entropy` | -0.142 | 0.426 |
| Focus vs Focus Open | `h0_censored_count` | -0.058 | 0.429 |
| Focus vs Focus Open | `directed_recurrence` | 0.121 | 0.518 |

![group_summary](../runs/four_group_path_homology/structure_group_summary.png)

![betti_curves](../runs/four_group_path_homology/structure_betti_curves.png)

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
$env:PYTHONPATH = "packages/pathhom_tda/src;src"
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
