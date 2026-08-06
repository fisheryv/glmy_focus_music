# Focus Music 纯 Path Homology 拓扑指纹 v2

生成日期：2026-08-03  
指纹 ID：`focus_path_homology_fingerprint_v2`  
冻结 JSON SHA-256：`9bf64f3c1d79c12ec428f1d9f552827d07e9f5c445d9236e7ab676699a62ef1f`  
数据切分：每组 195 discovery / 60 validation / 45 holdout  
主尺度：180 s

## 摘要

本报告根据 2026-08-02 完成的最新 Path Homology 单视角、对称 holdout 与
`L → L+P → L+P+S` 多尺度融合结果，重新提取 Focus Music 的拓扑指纹。新版指纹
完全由 Path Homology 产生，不再使用历史方案中的两个 Vietoris–Rips TDA 端点。

新版主指纹为 `L+P`：

- `L`：音高、节奏、调制三个局部状态转移 Path Homology 块等权融合；
- `P`：Acoustic phase 与 Chroma phase 两个相位提升 Path Homology
  `loop_score` 等权融合；
- `S`：宏观结构 Path Homology 保留为独立辅助层，不进入主指纹。

每个局部视角使用预设的 20 个图与 Path Homology 描述子，经过仅用
discovery/180 s 拟合的中位数填补、协方差白化和有效秩归一化。`L` 为 49 维，
`P` 为 2 维，最终 `L+P` 为 51 维。随后仅在 discovery 上拟合 L2 正则的
Focus-vs-Classical 逻辑判别器。

validation/180 s 上，新指纹 balanced accuracy 为 0.933，ROC-AUC 为 0.982。
已开启 holdout 的描述性结果为 0.911 与 0.982。相位加入局部块后带来正的距离
几何增量：Δpseudo-F=+6.910，单侧 p=0.001，FDR=0.002；结构加入 `L+P` 后
Δpseudo-F=-4.477，p=1.000。因此结构不能并入主指纹。

本方案仍属于探索性验证指纹：主相位块的 Acoustic/Chroma 选择参考过既有
validation 结果，holdout 也不是 pristine 外部样本。它可以用于 exact scoring、
shadow mode 和实验性 reranking；在没有生成实验和代理模型资格检验前，不能声称
采样内引导有效。

![指纹组成](../runs/focus_path_homology_fingerprint_v2/figures/fingerprint_composition.png)

[指纹组成 SVG](../runs/focus_path_homology_fingerprint_v2/figures/fingerprint_composition.svg)

## 1. 为什么必须替换历史指纹

历史 `focus_topology_fingerprint_open_v1` 的核心为：

1. Acoustic novelty delay 的 Vietoris–Rips H0；
2. Rhythm trajectory 的 Vietoris–Rips H0；
3. Acoustic phase Path Homology；
4. Rhythm phase Path Homology。

该版本来自项目早期 TDA 研究，并且其单类 Mahalanobis 中心距离没有通过
Open Focus/Classical 资格检验。最新实验重新完成了音高、节奏、调制、结构和相位
提升的 Path Homology 验证，不再需要用 TDA 端点填补局部尺度。因此旧指纹只应保留
为历史审计产物，不能继续代表当前研究结论。

新版与旧版的关系是：

| 项目 | 历史 open_v1 | 新版 Path Homology v2 |
|---|---|---|
| 局部核心 | 2 个 TDA H0 | Pitch/Rhythm/Modulation PH 全块 |
| 中尺度 | Acoustic/Rhythm phase | Acoustic/Chroma phase PH |
| 宏观结构 | 未进入 | 独立辅助层 S |
| 主评分 | Focus 单类中心距离 | discovery 训练的对比判别分数 |
| validation | 核心距离资格失败 | BA 0.933，AUC 0.982 |
| 用途 | 历史审计 | 当前 exact scoring 候选 |

## 2. 最新证据层级

### 2.1 数据

| split | 每组数量 | 角色 |
|---|---:|---|
| discovery | 195 | 拟合状态模型、块变换和判别器 |
| validation | 60 | 180 s 主检验；300 s 时长敏感性 |
| holdout | 45 | 哈希门控后的单次操作性确认 |

状态模型只用 discovery/180 s 拟合。validation 与 holdout 不参与码本、三分位、
白化矩阵或分类器拟合。

Classical holdout 在旧切分中曾属于 discovery，因此不是 pristine 外部确认集。
多尺度 `L+P` 方案也在查看既往单视角 validation 后形成，所以本指纹不能被包装成
全新的确认性发现。

### 2.2 单视角结果

validation/180 s 的预设 20 指标中：

| 视角 | FDR 发现 | 跨 180/300 s 稳定 | holdout 冻结方向复现 |
|---|---:|---:|---:|
| Pitch | 14 | 13 | 14/14 |
| Rhythm | 14 | 14 | 14/14 |
| Modulation | 12 | 10 个非零中位方向 | 10/10 |
| Structure | 6 | 5 | 4/6 联合 FDR，5/6 同方向 |

最新结论不支持稳定、普遍或 Focus 特异的普通状态图 H1/H2。调制 H1 恒为零，
音高与节奏 H1 高度零膨胀；因此新版不把 H1/H2 指标单独设为“越高越好”的控制
目标。

## 3. Path Homology 原理

### 3.1 有向状态图

对状态序列 \(s_1,\ldots,s_T\)，构造条件转移权重图
\(G=(V,E,w)\)。主过滤保留权重不小于阈值 \(\tau\) 的有向边：

\[
G_\tau=(V,\{(i,j):w_{ij}\ge\tau\}).
\]

允许的 \(p\)-路径必须沿有向边前进，其边界为

\[
\partial e_{i_0\ldots i_p}
=\sum_{q=0}^{p}(-1)^q
e_{i_0\ldots\widehat{i_q}\ldots i_p}.
\]

只保留边界仍由允许路径组成的链空间：

\[
\Omega_p=\{v\in A_p:\partial v\in A_{p-1}\},
\qquad
H_p=\ker\partial_p/\operatorname{im}\partial_{p+1}.
\]

各局部块在固定阈值 0.50–0.95 上汇总 20 个描述子：状态与边数量、密度、互惠、
自转移、转移/路径熵、定向复现，以及 H0/H1 Betti 与 persistence 汇总。

### 3.2 三个局部视角

- Pitch：beat-synchronous chroma → Tonnetz → discovery 拟合的 16 状态码本；
- Rhythm：8 维节奏窗口 → discovery 标准化与 10 状态聚类；
- Modulation：谱调制能量 → discovery 平衡三分位 Low/Medium/High。

三者共享同一 Path Homology 数学框架，但状态语义不同。

### 3.3 相位提升 Path Homology

相位块先从块级距离矩阵选择主周期：

\[
P^*=\arg\min_{P\in\mathcal P}\operatorname{median}_iD_{i,i+P}.
\]

周期位置映射到 6 个相位节点，并计算相邻相位边：

\[
q_i=\left\lfloor\frac{(i\bmod P^*)6}{P^*}\right\rfloor,
\qquad
r_i=\exp(-D_{i,i+P^*}/s),
\]

\[
c_k=\operatorname{mean}\{r_i:q_i=k\},
\qquad
w_k=\min(c_k,c_{k+1}).
\]

预定义六相位有向环完整出现的临界值为

\[
\lambda=\min_kw_k,
\]

即 `loop_score`。它属于相位提升 Path Homology，不是 Vietoris–Rips TDA；也不能
被解释为普通状态图自然发现了普遍 H1。

## 4. 指纹构造

### 4.1 discovery 拟合的块坐标

对原始块 \(X_b\)，仅用 discovery/180 s 拟合中位数填补、均值与协方差。删除
常数列后，以协方差伪逆的特征分解构造白化矩阵 \(W_b\)：

\[
Z_b=\frac{(X_b-\mu_b)W_b}{\sqrt{r_b}},
\qquad W_b^\top\Sigma_bW_b\approx I.
\]

\(r_b\) 是有效秩。主尺度有效秩为：

| 块 | 有效秩 |
|---|---:|
| Pitch | 13 |
| Rhythm | 13 |
| Modulation | 14 |
| Acoustic phase | 1 |
| Chroma phase | 1 |
| Structure | 13 |

除以 \(\sqrt{r_b}\) 后，各块的期望平方距离不会仅因维数更高而变大。

### 4.2 固定等权融合

定义

\[
\operatorname{Eq}(B_1,\ldots,B_k)
=\frac{1}{\sqrt{k}}[B_1|\cdots|B_k].
\]

则

\[
L=\operatorname{Eq}(Z_{pitch},Z_{rhythm},Z_{modulation}),
\]

\[
P=\operatorname{Eq}(Z_{acoustic\ phase},Z_{chroma\ phase}),
\]

\[
LP=\operatorname{Eq}(L,P).
\]

因此局部三块各占 `L` 距离贡献的 1/3，`L` 与 `P` 各占主指纹距离贡献的 1/2。
所有权重固定，不使用 validation 或 holdout 调权。

### 4.3 对比判别分数

在 390 首 discovery/180 s 上拟合 L2 逻辑回归：

\[
S_F(x)=w^\top LP(x)+b,
\qquad
p_F(x)=\sigma(S_F(x)).
\]

正类为 Open Focus，负类为 Classical，`C=10` 来自 discovery 内五折 CV 的固定
网格。分类阈值为 \(p_F=0.5\)。完整的 51 维系数、截距、各块中位数、保留列、
白化矩阵和有效秩已写入 JSON。

为了未来生成控制不把分数无限推高，定义 discovery Focus logit 的第 10 百分位
\(\tau_F\)，并使用分布带损失：

\[
L_{focus}(x)=\left[\max(0,\tau_F-S_F(x))\right]^2.
\]

进入典型 Focus 判别带后损失归零，避免继续强化单个图指标。

## 5. 指纹组成的实证选择

### 5.1 为什么选择 L+P

validation/180 s：

| 表示 | pseudo-F | p | Balanced accuracy | AUROC |
|---|---:|---:|---:|---:|
| L | 6.696 | 0.001 | 0.933 | 0.989 |
| P | 20.580 | 0.001 | 0.683 | 0.733 |
| L+P | 13.606 | 0.001 | 0.933 | 0.982 |
| S | 2.503 | 0.008 | 0.633 | 0.656 |
| L+P+S | 9.129 | 0.001 | 0.908 | 0.972 |

相位加入局部块后的配对增量为：

\[
\Delta_P=F_{L+P}-F_L=+6.910,
\quad p=0.001,
\quad q=0.002.
\]

而相位在回归掉 `L` 后仍有残差分离：pseudo-F=4.712，p=0.020，FDR=0.040。
这支持 `P` 提供非冗余中尺度几何信息。

相位没有提高 balanced accuracy，且 AUROC 比 `L` 低 0.007。因此 `L+P` 的选择
基于“多尺度拓扑指纹”的解释目标，不应写成预测性能优于 `L`。

### 5.2 为什么结构只作辅助层

\[
\Delta_S=F_{L+P+S}-F_{L+P}=-4.477,
\quad p=1.000.
\]

主要 180 s 条件残差 `S | L+P` 也不显著（p=0.107）。结构单块虽然可分，但加入
主空间增加了更多组内方差，降低 pseudo-F 和分类表现。因此：

- `S` 不替换 `P`；
- `S` 不进入主判别分数；
- `S` 可以独立报告、监测宏观退化，并作为未来新数据验证对象。

![验证表现](../runs/focus_path_homology_fingerprint_v2/figures/fingerprint_validation.png)

[验证表现 SVG](../runs/focus_path_homology_fingerprint_v2/figures/fingerprint_validation.svg)

## 6. validation 与 holdout 表现

### 6.1 当前主指纹

重新使用冻结的 discovery 拟合步骤计算得到：

| 数据层 | n | Balanced accuracy | ROC-AUC |
|---|---:|---:|---:|
| validation/180 s | 120 | 0.933 | 0.982 |
| opened holdout/180 s | 90 | 0.911 | 0.982 |

这些数值与多尺度融合报告一致。holdout 只作描述性兼容核对，不能用来重新选择
相位视角、C、阈值或权重。

![分数分布](../runs/focus_path_homology_fingerprint_v2/figures/focus_score_distribution.png)

[分数分布 SVG](../runs/focus_path_homology_fingerprint_v2/figures/focus_score_distribution.svg)

分数分布说明 `L+P` 的对比边界比旧单类中心距离更符合当前 Focus/Classical
验证结果，但仍有少数重叠样本。它描述数据集中的拓扑差异，不是“专注效果”的
概率。

## 7. 可解释的方向性签名

主指纹的 51 维判别器使用完整预设块；为了让其具有可审计解释，另冻结一组方向
签名：Pitch 14 项、Rhythm 14 项、Modulation 10 项，加上两个 phase
`loop_score`，共 40 项。

总体方向是：

- Focus 使用更少的局部状态与边；
- Focus 的路径熵、转移熵及多项 H0 汇总量更低；
- Focus 的自转移率与 directed recurrence 更高；
- Pitch reciprocity 更高，但 Rhythm reciprocity 更低；
- Pitch edge density 更高，但 Rhythm edge density 更低；
- Acoustic/Chroma phase `loop_score` 更高。

因此不能把“密度越高”“互惠越高”写成跨视角通用原则。状态表示不同，同名图指标
可能有不同方向。实际评分必须使用冻结视角前缀和完整系数。

![方向签名](../runs/focus_path_homology_fingerprint_v2/figures/directional_signature.png)

[方向签名 SVG](../runs/focus_path_homology_fingerprint_v2/figures/directional_signature.svg)

## 8. 面向 ACE-Step 的正确使用

### 8.1 允许的当前用途

- 对生成音频运行 exact Path Homology 并计算 `focus_logit`、`focus_probability`；
- shadow mode：记录分数，不改变采样；
- 同一 prompt/seed 候选池的实验性 exact reranking；
- 作为 LTSN 的教师标签与最终 exact verifier；
- Structure PH 作为独立宏观质量监测。

### 8.2 尚未被证明的用途

- 不能直接声称该分数提高注意力或功能性；
- 不能声称 L+P 比 L 有更高分类性能；
- 不能声称采样期梯度引导已经有效；
- 不能把 Structure 强行并入主损失；
- 不能把 H1/H2 单独最大化；
- 不能根据已开启 holdout 再调 C、权重、相位组成或目标分位数。

采样期引导仍需完成：exact reranking 可辨识性、LTSN 未见轨迹资格、配对生成
实验、音质非劣和 exact 复核。

## 9. 证据地位

### 支持

1. Open Focus 与 Classical 在局部 Pitch/Rhythm/Modulation Path Homology 上存在
   稳定组间差异。
2. 相位提升 Path Homology 为 `L` 增加了非冗余的中尺度距离几何。
3. `L+P` discovery 判别器在 validation/180 s 有较强区分性能。
4. 结构可独立描述宏观状态转移，但当前没有正融合增量。

### 不支持

1. TDA H0 继续作为当前拓扑指纹核心。
2. `L+P` 在分类上优于 `L`。
3. `L+P+S` 优于 `L+P`。
4. 稳定或普遍的普通状态图 H1/H2。
5. 拓扑分数与注意力、疗效、生产率或生成质量之间的因果关系。

## 10. 版本与迁移决定

| 版本 | 状态 | 决定 |
|---|---|---|
| `focus_topology_fingerprint_open_v1` | 历史、含 TDA、核心资格失败 | 保留审计，不再用于当前方案 |
| `focus_path_homology_fingerprint_v2` | 当前纯 PH、探索性验证 | 用于 exact scoring/shadow/reranking |

所有新 ACE-Step 文档和后续 surrogate 数据应记录 v2 JSON 的 SHA-256，不应只写
“topology fingerprint”而不带版本。

## 11. 可复现产物

配置与构建：

- `configs/focus_path_homology_fingerprint_v2.toml`
- `scripts/build_focus_path_homology_fingerprint.py`

机器可读产物：

- `metadata/focus_path_homology_fingerprint_v2.json`
- `metadata/focus_path_homology_fingerprint_v2_scores.csv`
- `metadata/focus_path_homology_fingerprint_v2_directions.csv`
- `metadata/focus_path_homology_fingerprint_v2_summary.json`

图形：

- `runs/focus_path_homology_fingerprint_v2/figures/`

JSON 冻结了所有块变换、51 维分类器系数与截距、目标 logit 分位数、验证指标、
输入 SHA-256、排除项和用途边界。分数表覆盖全部 1,200 个片段。

## 12. 最终结论

根据最新验证，Focus Music 的当前拓扑指纹不再是“2 个 TDA H0 + 相位提升”。
应更新为：

```text
局部 Path Homology L
  = Pitch PH + Rhythm PH + Modulation PH

中尺度相位 Path Homology P
  = Acoustic phase loop_score + Chroma phase loop_score

当前主指纹
  = L + P

宏观 Structure PH
  = 独立辅助层，不进入主分数
```

该 51 维纯 Path Homology 指纹在 validation/180 s 上达到 balanced accuracy
0.933、ROC-AUC 0.982，并避免了旧单类 TDA 核心距离错误偏好 Classical 的问题。
但它仍是探索性验证的声学拓扑指纹，而不是注意力因果指标或已经通过生成实验的
采样控制器。
