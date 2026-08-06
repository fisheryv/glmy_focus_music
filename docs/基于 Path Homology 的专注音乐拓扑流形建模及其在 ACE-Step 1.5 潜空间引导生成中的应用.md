# 基于 Path Homology 的专注音乐拓扑流形建模及其在 ACE-Step 1.5 潜空间引导生成中的应用

# Topological Manifold Modeling of Focus Music via Path Homology and Its Application to Latent-Space Guided Generation in ACE-Step 1.5

作者：[待填写]
学校：[待填写]
指导教师：[待填写]
日期：[待填写]

------

## 摘要

功能性专注音乐被广泛用于学习、写作和工作等持续注意场景，但“专注音乐”区别于普通音乐的结构性特征仍缺少可计算、可解释、可用于生成控制的数学表达。本文提出一种基于 Path Homology 的专注音乐拓扑流形建模方法，并将其用于 ACE-Step 1.5 音乐生成模型的潜空间引导。研究以已获授权的 Brain.fm Focus 音乐作为功能性专注音乐参考集，以已获授权的 Spotify Pop 音乐作为普通流行音乐对照集。Brain.fm 官方科学页面说明其音乐通过 placebo controls、EEG 和 fMRI 等方式进行测试，并强调其音乐设计目标包括减少注意力抓取元素，使音乐更适合作为背景支持。

本文不从神经声学机制或频谱调制角度建模专注音乐，而将音乐视为一个随时间演化的有向动态系统。具体而言，本文首先从音频片段中提取短时声学与时间结构特征，将连续音乐序列离散化为状态序列；随后由状态转移构建加权有向图；最后使用 path homology 与 persistent path homology 描述音乐状态演化中的有向连通性、循环性、稳定性、复现性和复杂度。Path homology 最初由 Grigor’yan、Lin、Muranov 和 Yau 等人在 path complex 与 digraph 框架下提出，其核心动机之一是为有向图建立同调理论。 Persistent path homology 进一步将该思想扩展到多尺度有向网络分析，并能够编码 directed network 的非对称结构。

在生成阶段，本文采用 ACE-Step 1.5 作为骨干音乐生成模型。ACE-Step 1.5 是一个开源音乐生成 foundation model，其论文报告该模型使用连续音频潜空间和 Diffusion Transformer 进行高效音乐生成，并支持本地运行和快速推理。 本文不微调 ACE-Step 主模型，而是在推理阶段引入 topology-guided latent intervention，包括 best-of-N 重排序、latent steering、Langevin-style latent refinement 和 denoising-time topology guidance。理论上，本文不要求证明 ACE-Step 潜空间与音乐物理空间之间存在全局双利普希茨映射；相反，本文提出更弱且可验证的理论框架：局部稳定性、拓扑代理评分器一致性和潜空间局部可控性。Persistent path diagram 的稳定性研究为这种局部稳定性论证提供了数学依据。

本文的主要贡献包括：第一，提出专注音乐的有向拓扑流形假设；第二，将 path homology 引入音乐时间结构建模；第三，构建 Brain.fm Focus 与 Spotify Pop 的授权双域对照实验框架；第四，提出基于拓扑评分的 ACE-Step 潜空间推理阶段引导方法；第五，建立无需全局双利普希茨证明的局部可控性理论框架。本文最终目标不是复制 Brain.fm 音乐，而是在版权合规与模型安全边界内，生成拓扑结构更接近功能性专注音乐参考流形的原创音乐。

**关键词：** 专注音乐；Path Homology；Persistent Path Homology；拓扑数据分析；有向图；音乐生成；ACE-Step 1.5；潜空间引导；Langevin dynamics

------

## Abstract

Focus music is widely used in learning, writing, and work scenarios that require sustained attention, yet its structural distinction from ordinary music remains difficult to formalize in a computable and interpretable way. This paper proposes a path-homology-based framework for modeling the topological manifold of focus music and applies it to latent-space guided generation in ACE-Step 1.5. The study uses licensed Brain.fm Focus music as a functional focus-music reference set and licensed Spotify Pop music as a general pop-music control set. Brain.fm’s official science page states that its music is tested with placebo controls, EEG, and fMRI, and that its design process aims to reduce attention-grabbing elements so that the music can remain comfortably in the background.

Rather than modeling focus music through a neuroacoustic or spectral-mechanism perspective, this paper treats music as a directed dynamical system evolving over time. Each audio segment is converted into a short-time sequence of acoustic and temporal-structure states. These states are then used to construct a weighted directed transition graph, from which path homology and persistent path homology are computed. Path homology was introduced in the framework of path complexes and digraphs, providing a homological theory for directed graphs. Persistent path homology extends this idea to multi-resolution directed networks and captures asymmetric structures that ordinary persistent homology does not naturally represent.

For generation, this paper adopts ACE-Step 1.5 as the backbone music generation model. ACE-Step 1.5 is an open-source music foundation model that uses a latent audio representation and a Diffusion Transformer for efficient music generation. Instead of fine-tuning the main generative model, this paper introduces topology-guided latent intervention during inference, including best-of-N reranking, latent steering, Langevin-style latent refinement, and denoising-time topology guidance. Theoretically, the paper does not require a global bi-Lipschitz correspondence between the ACE-Step latent space and the physical audio space. Instead, it relies on weaker and empirically testable conditions: local stability, surrogate-score consistency, and local latent controllability. Stability results for persistent path diagrams provide mathematical support for this framework.

The main contributions are fivefold: proposing a directed-topological-manifold hypothesis for focus music; applying path homology to temporal music structure; constructing a licensed dual-domain experimental framework using Brain.fm Focus and Spotify Pop; designing topology-guided latent-space generation for ACE-Step 1.5; and formulating a local controllability framework that avoids unrealistic global bi-Lipschitz assumptions.

**Keywords:** focus music; path homology; persistent path homology; topological data analysis; directed graph; music generation; ACE-Step 1.5; latent guidance; Langevin dynamics

------

# 1. 引言

## 1.1 研究背景

在学习、写作、编程和深度工作场景中，背景音乐常被用于支持持续注意。然而，普通流行音乐通常包含人声、歌词、强旋律钩子、明显段落变化、情绪高潮和节奏断裂等容易吸引注意的元素。对于专注任务而言，理想背景音乐并不一定是审美刺激最大化的音乐，而可能是结构稳定、分心度较低、时间推进可预测、重复但不过度单调的音乐。

Brain.fm 是当前较有代表性的功能性音乐平台之一。其官方科学页面强调其音乐经过 placebo control、EEG 和 fMRI 等方法测试，并指出其设计过程会削弱或移除注意力抓取元素，使音乐更适合作为背景支持。 因此，Brain.fm Focus 音乐适合作为“功能性专注音乐参考集”，但本文不将其定义为医学意义上的金标准，也不直接声称其对 ADHD 或其他注意障碍具有临床疗效。

本文关注的问题是：功能性专注音乐是否具有可由数学结构描述的时间组织方式？如果存在这种结构，能否将其用于生成模型的潜空间控制，使模型生成更接近专注音乐结构的原创音乐？

## 1.2 核心思想

本文将音乐看作一个随时间演化的动态系统。音乐的每一小段短时音频可以被表示为一个状态，整首音乐则是状态序列：

$$
X=(x_1,x_2,\ldots,x_T),\qquad x_t\in\mathbb R^d.
$$

如果音乐状态从 $x_t$ 演化到 $x_{t+1}$，则可以构建一个有向转移关系。将连续状态离散化后，音乐片段可以表示为加权有向图：

$$
G_X=(V,E,W).
$$

其中 $V$ 是音乐状态集合，$E$ 是状态转移边，$W$ 是边权矩阵。普通音频特征通常只能描述某一时刻或某一局部窗口的声学性质，而有向图可以表达音乐状态如何随时间推进。Path homology 则进一步提供一种分析该有向图拓扑结构的数学工具。

本文的核心假设是：功能性专注音乐相较普通流行音乐，可能具有更稳定、更低分叉、更高复现率和更可预测的有向状态转移结构。这种结构可以通过 path homology、persistent path homology、path entropy、directed recurrence 和有向循环持久性等指标被量化。

## 1.3 研究贡献

本文的贡献如下。

第一，提出专注音乐的有向拓扑流形假设。本文不把 focus music 仅视为音乐风格或情绪标签，而将其建模为有向状态转移系统，并认为其功能性可能与该系统的拓扑结构有关。

第二，将 path homology 引入音乐时间结构分析。Path homology 是针对有向图的同调理论，适合保留音乐状态演化中的时间方向性；相比普通无向 persistent homology，其更适合描述音乐转移路径的非对称结构。

第三，构建 Brain.fm Focus 与 Spotify Pop 的授权双域对照框架。本文将 Brain.fm Focus 作为功能性专注音乐参考集，将 Spotify Pop 作为普通流行音乐对照集，并设计 matched control 子集，以减少人声、响度、速度和配器等浅层混淆因素。

第四，提出 topology-guided latent intervention。本文基于 ACE-Step 1.5 的连续潜空间，在推理阶段引入拓扑评分器，而不是直接微调主生成模型。ACE-Step 1.5 的论文报告其具备高效本地生成能力和连续音频潜空间结构，使其适合作为潜空间引导实验平台。

第五，提出不依赖全局双利普希茨映射的理论框架。对于生成式音乐模型，潜空间到音乐拓扑结构的全局双利普希茨对应过强且不现实。本文以局部稳定性、代理一致性和局部可控性作为理论支撑，并通过实验验证这些性质。

------

# 2. 相关工作

## 2.1 功能性专注音乐

功能性音乐的核心目标不是前景审美，而是支持特定认知、情绪或行为状态。Brain.fm 官方页面将其音乐描述为经过科学测试的功能性音乐，并强调其音乐设计目标包括降低分心元素和支持背景使用。 本文以 Brain.fm Focus 音乐作为专注音乐参考集，但将研究问题限定在“拓扑结构相似性”和“生成控制”范围内，不作医学疗效判断。

## 2.2 音乐时间结构建模

传统音乐信息检索通常使用 tempo、loudness、MFCC、chroma、spectral centroid、spectral flux、onset density、self-similarity matrix 等特征。这些特征可以表达音色、节奏、和声与局部变化，但它们通常难以直接刻画音乐状态转移的方向性。音乐具有不可逆的时间结构：从 A 段到 B 段与从 B 段到 A 段在感知上并不等价。因此，本文将音乐表示为有向状态转移图，而非无向点云。

## 2.3 Path Homology 与有向图拓扑

Path homology 由 Grigor’yan、Lin、Muranov 和 Yau 等人在 path complex 与 digraph 框架下提出，目标之一是为 directed graphs 建立同调理论。 Persistent path homology 由 Chowdhury 和 Mémoli 扩展到多尺度有向网络分析；该工作指出，普通 persistent homology 对 directed/asymmetric data 的表达能力受限，而 persistent path homology 可以编码有向网络的非对称结构。 后续关于 persistent path diagram stability 的研究进一步说明，对于 weighted digraphs 或 edge-weighted path complexes，persistent path diagram 在适当距离下具有稳定性。

这些理论为本文提供了两个基础：第一，音乐状态转移图应当保留方向性；第二，若图权重变化较小，其 persistent path diagram 不应发生任意剧烈变化。这使得 path homology 可以成为音乐拓扑流形分析和潜空间引导的数学基础。

## 2.4 ACE-Step 1.5 与潜空间音乐生成

ACE-Step 1.5 是一个开源音乐生成 foundation model。其论文报告该模型具备高效推理能力，可在消费级硬件上运行，并支持文本到音乐、编辑、个性化等任务。 对本文而言，ACE-Step 1.5 的关键价值在于它提供了可操作的连续潜空间，使得研究者可以在不重训主模型的情况下进行推理阶段引导。

## 2.5 Diffusion Guidance 与 Langevin Dynamics

Score-based generative modeling 将生成过程看作由噪声分布反向演化到数据分布的过程，并使用 score field 指导采样。 Classifier guidance 则说明，可以利用外部分类器梯度影响 diffusion 采样轨迹，从而改变生成结果的条件属性。 本文借鉴这一思想，但外部引导信号不是类别分类器，而是拓扑代理评分器。该评分器学习生成音乐与专注音乐拓扑流形之间的关系，并在潜空间中提供可微引导方向。

------

# 3. 问题定义与符号

## 3.1 空间定义

设音乐物理空间为：$\mathcal X$，其中每个元素 $x\in\mathcal X$ 是一段音频。

设 ACE-Step 1.5 的潜空间为：$\mathcal Z$，其中每个元素 $z\in\mathcal Z$ 表示一个连续 latent 序列。

设拓扑描述子空间为：$\mathcal T$，其中每个元素 $\tau(x)\in\mathcal T$ 是由 path homology 与图统计构成的拓扑特征向量。

ACE-Step 生成器记为：$G_\theta:\mathcal Z\rightarrow\mathcal X.$

从音频到拓扑描述子的映射记为：$\Phi:\mathcal X\rightarrow\mathcal T$

潜空间到拓扑空间的复合映射为：$\Psi(z)=\Phi(G_\theta(z))$

## 3.2 数据集定义

设Focus Music 数据集为：

$$
\mathcal D_F={x_i^F}_{i=1}^{N_F}.
$$

设 Pop Music 对照数据集为：

$$
\mathcal D_P={x_i^P}_{i=1}^{N_P}.
$$

设 Classical Music 控制数据集为：

$$
\mathcal D_C={x_i^C}_{i=1}^{N_C}.
$$

其中 $\mathcal D_C$ 从 Spotify Pop 中选取，在 tempo、loudness、duration、energy、instrumentalness 或本地提取的等价指标上尽可能匹配 Focus Music。

Focus Music 拓扑流形定义为：

$$
\mathcal M_F={\tau(x):x\in\mathcal D_F}.
$$

Pop Music 拓扑流形定义为：

$$
\mathcal M_P={\tau(x):x\in\mathcal D_P}.
$$

------

# 4. 数据集构建与合规方案

## 4.1 数据来源

本文使用三组数据。

第一组为 Focus Music。该组作为功能性专注音乐参考集。Brain.fm 官方页面说明其音乐经过心理学注意任务、placebo control、EEG 和 fMRI 等方式测试。 本文将该组称为“功能性专注音乐参考集”，而非临床金标准。

第二组为 Pop Muisc。该组作为普通流行音乐对照集。Spotify 官方开发者政策对 Spotify Platform 与 Spotify Content 的使用设有严格限制，其中包括不得使用 Spotify Content 训练机器学习或 AI 模型。 因此，本文默认不使用 Spotify 音频训练生成模型；若用户持有额外书面授权，应在论文附录中说明授权范围。若授权不覆盖训练或生成式模型优化，则 Spotify 仅用于分析、对照和评价。

第三组为 Classical Music。该组用于排除“Focus Music 与 Pop Music 的差异只是人声、响度、速度或配器差异”这一混淆解释。

## 4.2 预处理流程

所有音频进行统一预处理：

$$
x \rightarrow \tilde{x}.
$$

预处理包括：

$$
\tilde{x}
=
\mathrm{Normalize}
(
\mathrm{Resample}
(
\mathrm{TrimSilence}(x)
)
).
$$

具体步骤为：

1. 统一采样率为 22.05kHz 或 44.1kHz；
2. 统一声道为 mono，必要时保留 stereo 作为补充实验；
3. 使用 loudness normalization 消除整体音量差异；
4. 切分为 30s、60s、120s 三种窗口；
5. 以曲目级别划分 train / validation / test，避免同一曲目片段泄漏。

推荐划分为：

$$
60\% : 20\% : 20\%.
$$

## 4.3 版权与发表策略

本文不公开 Brain.fm 或 Spotify 原始音频，不公开可逆重建的音频 latent，不公开可恢复原始音频的中间缓存。公开内容仅包括：

1. 代码；
2. 匿名化聚合统计；
3. path homology 指标；
4. 拓扑特征分布；
5. 模型结构；
6. 由 ACE-Step 生成的原创音频样本；
7. 不含版权音频的实验日志。

------

# 5. 音乐状态序列建模

## 5.1 短时声学特征

将音频片段 $x$ 按固定帧长 $\Delta t$ 切分：

$$
x\rightarrow (x_1,x_2,\ldots,x_T).
$$

对每一帧提取声学特征：

$$
a_t=
[
\mathrm{loudness},
\mathrm{MFCC},
\mathrm{chroma},
\mathrm{spectral\ centroid},
\mathrm{spectral\ flux},
\mathrm{tonnetz},
\mathrm{ZCR},
\mathrm{onset\ density}
].
$$

## 5.2 时间结构特征

为刻画音乐的局部时序结构，定义：

$$
r_t=
[
\mathrm{local\ recurrence},
\mathrm{self\ similarity},
\mathrm{transition\ novelty},
\mathrm{tempo\ normalized\ position},
\mathrm{onset\ interval\ statistics}
].
$$

其中 local recurrence 衡量当前状态是否与过去状态相似，transition novelty 衡量当前状态变化是否突兀。

最终每个时间帧表示为：

$$
u_t=[a_t,r_t]\in\mathbb R^d.
$$

整段音乐表示为：

$$
U=(u_1,u_2,\ldots,u_T).
$$

## 5.3 状态离散化

使用聚类或向量量化函数：

$$
q:\mathbb R^d\rightarrow V={v_1,\ldots,v_K}.
$$

定义离散状态序列：

$$
s_t=q(u_t).
$$

因此，音乐片段从连续特征序列变为状态路径：

$$
S=(s_1,s_2,\ldots,s_T).
$$

------

# 6. 有向状态转移图构建

## 6.1 有向图定义

对于任意相邻状态 (s_t=i) 和 (s_{t+1}=j)，在图中加入有向边：

$$
i\rightarrow j.
$$

得到有向图：

$$
G_X=(V,E,W).
$$

其中：

$$
E={(i,j):\exists t,\ s_t=i,\ s_{t+1}=j}.
$$

边权定义为经验转移概率：

$$
W_{ij}=
\frac{
\#{t:s_t=i,s_{t+1}=j}
}{
\#{t:s_t=i}+\varepsilon
}.
$$

矩阵形式为：

$$
W=
\begin{bmatrix}
W_{11} & W_{12} & \cdots & W_{1K}\\
W_{21} & W_{22} & \cdots & W_{2K}\\
\vdots & \vdots & \ddots & \vdots\\
W_{K1} & W_{K2} & \cdots & W_{KK}
\end{bmatrix}.
$$

## 6.2 基础图统计

本文定义以下图统计量。

**Path Entropy：**

$$
H_{\mathrm{path}}=-\sum_i \pi_i\sum_j W_{ij}\log(W_{ij}+\varepsilon),
$$

其中 $\pi_i$ 为状态 $i$ 的经验占比。该指标衡量状态转移的不确定性。

**Directed Recurrence：**

$$
R_{\mathrm{dir}}=
\frac{
\#{(t,t'):s_t=s_{t'},s_{t+1}=s_{t'+1}}
}{
(T-1)^2
}.
$$

该指标衡量相同有向转移模式的复现率。

**Transition Sparsity：**

$$
S_{\mathrm{sparse}}=1-\frac{|E|}{K^2}.
$$

该指标衡量转移图的稀疏程度。

**Transition Novelty：**

$$
N_{\mathrm{transition}}=
\frac{1}{T-1}\sum_{t=1}^{T-1}|u_{t+1}-u_t|_2.
$$

该指标衡量音乐状态变化的平均幅度。

------

# 7. Path Homology 拓扑特征

## 7.1 允许路径

给定有向图 $G=(V,E)$，一个允许的 $p$-path 定义为：

$$
e_{i_0i_1\cdots i_p},
$$

其中：

$$
(i_k,i_{k+1})\in E,\qquad k=0,1,\ldots,p-1.
$$

所有允许 $p$-path 张成向量空间：

$$
\mathcal A_p(G).
$$

## 7.2 边界算子

定义边界算子：

$$
\partial_p e_{i_0i_1\cdots i_p}=
\sum_{k=0}^{p}(-1)^k e_{i_0\cdots \widehat{i_k}\cdots i_p}.
$$

其中 $\widehat{i_k}$ 表示删除第 $k$ 个顶点。

由于删除顶点后所得路径不一定仍是允许路径，定义：

$$
\Omega_p(G)=
{u\in\mathcal A_p(G):\partial_p u\in\mathcal A_{p-1}(G)}.
$$

## 7.3 Path Homology 群

Path homology 群定义为：

$$
H_p^{path}(G)=
\ker(\partial_p|*{\Omega_p})
/
\mathrm{im}(\partial*{p+1}|*{\Omega*{p+1}}).
$$

对应 Betti 数为：

$$
\beta_p^{path}(G)=\dim H_p^{path}(G).
$$

本文主要使用：

$$
\beta_0^{path},\qquad \beta_1^{path}.
$$

其中 $\beta_0^{path}$ 描述有向连通结构，$\beta_1^{path}$ 描述有向循环结构。

## 7.4 Persistent Path Homology

由于音乐状态转移图具有边权，因此构造过滤：

$$
G^\epsilon=(V,E^\epsilon),
$$

其中：

$$
E^\epsilon={(i,j):W_{ij}\geq \epsilon}.
$$

随着阈值 $\epsilon$ 改变，得到一族有向图：

$$
G^{\epsilon_1}\subseteq G^{\epsilon_2}\subseteq \cdots \subseteq G^{\epsilon_L}.
$$

对每个 $G^\epsilon$ 计算 path homology，得到 persistent path diagram：

$$
D_p^{path}(G).
$$

Persistent path homology 能够在多个尺度下分析有向图结构，并编码普通 persistent homology 难以表达的非对称信息。 Persistent path diagram 的稳定性结果说明，该表示在加权有向图扰动下具有可控变化。

## 7.5 拓扑描述子

最终定义音乐片段的拓扑描述子：

$$
\tau(x)=
[
\beta_0^{path},
\beta_1^{path},
\mathrm{PersStat}(D_0),
\mathrm{PersStat}(D_1),
H_{\mathrm{path}},
R_{\mathrm{dir}},
S_{\mathrm{sparse}},
C_{\mathrm{cycle}},
N_{\mathrm{transition}}
].
$$

其中：

$$
C_{\mathrm{cycle}}=\mathrm{TotalPersistence}(D_1)
$$

或可定义为 top-k 有向循环寿命之和。

------

# 8. 专注音乐拓扑流形

## 8.1 流形定义

Focus Music 拓扑流形为：

$$
\mathcal M_F=
{\tau(x_i^F):x_i^F\in\mathcal D_F}.
$$

Pop Music 拓扑流形为：

$$
\mathcal M_P=
{\tau(x_i^P):x_i^P\in\mathcal D_P}.
$$

给定任意生成音乐 $y$，其与专注音乐流形的距离定义为：

$$
d_{\mathcal T}(\tau(y),\mathcal M_F)=
\min_{\tau_i\in\mathcal M_F}
d_{\mathcal T}(\tau(y),\tau_i).
$$

距离函数 $d_{\mathcal T}$ 可以采用：

$$
d_{\mathcal T} = d_{\mathrm{Mahalanobis}} + \lambda d_{\mathrm{diagram}}.
$$

其中 $d_{\mathrm{diagram}}$ 可由 bottleneck distance 或 Wasserstein distance 近似。

## 8.2 Exact Topology Score

定义 exact topology score：

$$
S_{\mathrm{exact}}(y)=

 -\alpha d_{\mathcal T}(\tau(y),\mathcal M_F)^2 + \beta d_{\mathcal T}(\tau(y),\mathcal M_P)^2 + \gamma R_{\mathrm{struct}}(y)

 \lambda R_{\mathrm{copy}}(y)

\mu R_{\mathrm{bad}}(y).
$$

其中结构奖励为：

$$
R_{\mathrm{struct}}(y)=
w_1R_{\mathrm{dir}}(y)

w_2H_{\mathrm{path}}(y) + w_3C_{\mathrm{cycle}}(y) + w_4S_{\mathrm{sparse}}(y)

w_5N_{\mathrm{transition}}(y).
$$

各项含义如下：

$$
d_{\mathcal T}(\tau(y),\mathcal M_F)
$$

表示与专注音乐拓扑流形的距离；

$$
d_{\mathcal T}(\tau(y),\mathcal M_P)
$$

表示与普通流行音乐拓扑流形的距离；

$$
R_{\mathrm{copy}}(y)
$$

表示与授权参考音乐过度相似的惩罚；

$$
R_{\mathrm{bad}}(y)
$$

表示音质劣化、爆音、失真、过度循环等问题的惩罚。

------

# 9. ACE-Step 1.5 潜空间引导

## 9.1 潜空间干预入口

ACE-Step 1.5 的论文将其描述为开源音乐 foundation model，支持高效生成和本地运行。 本文使用其潜空间作为干预对象。潜空间干预可发生在以下位置：

1. VAE continuous latent；
2. DiT denoising latent；
3. 中间层 hidden state；
4. 条件嵌入空间；
5. 生成后 latent refinement 阶段。

本文优先选择 continuous latent 和 denoising latent，因为它们是连续变量，适合梯度引导。

## 9.2 拓扑代理评分器

Exact score 依赖图构建与 path homology 计算，通常不可导。因此训练可导代理评分器：

$$
S_\varphi:\mathcal Z\rightarrow \mathbb R.
$$

目标为：

$$
S_\varphi(z)\approx S_{\mathrm{exact}}(G_\theta(z)).
$$

训练损失为：

$$
\mathcal L_{\mathrm{sur}}=

\mathcal L_{\mathrm{reg}}
+
\lambda_r\mathcal L_{\mathrm{rank}}
+
\lambda_s\mathcal L_{\mathrm{smooth}}.
$$

其中：

$$
\mathcal L_{\mathrm{reg}}=

\frac{1}{N}\sum_i \left( S_\varphi(z_i)

S_{\mathrm{exact}}(G_\theta(z_i))
\right)^2.
$$

排序损失为：

$$
\mathcal L_{\mathrm{rank}}=
\frac{1}{M}
\sum_{(i,j)}
\max
\left(
0,
m-
(S_\varphi(z_i)-S_\varphi(z_j))
\cdot
\mathrm{sgn}(s_i-s_j)
\right).
$$

平滑正则为：

$$
\mathcal L_{\mathrm{smooth}}=
\frac{1}{N}\sum_i
|\nabla_{z_i}S_\varphi(z_i)|_2^2.
$$

## 9.3 Best-of-N 重排序

最稳健的生成控制方式是 best-of-N 重排序：

$$
y_i\sim p_\theta(y|c),\qquad i=1,\ldots,N.
$$

选择：

$$
y^*=
\arg\max_{i}
S_{\mathrm{exact}}(y_i).
$$

该方法不修改 ACE-Step 内部结构，不需要反向传播，也不涉及训练主模型，因此最适合作为第一阶段实验基线。

## 9.4 Latent Steering

设 Focus Music latent 均值为：

$$
\mu_F=\frac{1}{N_F}\sum_{i=1}^{N_F}E(x_i^F).
$$

设 Pop Music latent 均值为：

$$
\mu_P=\frac{1}{N_P}\sum_{i=1}^{N_P}E(x_i^P).
$$

定义专注方向：

$$
v_{\mathrm{focus}}=\mu_F-\mu_P.
$$

对生成 latent 进行 steering：

$$
z'=z+\alpha v_{\mathrm{focus}}.
$$

也可以训练线性分类器，其法向量作为 concept activation vector：

$$
v_{\mathrm{CAV}}.
$$

然后：

$$
z'=z+\alpha v_{\mathrm{CAV}}.
$$

## 9.5 Langevin-Style Latent Refinement

定义能量函数：

$$
E_{\mathrm{focus}}(z)=
-S_\varphi(z)
+
\lambda_{\mathrm{prior}}|z-z_0|*2^2
+
\lambda*{\mathrm{bad}}R_{\mathrm{bad}}(z)
+
\lambda_{\mathrm{copy}}R_{\mathrm{copy}}(z).
$$

Langevin 更新为：

$$
z_{k+1}= z_k

\eta\nabla_zE_{\mathrm{focus}}(z_k)
+
\sqrt{2\eta T}\xi_k,
\qquad
\xi_k\sim\mathcal N(0,I).
$$

等价地：

$$
z_{k+1}= z_k + \eta\nabla_zS_\varphi(z_k)

\eta\lambda_{\mathrm{prior}}(z_k-z_0)

\eta\nabla_zR_{\mathrm{bad/copy}}(z_k)
+
\sqrt{2\eta T}\xi_k.
$$

该方法的直觉是：保持 ACE-Step 原始生成先验，同时沿着拓扑评分上升方向做小步调整。

## 9.6 Denoising-Time Topology Guidance

如果可访问 ACE-Step 的 denoising 过程，则可在采样阶段加入 topology guidance。设模型原始 denoising 更新为：

$$
z_{t-1}=f_\theta(z_t,t,c).
$$

加入引导后：

$$
z_{t-1}=
f_\theta(z_t,t,c)
+
\eta_t\nabla_{z_t}S_\varphi(z_t).
$$

也可写作噪声预测修正：

$$
\hat{\epsilon}_\theta = \epsilon_\theta(z_t,t,c)

s_t\sigma_t\nabla_{z_t}S_\varphi(z_t).
$$

该形式借鉴 classifier guidance 的思想，即用外部可微信号改变扩散采样方向。Classifier guidance 在扩散模型中被用于调节条件生成质量和多样性。

------

# 10. 理论框架

## 10.1 不要求全局双利普希茨映射

若要求潜空间与拓扑空间之间存在全局双利普希茨映射，则需要：

$$
c|z_1-z_2|
\le
d_{\mathcal T}(\Psi(z_1),\Psi(z_2))
\le
C|z_1-z_2|.
$$

该条件过强。原因包括：

第一，不同 latent 可能生成拓扑结构相似的音乐；

第二，path homology 描述子是压缩表示，不可能保留全部音频信息；

第三，音乐中不同音色、配器和和声可能共享相似的状态转移拓扑；

第四，生成模型潜空间通常不具备全局可逆性。

因此，本文只要求以下三个更弱条件。

## 10.2 命题一：局部稳定性

**命题 1。** 设 $G_\theta$、特征提取函数 $\phi$ 和图构造函数 $\Gamma$ 在局部区域 $U\subset\mathcal Z$ 上 Lipschitz 连续，且 persistent path diagram 对加权有向图扰动稳定。则存在常数 $C_U>0$，使得：

$$
d_B(D_p(\Psi(z_1)),D_p(\Psi(z_2)))
\le
C_U|z_1-z_2|
$$

对任意 $z_1,z_2\in U$ 成立。

**证明思路。**

由局部 Lipschitz 连续性：

$$
|G_\theta(z_1)-G_\theta(z_2)|
\le
L_G|z_1-z_2|.
$$

$$
|\phi(G_\theta(z_1))-\phi(G_\theta(z_2))|
\le
L_\phi L_G|z_1-z_2|.
$$

$$
|\Gamma(\phi(G_\theta(z_1)))-\Gamma(\phi(G_\theta(z_2)))|
\le
L_\Gamma L_\phi L_G|z_1-z_2|.
$$

由 persistent path diagram 稳定性可得：

$$
d_B(D_p(\Psi(z_1)),D_p(\Psi(z_2)))
\le
L_{\mathrm{PPH}}
L_\Gamma L_\phi L_G
|z_1-z_2|.
$$

令：

$$
C_U=L_{\mathrm{PPH}}L_\Gamma L_\phi L_G.
$$

得证。Persistent path diagram 的稳定性已有相关理论研究支持。

## 10.3 命题二：代理一致性

**命题 2。** 若存在 $\delta>0$，使得：

$$
\sup_{z\in U}
|S_\varphi(z)-S_{\mathrm{exact}}(G_\theta(z))|
\le
\delta,
$$

则对任意 $z_a,z_b\in U$，若：

$$
S_\varphi(z_a)-S_\varphi(z_b)>2\delta,
$$

则：

$$
S_{\mathrm{exact}}(G_\theta(z_a))

> 

S_{\mathrm{exact}}(G_\theta(z_b)).
$$

**证明。**

由误差界：

$$
S_{\mathrm{exact}}(G_\theta(z_a))
\ge
S_\varphi(z_a)-\delta.
$$

$$
S_{\mathrm{exact}}(G_\theta(z_b))
\le
S_\varphi(z_b)+\delta.
$$

因此：

$$
S_{\mathrm{exact}}(G_\theta(z_a))=
S_{\mathrm{exact}}(G_\theta(z_b))
\ge
S_\varphi(z_a)-S_\varphi(z_b)-2\delta>0.
$$

故命题成立。

## 10.4 命题三：局部可控性

**命题 3。** 若 $S_\varphi$ 在 $z_0$ 的邻域内可微且 $L$-smooth，并且：

$$
\nabla_zS_\varphi(z_0)\neq0,
$$

则存在足够小的 $\eta>0$，使：

$$
S_\varphi(z_0+\eta\nabla_zS_\varphi(z_0))

> 

S_\varphi(z_0).
$$

**证明思路。**

由 smooth 函数的一阶增长界：

$$
S_\varphi(z_0+\eta g) \ge S_\varphi(z_0) + \eta\langle\nabla S_\varphi(z_0),g\rangle=
\frac{L\eta^2}{2}|g|^2.
$$

令：

$$
g=\nabla S_\varphi(z_0).
$$

则：

$$
S_\varphi(z_0+\eta g) \ge S_\varphi(z_0) + \eta|g|^2=
\frac{L\eta^2}{2}|g|^2.
$$

只要：

$$
0<\eta<\frac{2}{L},
$$

右侧增量为正。因此局部可控性成立。

------

# 11. 算法

## 算法一：音乐片段的 Path Homology 特征提取

```text
输入：
    音频片段 x
    帧长 Δt
    状态数 K
    filtration 阈值集合 Ε={ε1,...,εL}

输出：
    拓扑特征向量 τ(x)

1:  U ← FrameAndExtractAcousticTemporalFeatures(x, Δt)
2:  # U = [u1,...,uT], ut = [acoustic features, temporal features]

3:  S ← QuantizeStates(U, K)
4:  # S = [s1,...,sT]

5:  W ← BuildDirectedTransitionMatrix(S)
6:  # Wij = P(si → sj)

7:  H_path ← PathEntropy(W)
8:  R_dir ← DirectedRecurrence(S)
9:  S_sparse ← TransitionSparsity(W)
10: N_transition ← TransitionNovelty(U)

11: for each ε in Ε do
12:     Gε ← DirectedGraph(V={1,...,K}, Eε={(i,j): Wij ≥ ε})
13:     Ω0, Ω1, Ω2 ← BuildAllowedPathSpaces(Gε)
14:     ∂1, ∂2 ← BuildBoundaryOperators(Ω1, Ω2)
15:     β0(ε), β1(ε) ← ComputePathBetti(∂1, ∂2)
16: end for

17: D0, D1 ← PersistentPathDiagrams({β0(ε)}, {β1(ε)}, Ε)
18: pers_stats ← DiagramStatistics(D0, D1)
19: C_cycle ← DirectedCyclePersistence(D1)

20: τ(x) ← Concat(
        β0, β1,
        pers_stats,
        H_path,
        R_dir,
        S_sparse,
        C_cycle,
        N_transition
    )

21: return τ(x)
```

## 算法二：拓扑代理评分器训练

```text
输入：
    latent 数据集 Z={zi}
    exact score 标签 yi=S_exact(Gθ(zi))
    margin m
    权重 λr, λs

输出：
    拓扑代理评分器 Sφ

1: 初始化 Sφ
2: repeat
3:     采样 batch {zi, yi}
4:     ŷi ← Sφ(zi)
5:     L_reg ← mean((ŷi - yi)^2)

6:     构造样本对 (zi, zj)
7:     L_rank ← mean(max(0, m - (ŷi - ŷj) * sign(yi - yj)))

8:     L_smooth ← mean(||∇zi Sφ(zi)||^2)

9:     L ← L_reg + λr L_rank + λs L_smooth
10:    使用 AdamW 更新 φ
11: until 验证集 Spearman ρ 与 pairwise accuracy 收敛

12: return Sφ
```

## 算法三：Best-of-N 拓扑重排序

```text
输入：
    prompt c
    采样数量 N
    ACE-Step 模型 Gθ
    exact score S_exact

输出：
    排序后的候选音乐列表

1: candidates ← []
2: for i = 1,...,N do
3:     yi ← SampleFromACEStep(c)
4:     si ← S_exact(yi)
5:     candidates.append((yi, si))
6: end for
7: candidates ← SortDescending(candidates, key=score)
8: return candidates
```

## 算法四：Latent Langevin Guidance

```text
输入：
    初始 latent z0
    拓扑代理评分器 Sφ
    步长 η
    步数 K
    温度 T
    prior 正则 λprior
    copy 正则 λcopy
    quality 正则 λbad

输出：
    引导后的 latent zK

1: z ← z0
2: for k = 1,...,K do
3:     g ← ∇z Sφ(z)
4:     g ← g - λprior (z - z0)
5:     g ← g - λcopy ∇z R_copy(z)
6:     g ← g - λbad ∇z R_bad(z)
7:     g ← ClipByNorm(g, gmax)
8:     ξ ~ N(0, I)
9:     z ← z + ηg + sqrt(2ηT)ξ
10: end for
11: return z
```

## 算法五：Denoising-Time Topology Guidance

```text
输入：
    初始噪声 zT
    条件 c
    ACE-Step denoiser εθ
    拓扑代理评分器 Sφ
    guidance scale 调度 {st}

输出：
    生成音频 y

1: z ← zT
2: for t = T,...,1 do
3:     ε ← εθ(z, t, c)
4:     g ← ∇z Sφ(z)
5:     g ← ClipByNorm(g, gmax)
6:     ε_hat ← ε - st * σt * g
7:     z ← ReverseDenoisingStep(z, ε_hat, t)
8: end for
9: y ← Decode(z)
10: return y
```

------

# 12. 实验设计

## 12.1 实验一：Brain.fm Focus 与 Spotify Pop 拓扑差异

**目的：** 验证两类音乐是否在有向状态转移图和 path homology 特征上存在显著差异。

**输入：**

$$
\mathcal D_F,\quad \mathcal D_P,\quad \mathcal D_C.
$$

**方法：**

1. 提取短时声学与时间结构特征；
2. 构建有向状态转移图；
3. 计算 path homology；
4. 计算 path entropy、directed recurrence、transition sparsity；
5. 使用统计检验比较组间差异。

**统计方法：**

若数据近似正态，使用 Welch t-test；否则使用 Mann–Whitney U test。多重检验使用 Benjamini–Hochberg FDR 校正。

## 12.2 实验二：拓扑特征分类有效性

比较以下特征组：

| 特征组                         | 内容                                                         |
| ------------------------------ | ------------------------------------------------------------ |
| Acoustic-only                  | MFCC、chroma、loudness、spectral centroid、onset density     |
| Temporal-only                  | recurrence、transition novelty、self-similarity              |
| Topology-only                  | path Betti numbers、persistent path statistics、path entropy、directed recurrence |
| Acoustic + Temporal            | 声学特征 + 时间结构特征                                      |
| Acoustic + Temporal + Topology | 完整模型                                                     |

分类模型包括：

$$
\mathrm{Logistic\ Regression},
\mathrm{SVM},
\mathrm{Random\ Forest},
\mathrm{XGBoost},
\mathrm{MLP}.
$$

评价指标：

$$
\mathrm{Accuracy},\quad
\mathrm{Macro\text{-}F1},\quad
\mathrm{Balanced\ Accuracy},\quad
\mathrm{AUROC},\quad
\mathrm{AUPRC}.
$$

## 12.3 实验三：拓扑流形可视化

使用 UMAP、PCA 或 diffusion maps 对 $\tau(x)$ 进行降维可视化。重点观察：

1. Brain.fm Focus 是否形成相对集中区域；
2. Spotify Pop 是否与 Brain.fm Focus 分离；
3. Matched Control 是否仍与 Brain.fm Focus 存在拓扑差异；
4. ACE-Step 生成样本是否能通过引导向 $\mathcal M_F$ 靠近。

## 12.4 实验四：局部稳定性

对 latent 加扰动：

$$
z'=z+\epsilon u,\qquad u\sim\mathcal N(0,I).
$$

计算：

$$
\rho(\epsilon)=
\frac{
d_{\mathcal T}(\Psi(z),\Psi(z'))
}{
|z-z'|
}.
$$

若 $\rho(\epsilon)$ 在小扰动区域内有界，说明局部稳定性成立。

## 12.5 实验五：代理评分器一致性

比较：

$$
S_\varphi(z)
$$

与：

$$
S_{\mathrm{exact}}(G_\theta(z)).
$$

指标包括：

$$
R^2,\quad
\mathrm{Spearman}\ \rho,\quad
\mathrm{Kendall}\ \tau,\quad
\mathrm{Pairwise\ Accuracy},\quad
\mathrm{Top\text{-}k\ Overlap}.
$$

## 12.6 实验六：生成控制实验

比较以下生成方法：

| 方法               | 描述                                |
| ------------------ | ----------------------------------- |
| Base               | ACE-Step 原始 prompt 生成           |
| Best-of-N          | 生成多个候选后按 exact score 重排序 |
| Latent Steering    | 沿 focus direction 移动 latent      |
| Langevin Guidance  | 使用代理评分器做潜空间迭代优化      |
| Denoising Guidance | 在去噪过程中加入拓扑梯度            |

评价指标：

$$
S_{\mathrm{exact}},
\quad
d_{\mathcal T}(\tau(y),\mathcal M_F),
\quad
d_{\mathcal T}(\tau(y),\mathcal M_P),
\quad
H_{\mathrm{path}},
\quad
R_{\mathrm{dir}},
\quad
C_{\mathrm{cycle}},
\quad
S_{\mathrm{sparse}},
\quad
N_{\mathrm{transition}}.
$$

同时评估：

1. 音质；
2. 多样性；
3. 是否过度重复；
4. 与授权音乐的相似度；
5. 人类主观分心度评分。

------

# 13. 复杂度分析

设状态数为 (K)，时间帧数为 (T)，过滤层数为 (L)。

状态转移矩阵构建复杂度为：

$$
O(T).
$$

边数上界为：

$$
|E|\le K^2.
$$

低阶 path homology 计算主要涉及边界矩阵秩计算。若仅计算 $p\le1$，复杂度主要由 $\partial_1$ 与 $\partial_2$ 的稀疏矩阵秩决定。最坏情况下复杂度较高，但实际可通过以下方式降低：

1. 限制 $K\le128$；
2. 仅计算 $\beta_0^{path}$ 与 $\beta_1^{path}$；
3. 使用稀疏矩阵；
4. 使用 top-k 边或阈值过滤；
5. 使用滑动窗口并行计算；
6. 对 filtration 层数 $L$ 取 16 或 24。

已有研究关注一维 persistent path homology 的高效算法，说明 $H_1$ 层面的计算可以通过结构性优化降低成本。

------

# 14. 预期结果与结果报告模板

由于本文为研究设计与方法论文草稿，实际数值需在实验完成后填入。预期结果如下。

第一，Brain.fm Focus 的 path entropy 低于 Spotify Pop，说明其状态转移更可预测。

第二，Brain.fm Focus 的 directed recurrence 高于 Spotify Pop，说明其有向状态转移模式更稳定复现。

第三，Brain.fm Focus 的 persistent path homology 特征在拓扑空间中形成相对集中的区域。

第四，Topology-only 特征组的分类性能应高于 acoustic-only 的一部分任务，Acoustic + Temporal + Topology 应达到最佳或接近最佳表现。

第五，Best-of-N 重排序可显著提升生成音乐的 exact topology score。

第六，Langevin guidance 能进一步降低生成音乐与 $\mathcal M_F$ 的距离，但过强 guidance 可能损害音质。

推荐结果表如下：

| 方法      | $S_{\mathrm{exact}}$ | $d(\mathcal M_F)$ | $d(\mathcal M_P)$ | $H_{\mathrm{path}}$ | $R_{\mathrm{dir}}$ | 音质 |
| --------- | -------------------- | ----------------- | ----------------- | ------------------- | ------------------ | ---- |
| Base      | 待填                 | 待填              | 待填              | 待填                | 待填               | 待填 |
| Best-of-N | 待填                 | 待填              | 待填              | 待填                | 待填               | 待填 |
| Steering  | 待填                 | 待填              | 待填              | 待填                | 待填               | 待填 |
| Langevin  | 待填                 | 待填              | 待填              | 待填                | 待填               | 待填 |
| Denoising | 待填                 | 待填              | 待填              | 待填                | 待填               | 待填 |

------

# 15. 讨论

## 15.1 拓扑结构是否足以定义专注音乐

本文并不声称拓扑结构是专注音乐的唯一决定因素。专注音乐还受到音色、响度、频谱平衡、文化偏好、个人习惯和任务类型影响。但本文认为，拓扑结构能够提供一种比单帧声学特征更高层的时间组织描述。

如果实验显示 Brain.fm Focus 在 path homology 空间中形成稳定区域，则说明功能性专注音乐至少具有某些可计算的有向时间结构特征。如果实验结果不显著，也能说明当前状态离散化或图构建方法尚不足以捕捉其结构，或说明专注音乐的关键特征主要位于其他层面。

## 15.2 为什么使用 Path Homology 而非普通 Persistent Homology

普通 persistent homology 适合分析点云和无向距离结构，但音乐具有天然方向性。音乐从 (s_t) 到 (s_{t+1}) 的转移与反向转移并不等价。Persistent path homology 的价值正在于其能处理 directed networks，并编码非对称结构。 因此，path homology 更适合本文的音乐时间结构建模。

## 15.3 为什么不直接微调 ACE-Step

直接微调 ACE-Step 主模型存在三个问题。

第一，版权风险较高。Spotify 官方政策限制将 Spotify Content 用于 AI/ML 训练。

第二，直接微调可能导致模型学习具体曲风或曲目特征，而非抽象拓扑结构。

第三，推理阶段控制更轻量、更可解释，也更适合作为科学竞赛项目落地。

因此，本文优先采用：

$$
\mathrm{analysis}
\rightarrow
\mathrm{topology\ score}
\rightarrow
\mathrm{inference\ guidance}
$$

而不是：

$$
\mathrm{copyrighted\ audio}
\rightarrow
\mathrm{fine\ tuning}
\rightarrow
\mathrm{style\ imitation}.
$$

------

# 16. 局限性

第一，Brain.fm Focus 只能作为功能性专注音乐参考集，不能直接等同于临床疗效标准。本文不作医学诊断或治疗声称。

第二，Spotify Pop 的使用必须严格受授权范围约束。若授权不允许训练或生成优化，则只能用于分析和评价。

第三，path homology 计算复杂度较高，尤其是状态数 $K$ 较大时。因此主实验应限制在低阶 path homology。

第四，状态离散化方法会显著影响有向图结构。如果 $K$ 过小，会丢失细节；如果 $K$ 过大，会导致图过稀疏。

第五，拓扑代理评分器可能学习到 exact score 的偏差。因此必须使用 held-out exact path homology 结果验证。

第六，潜空间引导可能牺牲音质。需要通过 prior regularization、gradient clipping 和人工听感评价控制风险。

------

# 17. 伦理、版权与合规声明

本文使用 Brain.fm 与 Spotify 音频的前提是已获得合法授权。所有原始音频仅用于本地研究分析，不公开、不再分发、不作为公开训练集发布。

Spotify 官方开发者政策明确限制将 Spotify Content 用于机器学习或 AI 模型训练。 因此，除非额外授权明确覆盖相关用途，本文不使用 Spotify 音频训练或微调任何生成模型。Spotify Pop 默认仅作为分析、对照和评价数据使用。

本文生成音乐的目标不是复制 Brain.fm 或 Spotify 中的任何曲目，而是根据抽象拓扑指标进行原创生成。为降低复制风险，本文加入：

$$
R_{\mathrm{copy}}(y)
$$

作为惩罚项，并在结果中报告与授权参考音乐的相似度。

如进行人类听感评价，应采用匿名随机音频编号，不暴露数据来源。若涉及未成年人或注意困难群体，需要伦理审批与监护人同意。本文不进行临床疗效宣称。

------

# 18. 未来工作

未来工作包括：

第一，改进状态离散化方法，例如使用 VQ-VAE、self-supervised audio tokenizer 或 contrastive temporal clustering。

第二，扩展 path homology 计算到更高阶结构，但需要控制复杂度。

第三，引入 soft graph construction，使 latent perturbation 到 graph topology 的映射更加平滑。

第四，研究不同任务场景下的专注音乐拓扑结构，例如阅读、写作、编程、数学推理和绘画。

第五，在严格伦理审查下，结合实际持续注意任务评价生成音乐的行为效果。

第六，将 topology-guided generation 扩展到其他功能性音频，如睡眠音乐、冥想音乐、运动节奏音乐和情绪调节声音设计。

------

# 19. 结论

本文提出了一套基于 path homology 的专注音乐拓扑流形建模与生成控制方法。研究将音乐视为有向动态系统，从短时声学与时间结构特征出发，构建加权有向状态转移图，并使用 path homology 与 persistent path homology 描述音乐状态演化的拓扑结构。该方法保留了音乐时间方向性，能够捕捉普通静态声学特征难以表达的状态转移模式。

在生成阶段，本文基于 ACE-Step 1.5 的潜空间结构，提出 topology-guided latent intervention。该方法不依赖对主生成模型的重训，也不要求潜空间与音乐物理空间之间存在全局双利普希茨映射；相反，本文以局部稳定性、代理一致性和局部可控性作为理论基础，并通过 best-of-N、latent steering、Langevin refinement 和 denoising-time guidance 实现推理阶段控制。

本文的核心观点是：专注音乐可以被建模为一种具有低分叉、高复现、稳定有向循环和可预测状态转移的时间拓扑结构。Path homology 为这种结构提供数学表达，ACE-Step 潜空间为这种结构提供生成控制入口。若后续实验验证成立，本文将为功能性音乐分析、拓扑数据分析与生成式音乐 AI 的结合提供一种新的研究范式。

------

# 参考文献

[1] Brain.fm. “Our Science.” Brain.fm official science page.

[2] A. Grigor’yan, Y. Lin, Y. Muranov, and S.-T. Yau. “Homologies of path complexes and digraphs.” arXiv:1207.2834, 2012.

[3] S. Chowdhury and F. Mémoli. “Persistent Path Homology of Directed Networks.” arXiv:1701.00565, 2017.

[4] S. Zhang. “Stability of Persistent Path Diagrams.” arXiv:2406.11998, 2024.

[5] T. K. Dey, T. Li, and Y. Wang. “An efficient algorithm for 1-dimensional persistent path homology.” arXiv:2001.09549, 2020.

[6] J. Gong, Y. Song, W. Zhao, S. Wang, S. Xu, and J. Guo. “ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation.” arXiv:2602.00744, 2026.

[7] Spotify for Developers. “Developer Policy.”

[8] J. Song, C. Meng, and S. Ermon. “Score-Based Generative Modeling through Stochastic Differential Equations.” arXiv:2011.13456, 2020.

[9] P. Dhariwal and A. Nichol. “Diffusion Models Beat GANs on Image Synthesis.” arXiv:2105.05233, 2021.