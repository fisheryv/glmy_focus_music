# LTSN 网络架构与训练方案

日期：2026-08-03  
适用指纹：`focus_path_homology_fingerprint_v2`  
指纹 SHA-256：`9bf64f3c1d79c12ec428f1d9f552827d07e9f5c445d9236e7ab676699a62ef1f`  
状态：设计完成；尚未实现、训练或通过资格检验

## 摘要

LTSN（Latent Topology Surrogate Network）是一个小型、时间条件化、可微的时序网络。
它接收 ACE-Step 1.5 Turbo 某一步的预测干净潜变量

\[
\hat x_0=x_t-t v_t,
\qquad \hat x_0\in\mathbb R^{B\times T\times64},
\]

并预测当前纯 Path Homology v2 指纹的 51 维坐标及其不确定度。51 维由
Pitch 16、Rhythm 16、Modulation 17、Acoustic phase 1 和 Chroma phase 1
组成；Structure PH 不进入网络的主输出和采样梯度。

LTSN 不是拓扑算法的替代品。Exact Path Homology 仍负责离线教师标签和最终裁判；
LTSN 只提供采样期所需的可微近似。任何代理分数改善都必须通过解码后 exact PH
复核，否则视为 proxy gaming。

![LTSN 网络架构](../runs/ltsn_design/figures/ltsn_network_architecture.png)

[LTSN 网络架构 SVG](../runs/ltsn_design/figures/ltsn_network_architecture.svg)

## 1. 设计目标与边界

### 1.1 网络要学习什么

冻结指纹使用 discovery/180 s 拟合的块变换。令各块白化坐标为 \(Z_b\)，则真正
进入冻结分类器的 51 维 `L+P` 坐标是

\[
z=\left[
\frac{Z_{pitch}}{\sqrt6}\middle|
\frac{Z_{rhythm}}{\sqrt6}\middle|
\frac{Z_{modulation}}{\sqrt6}\middle|
\frac{Z_{A\text{-}phase}}{2}\middle|
\frac{Z_{C\text{-}phase}}{2}
\right]
\in\mathbb R^{51}.
\]

这里的系数来自 `L` 内三块等权、`P` 内两块等权以及最终 `L/P` 两尺度等权。LTSN
必须预测这个已缩放的 51 维坐标，而不是未加层级尺度的原始块拼接。

LTSN 学习条件分布

\[
g_\phi(\hat x_0,t)\rightarrow
(\hat\mu_z,\log\hat\sigma_z^2),
\]

其中 \(\hat\mu_z\in\mathbb R^{51}\) 是坐标预测，
\(\log\hat\sigma_z^2\in\mathbb R^{51}\) 是异方差不确定度。

Focus logit 不使用另一个自由的分类头，而由冻结指纹系数确定性计算：

\[
\hat S_F=w^\top\hat\mu_z+b.
\]

这样可以保证“代理坐标”与“代理 Focus 分数”一致，减少网络只迎合总分、却破坏
单个视角的机会。

### 1.2 网络明确不学习什么

- 不预测或最大化历史 Vietoris–Rips TDA 端点；
- 不把 Structure PH 并入主输出，因为结构对 `L+P` 没有正增量；
- 不单独最大化 H1/H2；
- 不输入 prompt 文本、Focus/Classical 标签或冻结 exact 分数，避免信息捷径；
- 不把 LTSN 输出解释成注意力、疗效、生产率或生成质量概率。

## 2. 输入与时间条件

### 2.1 ACE-Step 潜变量

ACE-Step 1.5 Turbo 的声学潜变量为 64 维、25 Hz。180 s 音频约有

\[
T=180\times25=4500
\]

个潜帧。训练和引导使用 `x0_hat`，而不是只使用最终 `pred_latents`。在每个采样步，
由真实的 \(x_t\)、速度 \(v_t\) 和噪声时间 \(t\) 构造该步自己的
\(\hat x_0\)。

### 2.2 为什么必须输入时间步

相同的 \(\hat x_0\) 误差形态会随噪声时间改变；第 4 步和第 6 步的预测置信度也不
相同。因此把标量 \(t\) 和离散 step id 编码为 128 维条件向量：

\[
e_t=\operatorname{MLP}(\operatorname{Fourier}(t,\text{step}))\in\mathbb R^{128}.
\]

它通过 FiLM/AdaLN 调制 TCN 和 Transformer：

\[
\operatorname{FiLM}(h,e_t)=\gamma(e_t)\odot\operatorname{Norm}(h)+\beta(e_t).
\]

## 3. 网络架构

### 3.1 Stem：把 25 Hz 长序列压缩到可处理长度

输入先做逐帧 LayerNorm，再转为 Conv1d 布局 `[B,64,T]`。Stem 使用：

```text
Conv1d(64, 128, kernel=9, stride=5, padding=4)
GroupNorm → SiLU → Dropout(0.05)
```

输出长度约为 900，每个 token 间隔约 0.2 s。kernel=9 在下采样前覆盖约 0.36 s，
足以整合局部潜帧而不过早抹去节奏边界。

### 3.2 局部 TCN：学习短时状态转移线索

Stem 后用 1×1 卷积投影到 192 通道，再串联 4 个残差 TCN block：

| block | kernel | dilation | 通道 | 作用 |
|---|---:|---:|---:|---|
| 1 | 3 | 1 | 192 | 邻近潜 token |
| 2 | 3 | 2 | 192 | 短节奏/音高转移 |
| 3 | 3 | 4 | 192 | 数秒局部组织 |
| 4 | 3 | 8 | 192 | 更长局部上下文 |

每个 block 使用 depthwise Conv1d、pointwise 1×1、FiLM、SiLU、dropout 0.1 和
残差连接。局部分支保留长度 900，服务于 Pitch、Rhythm、Modulation 坐标。

### 3.3 中尺度 TCN：覆盖相位闭环周期

局部特征再经 `Conv1d(192,256,kernel=7,stride=4)` 降至约 225 token，时间间隔约
0.8 s。随后使用 dilation 1/4/16/64 的四个 256 通道残差块。

扩张卷积和后续全局注意力共同覆盖从局部状态转移到数十秒相位周期的范围。这里只
学习 Acoustic/Chroma phase 的可微代理；exact `loop_score` 中的周期 `argmin`、
分箱和最弱边 `min` 仍留在教师管线。

### 3.4 全局编码器

225 token 已允许低成本全长注意力。推荐两个 Transformer encoder block：

```text
d_model=256, heads=8, FFN=1024, dropout=0.1, pre-norm
```

该层用于连接相距较远的重复位置。它不是 Structure PH 分支，不预测宏观段落拓扑。

### 3.5 多尺度汇聚

对局部序列和全局序列分别计算 masked attention、均值和标准差：

\[
p(h)=[p_{attn}(h)|\operatorname{mean}(h)|\operatorname{std}(h)].
\]

- 局部分支：\(192\times3=576\) 维；
- 中/全局分支：\(256\times3=768\) 维；
- 时间嵌入：128 维。

拼接为 1472 维，再经

```text
Linear(1472,512) → SiLU → Dropout(0.1)
Linear(512,256)  → SiLU
```

得到共享表征。整个网络预计约 3–4M 参数，不包含 ACE-Step、VAE 或 exact PH。

### 3.6 输出头

共享 256 维表征连接：

1. `coordinate_mean_head`：输出 51 维 \(\hat\mu_z\)；
2. `coordinate_logvar_head`：输出 51 维 \(\log\hat\sigma_z^2\)，限制在合理区间；
3. `ood_head`：可选 1 维安全/OOD logit；
4. 冻结确定性读出：用 v2 JSON 中的 \(w,b\) 计算 \(\hat S_F\)。

建议训练 3 个不同随机种子的 LTSN。推理时把成员间方差作为 epistemic uncertainty，
与单模型的 aleatoric uncertainty 合并。任一不确定度或 OOD 指标超过冻结阈值时，
topology corrector 必须 no-op。

## 4. 教师标签如何构建

### 4.1 每个快照必须有自己的标签

训练样本不是“中间 latent + 最终音频标签”。正确流程是：

1. 在 ACE-Step 无引导生成中保存 step 4、5、6 和最终 step 的 \(x_t,v_t,t\)；
2. 计算每一步的 \(\hat x_0=x_t-tv_t\)；
3. 分别用 VAE 解码每个 \(\hat x_0\)；
4. 分别运行 exact Pitch/Rhythm/Modulation/phase PH；
5. 使用冻结 v2 块变换得到各自的 51 维 \(z\) 和 Focus logit。

如果把最终标签复制给所有中间步骤，网络会学到错误的时间不变映射，采样梯度也会
系统性偏离真实 exact 目标。

### 4.2 建议的数据组成

只有当前 390 首 discovery/180 s 真实音频不足以覆盖生成潜空间。建议：

- 锚点：195 Focus + 195 Classical discovery 音频的 VAE 重建潜变量；
- 主体：新的 development prompt 集，建议至少 512 prompt × 4 seeds = 2048 条
  无引导 180 s 轨迹；
- 快照：每条轨迹标注 step 4/5/6/8，约 8192 个生成快照；
- 安全/OOD：额外 10–15% 静音、削波、极低动态、机械短循环和异常 latent；
- Qualification：另设未参与训练和超参数选择的 prompt/seed family。

数字是工程起点，不是已验证的最优样本量。如果 exact reranking 尚未证明生成候选
空间具有可辨识的 PH 差异，不应先投入大规模 LTSN 标注。

![LTSN 训练流程](../runs/ltsn_design/figures/ltsn_training_pipeline.png)

[LTSN 训练流程 SVG](../runs/ltsn_design/figures/ltsn_training_pipeline.svg)

## 5. 防泄漏切分

必须按 prompt 或真实曲目分组，而不是按 latent 快照或 seed 随机切分。同一生成轨迹
的 step 4/5/6/8 必须留在同一分区。

建议四层切分：

| 分区 | 建议比例 | 用途 |
|---|---:|---|
| train | 70% prompts | 拟合权重 |
| development | 15% prompts | 选择架构、损失权重和 early stopping |
| calibration | 15% prompts | 温度缩放、预测区间和 OOD 阈值 |
| qualification | 额外独立 prompts | 冻结后一次资格检验 |

当前音乐研究的 validation 和已开启 holdout 不应拿来继续调 LTSN。它们描述的是冻结
声学指纹证据，不是无限复用的代理开发集。

## 6. 损失函数

### 6.1 分层块平衡坐标损失

若直接对 51 维平均，49 个局部坐标会淹没两个 phase 坐标。按照 v2 的层级权重，
定义块权重

\[
\omega_{pitch}=\omega_{rhythm}=\omega_{modulation}=\frac16,
\qquad
\omega_{Aphase}=\omega_{Cphase}=\frac14.
\]

对每块先在维内平均，再在块间加权：

\[
L_{coord}=\sum_b\omega_b\frac1{d_b}
\sum_{j\in b}\operatorname{Huber}(\hat\mu_j-z_j).
\]

### 6.2 异方差负对数似然

令 \(s_j=\log\hat\sigma_j^2\)，则

\[
L_{NLL}=\sum_b\omega_b\frac1{d_b}\sum_{j\in b}
\frac12\left[e^{-s_j}(z_j-\hat\mu_j)^2+s_j\right].
\]

实现时应限制 \(s_j\) 的范围，避免通过无限放大方差逃避误差。

### 6.3 Focus 分数一致性

使用冻结读出得到 \(\hat S_F\)，并约束：

\[
L_{score}=\operatorname{Huber}(\hat S_F-S_F).
\]

这项损失强化与最终引导能量最相关的方向，但不能替代 51 维坐标损失。

### 6.4 同 prompt 排序损失

在同 prompt 的不同 seed 之间构造分数对，只使用 exact 分数差超过预设 margin 的
样本：

\[
L_{rank}=\log\left(1+\exp\left[
-y_{ik}(\hat S_i-\hat S_k)/\tau\right]\right),
\quad y_{ik}=\operatorname{sign}(S_i-S_k).
\]

它直接训练 LTSN 复现 reranking 顺序。

### 6.5 轨迹增量损失

不能简单要求相邻 step 的预测平滑，因为真实 PH 坐标也在变化。应匹配 exact 增量：

\[
L_{\Delta}=\operatorname{Huber}
\left[(\hat\mu_{s+1}-\hat\mu_s)-(z_{s+1}-z_s)\right].
\]

### 6.6 OOD 损失与总损失

安全负例使用二元交叉熵 \(L_{OOD}\)。建议的 development 起点为

\[
L=L_{coord}+0.25L_{NLL}+0.5L_{score}
+0.2L_{rank}+0.2L_{\Delta}+0.1L_{OOD}.
\]

这些系数不是新的实证结论。它们只能在 development 分区上选择一次，并在
qualification 前与模型结构、数据哈希一起冻结。

## 7. 训练配置

建议起始配置：

| 项目 | 配置 |
|---|---|
| optimizer | AdamW，lr `3e-4`，weight decay `1e-2` |
| schedule | 5% warmup + cosine decay，最低 lr `1e-6` |
| precision | bf16；损失与方差计算保留 fp32 |
| batch | 每卡 8–16，梯度累积至 effective batch 64 |
| gradient clip | global norm 1.0 |
| regularization | dropout 0.1；不使用破坏标签的随机裁剪/时间打乱 |
| sampler | step 4/5/6/8 平衡；Focus logit 分位数平衡；OOD 10–15% |
| stopping | development `L_score` 与 block Spearman 联合 early stopping |
| ensemble | 三个 seed；架构和数据完全一致 |

训练时不加载 ACE-Step DiT 或 VAE，只读取预计算 latent 和 exact 标签，因此网络训练
本身较轻；成本主要来自轨迹生成、四次 VAE 解码和 exact PH 标注。

不应对 180 s latent 做随机时间裁剪后仍沿用整曲标签，也不应把时间打乱当作普通
augmentation。若要训练短片段版本，必须为该片段重新计算 exact 标签，并把它作为
不同尺度模型处理。

## 8. 资格检验与校准

冻结后的 qualification 集必须按每个采样 step 分层报告，不能只给所有 step 的平均。
沿用当前引导方案的最低门槛：

- Focus logit Spearman \(\rho\ge0.70\)；
- 51 个坐标的 Spearman 中位数 \(\rho\ge0.50\)；
- Pitch、Rhythm、Modulation、Phase 各块距离 \(\rho\ge0.50\)；
- exact 高/低四分位排序准确率 ≥0.65；
- 90% 预测区间覆盖率在 0.85–0.95；
- OOD 或高不确定样本能够触发 no-op；
- 在未见 latent 上按代理梯度优化后，解码音频的 exact score 必须同向改善。

还应报告：MAE、RMSE、校准曲线、每块误差、每步误差、prompt-cluster bootstrap CI、
OOD AUROC、推理耗时和显存。任何主块失败都不能被总体 logit 相关掩盖。

## 9. 训练完成后如何进入采样器

只有资格通过后，才能在 step 4–6 对 detached \(\hat x_0\) 开启梯度：

\[
E(\hat x_0)=\left[\max(0,\tau_F-\hat S_F(\hat x_0))\right]^2.
\]

DiT 保持 `no_grad`；仅令 \(\hat x_0\) 对 LTSN 求导：

\[
g=\nabla_{\hat x_0}E,
\qquad
\Delta x=-\eta_s\,\operatorname{RMSClip}(g,c_s).
\]

推荐只比较 0.25%、0.5%、1.0% latent RMS 三个预设强度，并只在 development 选择
一次。若 ensemble disagreement、预测区间宽度或 OOD 超过阈值，则
\(\Delta x=0\)。每个 guided 输出仍须解码后执行 exact PH、质量检查和盲听非劣检验。

## 10. 建议实施顺序

1. 先完成纯 PH v2 的 exact reranking 可辨识性实验；
2. 增加 ACE-Step 中间轨迹记录，不改变采样结果；
3. 建立带哈希的 latent/label 数据集；
4. 实现 LTSN、损失和 prompt-group sampler；
5. 在 train/development 拟合并选择一次架构；
6. 在 calibration 冻结不确定度和 OOD 阈值；
7. 在独立 qualification prompts 上执行一次资格检验；
8. 通过后才开发 step 4–6 的弱梯度 corrector；
9. 最后以新的配对 baseline/guided 数据做 exact 与质量确认。

任一步失败都应停止升级并退回 shadow mode 或 exact reranking，不能用失败数据继续
改变冻结拓扑目标。

## 11. 建议代码边界

```text
src/generation/path_homology_surrogate.py
src/generation/ltsn_dataset.py
src/generation/ltsn_losses.py
src/generation/topology_corrector.py
scripts/collect_ltsn_trajectories.py
scripts/build_ltsn_labels.py
scripts/train_path_homology_surrogate.py
tests/test_path_homology_surrogate.py
```

模型 checkpoint 必须记录指纹 JSON、状态模型、LTSN 配置、训练清单、切分清单和
exact 标签表的 SHA-256。任何一个上游哈希改变，都应生成新的模型版本，而不是覆盖
旧 checkpoint。

## 12. 结论

推荐的 LTSN 是“多尺度卷积/TCN + 小型全局 Transformer + 分层坐标/不确定度头”的
轻量代理。它同时保留局部状态转移和中尺度相位信息，并通过冻结确定性读出对齐
51 维纯 Path Homology v2 指纹。最关键的训练原则不是网络更大，而是：每个
`x0_hat` 快照单独 exact 标注、按 prompt 防泄漏切分、块平衡损失、不确定度 no-op、
以及代理优化后的 exact 复核。

在这些门槛通过以前，LTSN 仍是待验证工程设计，不能被描述为已经实现的 ACE-Step
采样期拓扑引导。
