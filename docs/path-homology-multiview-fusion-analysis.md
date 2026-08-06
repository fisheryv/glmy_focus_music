# Path Homology 多视角融合最终分析

生成日期：2026-08-02。局部三视角采用音高、节奏、调制等权融合；结构以 0.5 权重与局部块做层级融合。所有块变换仅在 discovery 拟合。

## 冻结结论

- validation/180 s：pitch pseudo-F=7.588，local=6.696，hierarchical=4.291。
- holdout/180 s：pitch=7.402，local=5.359，hierarchical=3.853；三者 p 均为 0.001。
- local 相对 pitch 的 holdout 增量 Δpseudo-F=-2.044、单侧 p=1.000。
- 加入 structure 的 holdout 增量 Δpseudo-F=-1.506、单侧 p=1.000。

结论：局部融合可以区分两组，但没有超过音高；结构也没有提供正增量。组间显著不等于互补信息增加。权重在 holdout 前锁定，之后没有改变。

![Validation ablation](../runs/multiview_fusion/figures/multiview_permanova_ablation.png)

[SVG](../runs/multiview_fusion/figures/multiview_permanova_ablation.svg)

![Holdout endpoints](../runs/symmetric_holdout_final/figures/holdout_frozen_endpoints.png)

[SVG](../runs/symmetric_holdout_final/figures/holdout_frozen_endpoints.svg)

证据边界：融合设计参考过既往单视角结果，因此 validation 融合仍属探索性整合；holdout 是重切分后的操作性最终确认，不是 pristine 外部确认。
