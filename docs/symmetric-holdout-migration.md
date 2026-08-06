# 对称 Holdout 切分迁移报告

生成日期：2026-08-02。切分版本：`symmetric_holdout_v2`。

## 1. 结果

规范 Open Focus 与 Classical 数据现均采用：

| 组别 | discovery | validation | holdout | 总计 |
|---|---:|---:|---:|---:|
| Open Focus | 195 | 60 | 45 | 300 |
| Classical | 195 | 60 | 45 | 300 |
| 合计 | 390 | 120 | 90 | 600 |

Open Focus 原有切分完全未变。Classical 有 184 首曲目发生迁移：旧 discovery 中
60 首进入 validation、45 首进入 holdout；旧 validation 的 79 首全部进入 discovery。

## 2. Classical 分组

Classical 继续以 `composer_key` 为不可拆分单元，同一作曲家及其作品不得跨 split。

| Split | 作曲家 |
|---|---|
| discovery | Bach、Beethoven、Borodin、Cambini、Haydn |
| validation | Mozart、Schubert |
| holdout | Brahms、Dvořák、Fauré、Mendelssohn、Ravel、Suk |

作曲家泄漏为 0，album/work 泄漏为 0。validation 同时包含钢琴独奏、弦乐四重奏、
弦乐室内乐和混合室内乐。

Classical holdout 的组成限制必须显式保留：在“精确 195/60/45”“作曲家完全隔离”以及
“validation 保留钢琴与室内乐覆盖”三个约束同时成立时，Bach 和 Beethoven 只能留在
discovery，因此 holdout 没有 `piano_solo`。其45首由 mixed chamber 21、string chamber
11、string quartet 13 构成。未来不能把该 holdout 解释为完整 Classical 风格总体的
无偏抽样。

## 3. 规范产物

- `metadata/control_classical.csv`：新的逐曲 Classical split；
- `metadata/split_discovery.csv`、`split_validation.csv`、`split_holdout.csv`：两组规范切分；
- `metadata/split_assignment_v2.csv`：600首曲目的单表切分；
- `metadata/classical_split_change_log.csv`：184首迁移曲目的旧/新标签；
- `metadata/symmetric_holdout_audit.json`：计数、作曲家、子池、泄漏、哈希和失效清单；
- `metadata/dataset_summary.json`、`control_dataset_summary.json`：更新后的规范汇总。

元数据校验结果：600首曲目与600条许可记录完整；discovery/validation/holdout 分别为
390/120/90；重复分配为0；Classical 候选审计错误和警告均为0。

## 4. 既有结果的状态

不能仅把旧派生表中的 `split` 字段替换为新标签。旧状态码本曾使用 Classical 221首
discovery 拟合，其中包含当前 validation 与 holdout 曲目；同时旧 validation 的79首
现在进入 discovery。因此以下产物全部降级为历史结果：

- `metadata/preprocessed_segments.csv` 及旧 split 路径中的 WAV；
- `metadata/feature_segments.csv` 和各视角状态特征；
- pitch、rhythm、modulation、structure 的图、持续同调与统计表；
- 两阶段多视角融合结果；
- 依赖旧 discovery 目标分布的生成重排序或代理模型。

完整的38份失效元数据清单见 `metadata/symmetric_holdout_audit.json`。四个单视角报告和
多视角融合报告已添加历史结果警告。

## 5. 后续重跑顺序

1. 按新 split 重新生成/整理 180 s 与 300 s 预处理路径和清单；
2. 仅用两组各195首 discovery/180 s 重新拟合所有状态模型；
3. 重新转换全部 discovery、validation 和 holdout；
4. 重跑四视角 Path Homology；
5. 先在 validation（60+60）完成冻结检验；
6. 只有分析方案冻结后，才开启 holdout（45+45）进行一次最终确认；
7. holdout 结果不用于再调参数、挑指标或改变融合权重。

旧 validation 已被反复查看，因此新 validation 也不能自动恢复为“从未触碰”的确认性
测试集；真正新的最终确认应以当前对称 holdout 为准，并在开启前冻结全部方法。

