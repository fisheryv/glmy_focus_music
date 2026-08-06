# Brain.fm 到 Open Focus 迁移报告

> 历史迁移记录：此报告描述 Pop 尚属规范对照时的三组状态。当前规范数据集已进一步简化为 Focus 与 Classical 两组，见 `docs/two-group-dataset-migration.md`。

迁移日期：2026-08-02。

## 结论

规范数据集已从“Brain.fm Focus 200 + Pop 300 + Classical 300”切换为“Jamendo Open Focus 300 + Pop 300 + Classical 300”。这里的 Open Focus 指公开来源、逐曲 CC 许可和可审计 provenance；由于部分许可含 NC/ND，它不表示无条件开放。`focus` 标签保持不变，但音频来源、许可、切分、预处理片段和状态模型均已替换。Brain.fm 不再进入规范清单或模型拟合。

## 新规范数据集

| 项目 | 数值 |
|---|---:|
| 曲目 | 900 |
| Focus / Pop / Classical | 300 / 300 / 300 |
| discovery / validation / holdout | 640 / 215 / 45 |
| 180 s + 300 s 片段 | 1,800 |
| 总源时长 | 76.68 小时 |
| 特征片段转换成功 | 1,800 / 1,800 |

开放 Focus 的 300 个源文件及 600 个 WAV 已逐文件复验 SHA-256。规范元数据校验结果为 900 tracks、900 licenses、0 errors、0 warnings。

## 受限归档

旧数据位于 `restricted_archive/brainfm_legacy_2026-08-02/`：

- Brain.fm 原音频与私有映射；
- 400 个可逆预处理 WAV；
- 旧 Focus 声学、音高、节奏、调制、结构特征与 sidecar；
- 旧状态模型、pitch_v2 码本和规范元数据快照。

归档共 3,441 个文件、19.649 GiB。逐文件清单为归档内 `file_inventory.csv`，清单 SHA-256 为 `7b667006c21272be621f207d4116ec3097ebd41baaedcfab6143115cefc7d3ab`。归档不可公开或再分发。

## 新状态模型

模型仅使用 discovery/180s 拟合：Focus 195、Pop 224、Classical 221。三组在 acoustic 与 rhythm 各平衡抽样 50,000 窗口；新模型 SHA-256 为 `37817f41be124aed8a726ed494072bf1e37be53948d6fd7bcc2aa65117b30464`。随后转换全部 1,800 个片段，0 失败。

## 科学证据边界

迁移没有自动把旧实验结论变成 Open Focus 结论。此前三组和四组 Path Homology、TDA、分类与生成目标结果包含 Brain.fm，因此全部标记为历史结果。它们只能用于追溯方法，不能用于声称当前开放 Focus 与 Pop/Classical 的差异。要获得新结论，必须使用当前清单和新状态模型重新运行分析及统计检验。

## 可复现入口

迁移工具为 `scripts/migrate_to_open_focus_dataset.py`，默认 dry-run，只有 `--execute` 才移动文件。规范审计见 `metadata/open_focus_migration_audit.json`，数据统计见 `metadata/dataset_summary.json`，结果有效性见 `metadata/RESULTS_STATUS.md`。
