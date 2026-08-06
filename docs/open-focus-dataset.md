# Open Focus 数据集

Open Focus 已于 2026-08-02 从独立候选组提升为规范 `focus` 组。它不再与 Brain.fm 并列为第四组；当前规范比较固定为 Focus 与 Classical 两组。Pop 已完整迁移到 `dataset_archive/pop_music_legacy_2026-08-02/`，不再进入规范清单或状态模型拟合。

## 构建链

1. Jamendo API 以 instrumental、verylow/low 速度和 focus/meditation/study 等标签建立候选池。
2. 标题或专辑中含规范化 `medit` 词干的候选优先；显式 include/exclude 决策保留在候选清单。
3. 下载后扫描 MPEG 帧，记录真实时长、采样率、声道、平均码率、文件 SHA-256 和音频 payload SHA-256。
4. 以 artist/album 分组形成 discovery 195、validation 60、holdout 45，检查跨 split 泄漏。
5. 预处理为 22,050 Hz 单声道 float32 WAV，目标 -15 LUFS、峰值上限 -1 dBFS，不补零。

## 当前审计

- 候选记录 607；选中且验证 300。
- 154 位艺术家、263 张专辑；95 首标题或专辑字段匹配 `medit`。
- 300/300 文件 SHA-256 复验通过；无内容重复或 split 泄漏。
- 600/600 预处理 WAV 通过输出哈希与属性检查。

## 复验命令

```powershell
$env:PYTHONPATH = "packages/pathhom_tda/src;src"
python -m data.focus_open audit metadata/focus_open_candidates.csv `
  --target-count 300 --minimum-bitrate-kbps 0 `
  --allowed-moods focus,meditation,relaxing,study,deepwork --verify-hash

python -m focus_topology.cli validate-manifest --root . --check-files
```

下载、重选和预处理的新默认目标均为 `data_raw/focus_music/`。排除项仍位于 `data_raw/focus_open_music_excluded/`，不会进入任何规范索引。
