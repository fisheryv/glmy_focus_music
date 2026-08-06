# 音频预处理结果

本页记录依据 `deep-research-report-sol.md` 与 `数据集与预处理方案.md` 实际执行的音频预处理。正式参数来自 `configs/pipeline.toml`，逐片段审计清单为 `metadata/preprocessed_segments.csv`，机器可读汇总为 `metadata/preprocessing_summary.json`。

## 固定流程

1. 先验证 800 首源音频的许可、SHA-256、文件存在性和源曲目级 split；
2. 对全部 502 个 MP3 做连续 MPEG 帧时长复核，修正 14 个 Pop 虚高时长并补齐 3 个 Classical 时长；
3. 每首源曲分别生成 180 秒与 300 秒尺度，中段优先；不足目标时长保留整曲，不循环、不补零、不跨曲拼接；
4. 解码并重采样为 22,050 Hz、mono、float32；
5. 使用两遍 EBU R128 loudnorm 对齐到 -15 LUFS，再在最终采样率执行 -1 dBFS 软限幅与响度校准；
6. 原子写入 WAV，记录输出哈希、实际时长、响度、峰值、限幅比例、源 track ID 与继承的 split；
7. 重跑时仅复用配置哈希、源时长、输出属性和 SHA-256 全部匹配的文件。

静音窗不按“最大响度”择优，以免引入内容偏差。固定回退顺序为：中心窗、前一相邻窗、后一相邻窗、首窗、尾窗。全数据仅 1 个 180 秒 Pop 片段使用了静音回退。

## 最终结果

| 指标 | 结果 |
|---|---:|
| 源曲目 | 800 |
| 输出片段 | 1,600 |
| 180 秒尺度 | 800 |
| 300 秒尺度 | 800 |
| Focus / Pop / Classical | 400 / 600 / 600 |
| discovery / validation / holdout | 1,150 / 390 / 60 |
| 短曲整曲片段 | 577 |
| 输出音频总时长 | 90.41 小时 |
| 输出大小 | 26.73 GiB |
| -16 至 -14 LUFS 合格 | 1,600 / 1,600 |
| 最终 LUFS 范围 | -15.1995 至 -14.8005 |
| -1 dBFS 样本峰值合格 | 1,600 / 1,600 |
| 最大样本峰值 | 0.891249 |
| 失败 | 0 |

源曲目级 split 在切片前已经冻结。任何同一源曲目的 180 秒与 300 秒版本都继承相同 split，因此不存在片段级随机切分泄漏。

## 复现与审计

```powershell
$env:PYTHONPATH = "src"

# 只查看计划规模和预计空间
python -m data.preprocess --root . --dry-run

# 执行或断点续跑
python -m data.preprocess --root . --workers 6

# 再执行一次即进行属性与 SHA-256 复验；全部应为 verified_existing
python -m data.preprocess --root . --workers 6
```

分析 WAV 位于 `features/audio/{180s,300s}/<group>/<split>/`。该目录被 Git 忽略；尤其 Focus 派生 WAV 仍近似原音频，不属于可公开派生特征，不得发布。

## 开放 Focus 替代集（Jamendo，2026-08-01）

新加入的 300 首 `focus_open_music` 使用同一固定流程和同一配置哈希，但保持独立的元数据视图、
输出目录、逐片段清单和汇总，不覆盖上述 Brain.fm 基线：

| 指标 | 结果 |
|---|---:|
| 源曲目 | 300 |
| 输出片段 | 600 |
| 180 秒尺度 | 300 |
| 300 秒尺度 | 300 |
| discovery / validation / holdout | 390 / 120 / 90 |
| 短曲整曲片段 | 203 |
| 静音回退 | 0 |
| 输出音频总时长 | 35.69 小时 |
| 输出大小 | 10.55 GiB |
| -16 至 -14 LUFS 合格 | 600 / 600 |
| 最终 LUFS 范围 | -15.1997 至 -14.8035 |
| -1 dBFS 样本峰值合格 | 600 / 600 |
| 最大样本峰值 | 0.888737 |
| 失败 | 0 |
| 二次复验 | 600 / 600 `verified_existing` |

一首曲目 `focus_jamendo_0016398` 的 MPEG 帧扫描时长为 245 秒，但 FFmpeg 在 22,050 Hz
实际解码 5,279,940 个采样，即 239.453061 秒。该差异没有通过放宽短解码容差掩盖，而是记录在
`metadata/focus_open_duration_corrections.csv`，并在导出预处理索引时显式应用。

```powershell
$env:PYTHONPATH = "packages/pathhom_tda/src;src"

python -m data.focus_open prepare-preprocess metadata/focus_open_candidates.csv `
  --output-dir metadata/focus_open_preprocess `
  --data-root data_raw --expected-count 300 `
  --duration-corrections metadata/focus_open_duration_corrections.csv

python -m data.preprocess --root . `
  --metadata-dir metadata/focus_open_preprocess `
  --output-root features/audio_focus_open `
  --groups focus --workers 6 `
  --manifest metadata/focus_open_preprocessed_segments.csv `
  --summary metadata/focus_open_preprocessing_summary.json
```

开放 Focus WAV 位于 `features/audio_focus_open/{180s,300s}/focus/<split>/`。它们仍是近似原音频
的研究派生文件，不应作为可公开特征发布。历史选择的旧 WAV 分别保存在
`features/audio_focus_open_excluded_previous_selection_files`（364 个）和
`features/audio_focus_open_excluded_ambient_lofi_selection`（168 个）。本次标题/专辑优先重建后，
另有 66 个不再属于当前 300 首集合的 WAV 保存在
`features/audio_focus_open_excluded_pre_meditation_selection`，上述归档均不参与后续分析。
## 2026-08-02 规范数据更新

规范预处理清单现包含 900 首曲目和 1,800 个片段：Open Focus、Pop、Classical 各 600 个片段。Open Focus WAV 已从独立暂存目录提升为 `features/audio/{180s,300s}/focus/`；旧 Brain.fm WAV 已移入受限归档。当前机器可读结果以 `metadata/preprocessed_segments.csv` 和 `metadata/preprocessing_summary.json` 为准，本文后续旧数字仅保留为历史执行记录。

## 2026-08-02 两组简化更新

随后规范研究进一步简化为 Open Focus 与 Classical 两组。2026-08-02 的旧执行使用 Focus 195 与 Classical 221 个 discovery/180s 片段拟合，属于 `symmetric_holdout_v2` 之前的历史产物。当前规范切分已改为两组各 195/60/45；旧预处理清单和状态模型必须按新切分重跑，详见 `docs/symmetric-holdout-migration.md`。
