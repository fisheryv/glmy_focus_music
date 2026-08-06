# Classical 规范对照与 Pop 历史归档

当前规范研究只使用 Classical 作为 Focus 的对照；两组最终统计见 `metadata/dataset_summary.json`。下文 Pop 部分保留构建与授权追溯，但其数据已迁入 `dataset_archive/pop_music_legacy_2026-08-02/`，不再属于规范数据集。

本项目不把“网上可免费试听”等同于“可用于研究数据集”。候选、下载状态、逐曲许可、内容哈希和切分归属被记录在独立清单中；只有状态为 `verified` 的音频才能写入正式 `track_index.csv`。

## Pop 历史对照（已归档）

Pop 采用两级证据池。严格池来源为 [MTG-Jamendo Dataset](https://mtg.github.io/mtg-jamendo-dataset/)，筛选条件同时满足：

1. MTG 清洗标签包含 `genre---pop`；
2. 派生人工标注中三名标注者一致选择 `voice_instrumental---instrumental`；
3. MTG 的 `audio_licenses.txt` 中存在逐曲 Creative Commons 许可记录；
4. 每名艺术家最多保留 2 首，避免少数艺术家支配结果。

补充池使用 Jamendo 当前 tracks API，服务端查询同时限定 `tags=pop` 和 `vocalinstrumental=instrumental`，并只保留 `audiodownload_allowed=true` 且存在逐曲 CC 许可 URL 的结果。补充池与严格池使用不同的 `source_dataset` 和 `instrumental_evidence`，便于把 132 首严格池单独作为敏感性分析。

当前固定清单为 `metadata/control_pop.csv`，共 300 首：严格 MTG 池 132、API 补充池 168；discovery 224、validation 76，覆盖 231 名艺术家和 261 张专辑，实际每名艺术家最多 4 首。300/300 个音频均已下载并通过格式与 SHA-256 校验；经连续 MPEG 帧复核后总时长 18.88 小时、磁盘占用 1.49 GiB。discovery/validation 按艺术家整组划分，同一艺术家和专辑不能跨集合。原音频按保守策略不公开再分发。

Jamendo API 每个读取请求都需要用户自己的 `client_id`。官方文档中的公开测试 ID 仅用于快速测试，并且在 2026-07-16 实测已被暂停，不能用于批量构建数据集。

Jamendo 当前不可下载的 44 个源候选记录在 `metadata/control_pop_exclusions.csv`：19 个由 tracks API 明确返回 `audiodownload_allowed=false`，24 个已不再由 tracks API 返回，另 1 个由下载端点稳定返回 404。这些候选涉及 4 位艺术家，因此按艺术家整组排除。重建清单时会自动应用排除表，并保留已有已验证文件的状态和原有艺术家切分。

## Classical 对照

基础池不是 Musopen 的任意用户上传，而是 Musopen 通过 Kickstarter 出资录制并发布到公有领域的 2012 项目：[The Musopen Lossless DVD](https://archive.org/details/musopen-lossless-dvd)。Internet Archive 条目标注为 Public Domain Mark 1.0。

为了降低古典音乐内部跨度，只保留：

- Bach Goldberg Variations 与 Schubert piano sonatas；
- Beethoven、Borodin、Dvořák、Haydn、Mendelssohn、Mozart、Suk string quartets。

扩展池来自 [MusicNet Zenodo 记录](https://doi.org/10.5281/zenodo.5120004)。该记录整体采用 CC BY 4.0，包含 330 条带来源元数据的古典录音。扩展时排除 `source=Museopen` 以及与基础池相同的作品，按作曲家平衡后选取 205 条；最终 Classical 清单为 300 条，覆盖钢琴独奏、弦乐室内乐、混合室内乐与其他器乐独奏。

discovery/validation/holdout 按作曲家整组划分，当前为 195/60/45；同一作曲家和作品不能跨集合。这样比随机按乐章切分更严格，可以降低同一作品、演奏风格或录音来源造成的信息泄漏。切分细节与 holdout 组成限制见 `docs/symmetric-holdout-migration.md`。

当前固定清单为 `metadata/control_classical.csv`，共 300 首：Musopen 基础池 95、MusicNet 扩展池 205；discovery 195、validation 60、holdout 45，覆盖 13 位作曲家。300/300 个音频均已下载并通过格式与 SHA-256 校验；补齐 3 个备用 MP3 的连续帧时长后，总时长 28.31 小时、磁盘占用 14.64 GiB。编制子池包括 piano solo 145、string quartet 37、mixed chamber 64、string chamber 30、solo instrument 24。Musopen 基础池中 92 个来自 lossless DVD，3 个因虚拟 ZIP 成员持续返回 503，切换到同一 Musopen 项目的标准 DVD MP3 备用源。

预处理前对全部 502 个 MP3 做了连续 MPEG 帧扫描。除上述 3 个 Classical 备用文件外，另有 14 个 Pop 文件的有效帧时长与目录/Xing 时长相差超过 2 秒，已统一以有效帧时长为准；逐曲修正记录见 `metadata/duration_corrections.csv`。

尽管该特定条目明确标为 Public Domain Mark 1.0，Musopen FAQ 仍提醒公共领域判断具有司法辖区差异；公开发布前应保留来源页面、下载日期与哈希，并再次核对适用地区要求。

## 复现命令

```powershell
$env:PYTHONPATH = "src"

# 1. 固定官方 MTG 元数据版本
git clone --depth 1 https://github.com/MTG/mtg-jamendo-dataset.git data_sources/mtg-jamendo-dataset

# 2. 重建候选清单
python -m data.controls select-pop
python -m data.controls catalog-classical

# 3. 下载 Classical；可重复执行，自动跳过 verified 文件并续传 .part
python -m data.controls download metadata/control_classical.csv --workers 4

# 若 lossless ZIP 的个别成员持续 503，可切换到同一项目、同一 PDM 许可的标准 DVD MP3
python -m data.controls fallback-classical metadata/control_classical.csv
python -m data.controls download metadata/control_classical.csv --workers 2

# 审计若发现单曲内容哈希异常，可从清单中的原始 URL 强制重下该曲
python -m data.controls download metadata/control_classical.csv --workers 1 --force-track TRACK_ID

# 扩展 Classical 到 300；官方归档支持分段断点续传，并在解包前验证 MD5
python -m data.controls fetch-musicnet data_sources/musicnet/musicnet.tar.gz --workers 4
python -m data.controls extend-classical-musicnet metadata/control_classical.csv --target-total 300
python -m data.controls extract-musicnet data_sources/musicnet/musicnet.tar.gz

# 4. 下载 Pop；必须使用自己的 Jamendo read-only client ID
$env:JAMENDO_CLIENT_ID = "YOUR_CLIENT_ID"
python -m data.controls supplement-pop-jamendo metadata/control_pop.csv --target-total 300 --max-per-artist 6 --max-pages 6
python -m data.controls download metadata/control_pop.csv --workers 4

# 5. 只把已校验文件写入正式台账，然后做全量哈希审计
python -m data.controls finalize metadata/control_pop.csv metadata/control_classical.csv
python -m data.controls audit metadata/control_pop.csv metadata/control_classical.csv --verify-hash
python -m cli validate-manifest --root . --check-files
```

## 统计使用建议

扩大到每组 300 首主要提高覆盖度和稳健性，不应把同一艺术家或作曲家的多首曲目当作完全独立样本来夸大显著性。主分析应继续使用艺术家/作曲家整组切分，并报告按艺术家或作曲家聚类的 bootstrap / permutation 结果；同时单独报告 Pop 严格 MTG 132 首子集，确认结论不依赖 API 补充层。

## 占位音频

仓库原有的 `pop_00.wav`–`pop_03.wav` 和 `classical_00.wav`–`classical_03.wav` 均为 20 秒占位文件，没有进入候选清单、正式曲目索引或实验切分。不要把它们计入研究样本。
