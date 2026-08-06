# Upstream metadata sources

该目录保存可重新获取的上游元数据仓库，因此默认不进入版本控制。

Pop 对照使用 MTG 官方仓库：

```powershell
git clone --depth 1 https://github.com/MTG/mtg-jamendo-dataset.git data_sources/mtg-jamendo-dataset
```

正式实验需在 `reproducibility/source_revisions.csv` 记录上游 commit。Musopen classical catalog 由 `focus-controls catalog-classical` 从 Internet Archive 上 Musopen 自筹录制项目的公开目录实时生成，不保存网页镜像。

Classical 扩展使用 Zenodo 官方 MusicNet 记录（DOI `10.5281/zenodo.5120004`）：

```powershell
python -m data.controls fetch-musicnet data_sources/musicnet/musicnet.tar.gz --workers 2
python -m data.controls extend-classical-musicnet metadata/control_classical.csv
python -m data.controls extract-musicnet data_sources/musicnet/musicnet.tar.gz
```

归档的官方 MD5 为 `844764911fa0d5b97c97da944a057590`。程序只从 11.1 GB 归档中解出最终清单选中的 205 个 WAV。
