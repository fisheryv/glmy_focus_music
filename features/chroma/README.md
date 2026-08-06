# Chroma features

每个片段保存 beat-synchronous 12 维 chroma、时间轴、有效性掩码，以及 12 个
pitch class 加 1 个不确定态的离散序列。状态 12 表示静音或最大 chroma 与次大
chroma 的比值低于 1.15。

输出按 `scale/group/split/segment_id.npz` 分区，由 `focus-features` 生成；音频本身
不会写入该目录。
