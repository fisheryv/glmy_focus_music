# ruff: noqa: E501
from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "path_homology_pitch_v2_state_transition.pdf"
FIGURES = ROOT / "runs" / "pitch_v2_path_homology"
FONT = Path(r"C:\Windows\Fonts\simhei.ttf")


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text).replace("\n", "<br/>"), style)


def _image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def _table(data: list[list[str]], widths: list[float], font_size: float = 8.2) -> Table:
    cell_style = ParagraphStyle(
        "CellCN",
        fontName="SimHei",
        fontSize=font_size,
        leading=font_size + 3,
        wordWrap="CJK",
    )
    cells = [[_p(str(value), cell_style) for value in row] for row in data]
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "SimHei"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#153E5C")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#B8C2CC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6F8")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _footer(canvas, document) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D5DCE2"))
    canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
    canvas.setFont("SimHei", 7.5)
    canvas.setFillColor(colors.HexColor("#59636E"))
    canvas.drawString(18 * mm, 8.5 * mm, "Focus Music GLMY · Path Homology pitch_v2")
    canvas.drawRightString(A4[0] - 18 * mm, 8.5 * mm, f"第 {document.page} 页")
    canvas.restoreState()


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_pdf() -> Path:
    pdfmetrics.registerFont(TTFont("SimHei", str(FONT)))
    summary = json.loads((ROOT / "metadata" / "pitch_v2_summary.json").read_text(encoding="utf-8"))
    tests = _load_csv(ROOT / "metadata" / "pitch_v2_statistical_tests.csv")
    pairwise = _load_csv(ROOT / "metadata" / "pitch_v2_pairwise_tests.csv")
    primary = sorted(
        (row for row in tests if row["analysis_set"] == "primary_validation_180"),
        key=lambda row: float(row["p_fdr_bh"]),
    )
    focus_pop = sorted(
        (
            row
            for row in pairwise
            if row["analysis_set"] == "primary_validation_180"
            and row["group_a"] == "focus"
            and row["group_b"] == "pop"
            and float(row["p_fdr_bh"]) <= 0.10
        ),
        key=lambda row: float(row["p_fdr_bh"]),
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCN",
        parent=styles["Title"],
        fontName="SimHei",
        fontSize=22,
        leading=32,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#153E5C"),
        spaceAfter=10,
    )
    subtitle = ParagraphStyle(
        "SubtitleCN",
        parent=styles["Normal"],
        fontName="SimHei",
        fontSize=10.3,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#59636E"),
        wordWrap="CJK",
    )
    h1 = ParagraphStyle(
        "H1CN",
        parent=styles["Heading1"],
        fontName="SimHei",
        fontSize=15.5,
        leading=21,
        textColor=colors.HexColor("#153E5C"),
        spaceBefore=7,
        spaceAfter=6,
    )
    h2 = ParagraphStyle(
        "H2CN",
        parent=styles["Heading2"],
        fontName="SimHei",
        fontSize=11.5,
        leading=17,
        textColor=colors.HexColor("#28536B"),
        spaceBefore=5,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName="SimHei",
        fontSize=9.1,
        leading=14.8,
        alignment=TA_LEFT,
        wordWrap="CJK",
        spaceAfter=4.5,
    )
    formula = ParagraphStyle(
        "FormulaCN",
        parent=body,
        fontSize=8.4,
        leading=13.2,
        leftIndent=7 * mm,
        rightIndent=7 * mm,
        backColor=colors.HexColor("#F1F5F9"),
        borderPadding=5,
        spaceBefore=3,
        spaceAfter=6,
        wordWrap=None,
    )
    caption = ParagraphStyle(
        "CaptionCN",
        parent=body,
        fontSize=7.9,
        leading=12.2,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#59636E"),
        spaceBefore=3,
        spaceAfter=6,
    )
    bullet = ParagraphStyle(
        "BulletCN",
        parent=body,
        leftIndent=6 * mm,
        firstLineIndent=-3 * mm,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Path Homology pitch_v2 Tonnetz Harmonic Skeleton Analysis",
        author="Focus Music GLMY research pipeline",
    )
    story: list[object] = []

    story.extend(
        [
            Spacer(1, 25 * mm),
            _p("Path Homology pitch_v2", title),
            _p("Tonnetz 谐波骨架视角", title),
            _p("六维和声关系、冻结码本与持续有向拓扑", title),
            Spacer(1, 7 * mm),
            _p("新视角的完整重跑与三类音乐比较", subtitle),
            _p("生成日期：2026-08-01", subtitle),
            Spacer(1, 15 * mm),
            _table(
                [
                    ["核心项目", "结果"],
                    ["独立重跑", "1,600 个片段视图，0 失败"],
                    ["码本训练", "Discovery/180 s；三组各 32,709 节拍"],
                    ["状态空间", "16 个全局 Tonnetz 原型；无效节拍掩蔽"],
                    ["主比较集", "validation/180 s，n=195"],
                    ["Omnibus FDR", "19/20；H1 结果零膨胀"],
                    ["Focus-Pop FDR", "4/20；均为路径熵或 H0 指标"],
                ],
                [60 * mm, 96 * mm],
                8.8,
            ),
            Spacer(1, 15 * mm),
            _p(
                "证据等级：本方法在既有项目上事后提出，validation 结果属于探索性验证，不替代原 confirmatory pitch 分析，也不构成注意力或因果效果证据。",
                subtitle,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _p("1. 视角思想与 Tonnetz 原理", h1),
            _p(
                "pitch_v2 不再把每个节拍压缩为单一主导音级，而是把 12 维节拍 chroma 投影到六维五度、大小三度圆周坐标，再映射到 Discovery-only 的共享谐波原型。",
                body,
            ),
            _p("c_bar[b] = (1/|I_b|) sum_(t in I_b) c[t];  c_tilde[b] = c_bar[b] / (sum_p c_bar[b,p] + epsilon)", formula),
            _p("s = (7/6,7/6,3/2,3/2,2/3,2/3);  R = (1,1,1,1,0.5,0.5)", formula),
            _p("Phi_(r,p) = R_r cos(pi [s_r p - 0.5 I(r even)]);  z_b = Phi c_tilde[b] in R^6", formula),
            _p(
                "低置信节拍沿用既有 1.15 主峰比规则并设为缺失，不形成第 17 个状态，也不跨缺失位置计转移。validation/180 s 的有效比例中位数为 Classical 80.5%、Focus 75.4%、Pop 75.8%。",
                body,
            ),
            _p("2. Discovery-only 冻结码本", h1),
            _p(
                "三组严格平衡抽样，共 98,127 个训练节拍。固定 V_pitch=16，用 MiniBatch K-means 拟合中心；validation、300 s 与 holdout 只使用冻结中心。",
                body,
            ),
            _p("{mu_v} = argmin sum_i min_v ||z_i - mu_v||_2^2;  s_b = argmin_v ||z_b - mu_v||_2^2", formula),
            _image(FIGURES / "pitch_v2_codebook.png", 170 * mm, 113 * mm),
            _p(
                "图 1. 16 个原型的平均 chroma 轮廓及 K=8/12/16/24 诊断。V=16 是预先冻结的分辨率；V=12 的 silhouette 与稳定性更高，需优先做敏感性复核。",
                caption,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _p("3. 从冻结状态路径直接构建有向图", h1),
            _p("主分析不构造或读取 SSM。冻结状态路径直接通过相邻有效状态计数进入有向图，因此移除 SSM 不改变任何拓扑数值。", body),
            _p("相邻有效状态计数并按源状态归一化；每个源状态最多保留 6 条出边，自转移不进入同调图。", body),
            _p("C_uv = #{b: s_b=u, s_(b+1)=v};  p_uv = C_uv / sum_w C_uw", formula),
            _image(FIGURES / "pitch_v2_directed_state_graph.png", 128 * mm, 116 * mm),
            _p(
                "图 2. 示例的完整有向图。节点按冻结中心二维 PCA 的角度顺序排成圆环，仅用于避免标签重叠。S03、S09、S12 构成主要方向分量。",
                caption,
            ),
            _p("4. Path Homology", h1),
            _p("G_tau 保留 p_uv >= tau 的非自环边，tau 下降时只增加边。", body),
            _p("G_0.95 subset G_0.90 subset ... subset G_0.05;  plotting coordinate a = 1 - tau", formula),
            _p("partial e_(v0...vp) = sum_i (-1)^i e_(v0...vhat_i...vp)", formula),
            _p("Omega_p = {a in A_p : partial a in A_(p-1)};  H_p = ker(partial_p|Omega_p) / im(partial_(p+1)|Omega_(p+1))", formula),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _p("5. 持续同调示例", h1),
            _image(FIGURES / "pitch_v2_filtration_process.png", 174 * mm, 61 * mm),
            _p(
                "图 3. tau=0.80 时 beta1=0；tau=0.70 加入 S03->S12 后，S12->S09->S03->S12 形成 H1；tau=0.20 加入反向与捷径边后，该类被允许 2-路径边界填充。",
                caption,
            ),
            _p(
                "主要 H1 区间为 [0.70,0.20)_tau，寿命 0.50；在 a=1-tau 中为 [0.30,0.80)。严格判定来自边界矩阵而非简单环计数。",
                body,
            ),
            _image(FIGURES / "pitch_v2_persistence_diagram.png", 110 * mm, 95 * mm),
            _p("图 4. 持久图：蓝色为 H0，红色为 H1，空心外圈表示右删失。", caption),
            PageBreak(),
            _p("6. Barcode 与重跑审计", h1),
            _image(FIGURES / "pitch_v2_barcode.png", 170 * mm, 88 * mm),
            _p(
                "图 5. 示例 barcode。5 个 H0 分量在观测终点仍存活；有限 H1 区间寿命为 0.50。",
                caption,
            ),
            _p(
                "严格平衡码本版本完成 1,600/1,600 个片段。全量主阈值中 60/1,600 个片段出现非零 H1；扩展至 tau=0.05 后为 1,146/1,600，说明 H1 对低概率边高度敏感。",
                body,
            ),
            _p(
                "固定 16 个全局状态只保证 ID 对齐，不强制所有原型进入单曲图；未出现原型不会作为人工孤立点抬高 beta0。",
                body,
            ),
            PageBreak(),
        ]
    )

    medians = summary["validation_180_group_medians"]
    story.extend(
        [
            _p("7. 三类音乐比较", h1),
            _table(
                [
                    ["指标", "Classical", "Focus", "Pop"],
                    ["状态数", f"{medians['classical']['vertex_count']:.0f}", f"{medians['focus']['vertex_count']:.0f}", f"{medians['pop']['vertex_count']:.0f}"],
                    ["有向边数", f"{medians['classical']['edge_count']:.0f}", f"{medians['focus']['edge_count']:.1f}", f"{medians['pop']['edge_count']:.0f}"],
                    ["自转移比", f"{medians['classical']['self_transition_ratio']:.4f}", f"{medians['focus']['self_transition_ratio']:.4f}", f"{medians['pop']['self_transition_ratio']:.4f}"],
                    ["路径熵", f"{medians['classical']['path_entropy']:.4f}", f"{medians['focus']['path_entropy']:.4f}", f"{medians['pop']['path_entropy']:.4f}"],
                    ["有向复现度", f"{medians['classical']['directed_recurrence']:.4f}", f"{medians['focus']['directed_recurrence']:.4f}", f"{medians['pop']['directed_recurrence']:.4f}"],
                    ["平均 beta0", f"{medians['classical']['h0_betti_mean']:.4f}", f"{medians['focus']['h0_betti_mean']:.4f}", f"{medians['pop']['h0_betti_mean']:.4f}"],
                ],
                [54 * mm, 33 * mm, 33 * mm, 33 * mm],
            ),
            Spacer(1, 4 * mm),
            _image(FIGURES / "pitch_v2_group_summary.png", 174 * mm, 51 * mm),
            _p(
                "图 6. Classical 的状态覆盖、网络规模和路径熵明显更高；Pop 的转移概率更集中。",
                caption,
            ),
            _p("主要 omnibus 效应", h2),
            _table(
                [["指标", "Classical", "Focus", "Pop", "epsilon2", "FDR q"]]
                + [
                    [
                        row["metric"],
                        f"{float(row['classical_median']):.3f}",
                        f"{float(row['focus_median']):.3f}",
                        f"{float(row['pop_median']):.3f}",
                        f"{float(row['epsilon_squared']):.3f}",
                        f"{float(row['p_fdr_bh']):.2g}",
                    ]
                    for row in primary[:7]
                ],
                [45 * mm, 23 * mm, 22 * mm, 22 * mm, 23 * mm, 25 * mm],
                7.0,
            ),
            PageBreak(),
        ]
    )

    h1_counts = summary["validation_180_h1_counts"]
    story.extend(
        [
            _p("8. Betti 曲线与零膨胀 H1", h1),
            _image(FIGURES / "pitch_v2_betti_curves.png", 174 * mm, 69 * mm),
            _p(
                "图 7. H0 反映状态覆盖与连通过程；H1 在低阈值附近出现峰值，Classical 在 tau=0.20 左右的敏感性峰值最大。",
                caption,
            ),
            _table(
                [
                    ["组别", "主阈值非零 H1", "扩展阈值非零 H1"],
                    ["Classical", f"{h1_counts['classical']['primary_nonzero']}/{h1_counts['classical']['n']}", f"{h1_counts['classical']['sensitivity_nonzero']}/{h1_counts['classical']['n']}"],
                    ["Focus", f"{h1_counts['focus']['primary_nonzero']}/{h1_counts['focus']['n']}", f"{h1_counts['focus']['sensitivity_nonzero']}/{h1_counts['focus']['n']}"],
                    ["Pop", f"{h1_counts['pop']['primary_nonzero']}/{h1_counts['pop']['n']}", f"{h1_counts['pop']['sensitivity_nonzero']}/{h1_counts['pop']['n']}"],
                ],
                [52 * mm, 52 * mm, 52 * mm],
            ),
            Spacer(1, 4 * mm),
            _p(
                "20 个指标中 19 个通过 pitch_v2 内部 FDR。5 个 H1 指标虽有 q 约 0.014，但三组中位数均为零、epsilon2 约 0.035；它们是发生率差异，不是强而普遍的环结构差异。",
                body,
            ),
            _p("Focus 与 Pop 的直接比较", h2),
            _table(
                [["指标", "秩二列相关", "FDR q"]]
                + [
                    [
                        row["metric"],
                        f"{float(row['rank_biserial_a_minus_b']):.3f}",
                        f"{float(row['p_fdr_bh']):.3f}",
                    ]
                    for row in focus_pop
                ],
                [70 * mm, 43 * mm, 43 * mm],
            ),
            _p(
                "仅路径熵、最大 beta0、H0 区间数和 H0 观察持久量通过 q<=0.10；方向均为 Focus 较高，效应较小。",
                body,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _p("9. 与旧 pitch 的比较", h1),
            _image(FIGURES / "pitch_v2_vs_pitch.png", 174 * mm, 58 * mm),
            _p("图 8. validation/180 s 的组中位数。", caption),
            _table(
                [
                    ["指标", "Classical pitch→v2", "Focus pitch→v2", "Pop pitch→v2"],
                    ["状态数", "13 → 16", "9 → 10", "9 → 10"],
                    ["边数", "63 → 67", "39 → 36.5", "34 → 32"],
                    ["路径熵", "1.792 → 1.700", "1.439 → 1.285", "1.355 → 1.197"],
                    ["平均 beta0", "11.667 → 13.167", "7.167 → 7.583", "6.417 → 6.917"],
                ],
                [43 * mm, 39 * mm, 37 * mm, 37 * mm],
                7.8,
            ),
            _p("9.1 结构解释", h2),
            _p(
                "固定码本扩大了可用状态字母表，因此三组状态数和 H0 略升；掩蔽低置信节拍切断部分转换，使 Focus 与 Pop 的边数和路径熵下降。",
                body,
            ),
            _p(
                "Classical 仍覆盖几乎全部 16 个原型，说明共享码本只解决 ID 对齐，不自动消除组间状态覆盖差异。",
                body,
            ),
            _p(
                "Focus-Pop 的 FDR 发现由旧 pitch 的 7 项降为 4 项，且边数不再显著；两组差异主要仍在 H0 连通过程和转移多样性，而不是稳定 H1。",
                body,
            ),
            _p("9.2 三组解读", h2),
            _p(
                "古典组：谐波原型覆盖最广、网络最大、路径熵最高。较高 beta0 主要来自更多顶点在高阈值下尚未连接，不能解释为更不连贯。",
                body,
            ),
            _p(
                "专注组：与流行组状态数中位数相同，但路径熵和 H0 持久量略高，说明相近字母表下转换更分散；不等于注意力效果。",
                body,
            ),
            _p(
                "流行组：边数和路径熵最低、复现度最高，表示概率质量更集中；主阈值非零 H1 较多，但仍只有 7/76。",
                body,
            ),
            PageBreak(),
        ]
    )

    story.append(_p("10. 局限性与下一步", h1))
    limitations = [
        "Tonnetz 提供和声邻接语义，但原型最强音级不能直接命名为主、属或具体和弦。",
        "当前表示保留绝对音级，不具备转调不变性；按调性旋转 chroma 将回答另一个功能和声问题。",
        "普通 K-means 使用六维欧氏弦距离；K-medoids 与环面测地距离需要作为敏感性分析。",
        "V=16 不是 Discovery 诊断的唯一最优值；V=12 的 silhouette 与稳定性更高，应优先复核。",
        "可信度掩码沿用旧主导音级规则，可能丢弃可由 Tonnetz 表示的复音节拍。",
        "单曲实际顶点数仍可变，H0 与状态覆盖强相关；应结合覆盖率、有效节拍数和边数解释。",
        "当前链复形没有把 Delta_uv 作为边标签；因此只能说顶点具有 Tonnetz 几何语义。",
        "该视角是事后提出的探索性验证；holdout 仅含 Focus，未用于三组检验。",
    ]
    for item in limitations:
        story.append(_p(f"- {item}", bullet))
    story.extend(
        [
            _p("11. 复现与产物", h1),
            _p("PYTHONPATH=src;packages/pyglmy/src  python scripts/run_pitch_v2_analysis.py", formula),
            _p("python scripts/render_pitch_v2_report.py", formula),
            _p(
                "主要产物：features/models/pitch_v2_codebook.*、features/pitch_v2/、metadata/pitch_v2_topology_*.csv、pitch_v2_statistical_tests.csv、pitch_v2_pairwise_tests.csv 与 pitch_v2_summary.json。",
                body,
            ),
            _p("参考文献", h1),
            _p("1. Harte, C., Sandler, M., and Gasser, M. (2006). Detecting Harmonic Change in Musical Audio.", body),
            _p("2. Mueller, M. (2015). Fundamentals of Music Processing. Springer.", body),
            _p("3. Grigor'yan, A. et al. (2012). Homologies of path complexes and digraphs.", body),
            _p("4. Chowdhury, S., and Memoli, F. (2018). Persistent path homology of directed networks. SODA.", body),
        ]
    )

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return OUTPUT


if __name__ == "__main__":
    print(build_pdf())
