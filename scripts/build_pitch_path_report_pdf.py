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
OUTPUT = ROOT / "output" / "pdf" / "path_homology_pitch_analysis.pdf"
FIGURES = ROOT / "runs" / "pitch_path_homology"
FONT = Path(r"C:\Windows\Fonts\simhei.ttf")


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text).replace("\n", "<br/>"), style)


def _scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def _table(data: list[list[str]], widths: list[float], font_size: float = 8.2) -> Table:
    cells = [
        [_paragraph(str(value), ParagraphStyle("Cell", fontName="SimHei", fontSize=font_size, leading=font_size + 3)) for value in row]
        for row in data
    ]
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
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
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
    canvas.drawString(18 * mm, 8.5 * mm, "Focus Music GLMY · Path Homology 音高视角")
    canvas.drawRightString(A4[0] - 18 * mm, 8.5 * mm, f"第 {document.page} 页")
    canvas.restoreState()


def _load_summary() -> dict[str, object]:
    return json.loads((ROOT / "metadata" / "pitch_topology_summary.json").read_text(encoding="utf-8"))


def _primary_pitch_rows() -> list[dict[str, str]]:
    with (ROOT / "metadata" / "topology_statistical_tests.csv").open(encoding="utf-8", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["analysis_set"] == "primary_validation_180" and row["view"] == "pitch"
        ]


def build_pdf() -> Path:
    pdfmetrics.registerFont(TTFont("SimHei", str(FONT)))
    summary = _load_summary()
    rows = _primary_pitch_rows()
    significant = sorted(
        (row for row in rows if float(row["p_fdr_bh"]) <= 0.10),
        key=lambda row: float(row["p_fdr_bh"]),
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCN",
        parent=styles["Title"],
        fontName="SimHei",
        fontSize=23,
        leading=34,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#153E5C"),
        spaceAfter=12,
    )
    subtitle = ParagraphStyle(
        "SubtitleCN",
        parent=styles["Normal"],
        fontName="SimHei",
        fontSize=10.5,
        leading=17,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#59636E"),
    )
    h1 = ParagraphStyle(
        "H1CN",
        parent=styles["Heading1"],
        fontName="SimHei",
        fontSize=16,
        leading=22,
        textColor=colors.HexColor("#153E5C"),
        spaceBefore=8,
        spaceAfter=7,
    )
    h2 = ParagraphStyle(
        "H2CN",
        parent=styles["Heading2"],
        fontName="SimHei",
        fontSize=12,
        leading=18,
        textColor=colors.HexColor("#28536B"),
        spaceBefore=6,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName="SimHei",
        fontSize=9.2,
        leading=15.2,
        alignment=TA_LEFT,
        wordWrap="CJK",
        spaceAfter=5,
    )
    formula = ParagraphStyle(
        "FormulaCN",
        parent=body,
        fontSize=8.7,
        leading=14,
        leftIndent=7 * mm,
        rightIndent=7 * mm,
        backColor=colors.HexColor("#F1F5F9"),
        borderPadding=5,
        spaceBefore=3,
        spaceAfter=7,
    )
    caption = ParagraphStyle(
        "CaptionCN",
        parent=body,
        fontSize=8,
        leading=12.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#59636E"),
        spaceBefore=3,
        spaceAfter=7,
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
        title="Path Homology 音高视角完整分析报告",
        author="Focus Music GLMY research pipeline",
    )
    story: list[object] = []

    story.extend(
        [
            Spacer(1, 25 * mm),
            _paragraph("Path Homology 音高视角", title),
            _paragraph("节拍同步 Chroma、有向状态转换与持续路径同调", title),
            Spacer(1, 7 * mm),
            _paragraph("三类音乐完整重跑与比较分析", subtitle),
            _paragraph("生成日期：2026-08-01", subtitle),
            Spacer(1, 18 * mm),
            _table(
                [
                    ["核心项目", "结果"],
                    ["独立 pitch 重跑", f"{summary['segment_views']:,} 个片段视图，0 失败"],
                    ["曲目与尺度", "800 首曲目；180 s 与 300 s 各 800 个"],
                    ["主比较集", "validation/180 s，n=195"],
                    ["主检验 FDR 发现", f"{len(significant)}/20 个 pitch 指标"],
                    ["主阈值非零 H1", "Classical 0/79；Focus 0/40；Pop 1/76"],
                ],
                [62 * mm, 92 * mm],
                9,
            ),
            Spacer(1, 17 * mm),
            _paragraph(
                "解释边界：本报告刻画音乐的音高状态覆盖与有向转换，不构成注意力提升、治疗效果或因果关系证据。",
                subtitle,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _paragraph("1. 音高视角的思想", h1),
            _paragraph(
                "音高视角把谱能量按八度折叠为 12 个音级，再按估计节拍边界做平均。每个节拍区间映射到最强音级；若最强与次强分量之比低于 1.15，或能量接近零，则映射到不确定态 U。最终得到 13 类离散状态路径。",
                body,
            ),
            _paragraph("c_bar[b] = (1/|I_b|) sum_{t in I_b} c[t]", formula),
            _paragraph(
                "s_b = argmax_p c_bar[b,p]，条件是 c_(1)/max(c_(2), 1e-8) >= 1.15；否则 s_b = U。",
                formula,
            ),
            _paragraph(
                "Chroma 弱化绝对音区与配器差别，保留音级内容；节拍同步减少逐帧颤动和微小对齐误差。它不是旋律转录：旋律、和弦、持续音和伴奏会共同影响状态。",
                body,
            ),
            _paragraph("2. 音高自相似矩阵", h1),
            _paragraph(
                "对每个节拍 chroma 做 L2 归一化，再计算余弦相似度。该 SSM 用于描述音级配置的重复，不参与宏观边界检测。",
                body,
            ),
            _paragraph("c_hat[i] = c_bar[i] / (||c_bar[i]||_2 + epsilon);  S_ij = c_hat[i]^T c_hat[j] in [0,1]", formula),
            _scaled_image(FIGURES / "pitch_chromagram_ssm.png", 167 * mm, 145 * mm),
            _paragraph(
                "图 1. 代表 Focus 片段的节拍同步 chromagram、13 类状态路径与音高 SSM。红点为 U。",
                caption,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _paragraph("3. 有向状态图", h1),
            _paragraph(
                "相邻状态的转移计数 C_uv 按源状态归一化为 p_uv。每个源状态只保留概率最高的 6 条出边；自转移用于描述统计，但在 Path Homology 图中排除。",
                body,
            ),
            _paragraph("C_uv = #{t: s_t=u, s_(t+1)=v};  p_uv = C_uv / sum_w C_uw", formula),
            _scaled_image(FIGURES / "pitch_directed_state_graph.png", 133 * mm, 132 * mm),
            _paragraph(
                "图 2. 代表片段的完整有向音高状态图。节点按半音圆排列，U 位于中心；边宽表示出向转移概率。",
                caption,
            ),
            _paragraph("4. Path Homology 原理", h1),
            _paragraph(
                "过滤图 G_tau 保留 p_uv >= tau 的非自环边。tau 从 0.95 降到 0.05 时只增加边，从而形成嵌套有向图。",
                body,
            ),
            _paragraph("G_0.95 subset G_0.90 subset ... subset G_0.05;  plotting coordinate a = 1 - tau", formula),
            _paragraph(
                "允许 p-路径的边界、Omega 路径链空间与路径同调分别为：",
                body,
            ),
            _paragraph(
                "partial e_(v0...vp) = sum_i (-1)^i e_(v0...vhat_i...vp)",
                formula,
            ),
            _paragraph(
                "Omega_p = {a in A_p : partial a in A_(p-1)}",
                formula,
            ),
            _paragraph(
                "H_p = ker(partial_p | Omega_p) / im(partial_(p+1) | Omega_(p+1))",
                formula,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _paragraph("5. 持续同调示例", h1),
            _scaled_image(FIGURES / "pitch_filtration_process.png", 174 * mm, 67 * mm),
            _paragraph(
                "图 3. tau=0.60 时 beta1=0；tau=0.50 时一个 H1 类出生；tau=0.15 时更多允许 2-路径边界将其填充，beta1 回到 0。",
                caption,
            ),
            _paragraph(
                "主要 H1 区间在阈值坐标中为 [0.50, 0.15)，寿命 0.35；在递增坐标 a=1-tau 中为 [0.50, 0.85)。另有一个寿命 0.05 的短区间。严格 H1 判定来自边界矩阵，而不是简单环计数。",
                body,
            ),
            _scaled_image(FIGURES / "pitch_persistence_diagram.png", 112 * mm, 100 * mm),
            _paragraph("图 4. 持久图：蓝色为 H0，红色为 H1，空心外圈为右删失。", caption),
            PageBreak(),
            _paragraph("6. Barcode", h1),
            _scaled_image(FIGURES / "pitch_barcode.png", 170 * mm, 96 * mm),
            _paragraph(
                "图 5. 同一样本的 barcode。蓝条为 H0，红条为 H1；空心端点表示在 tau=0.05 时仍存活。",
                caption,
            ),
            _paragraph("7. 重跑审计", h1),
            _paragraph(
                "独立重跑完成 1,600/1,600 个 pitch 片段视图，覆盖 Classical 600、Focus 400、Pop 600；180 s 与 300 s 各 800。与原四视角总表的 pitch 子集逐行比较，12 个核心图与同调指标的最大绝对差均为 0。",
                body,
            ),
            _paragraph(
                "主阈值范围内有 36/1,600 个片段出现非零 H1；扩展至 0.05 后有 519/1,600 个片段在至少一个阈值出现非零 H1。H1 对低概率边的纳入高度敏感。",
                body,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _paragraph("8. 三类音乐比较", h1),
            _table(
                [
                    ["指标", "Classical", "Focus", "Pop"],
                    ["状态数", "13", "9", "9"],
                    ["有向边数", "63", "39", "34"],
                    ["自转移比", "0.3059", "0.3790", "0.4192"],
                    ["路径熵", "1.7923", "1.4390", "1.3552"],
                    ["有向复现度", "0.0278", "0.0657", "0.0700"],
                    ["互惠性", "0.6207", "0.7603", "0.7434"],
                    ["平均 beta0", "11.6667", "7.1667", "6.4167"],
                    ["最大 beta1", "0", "0", "0"],
                ],
                [56 * mm, 32 * mm, 32 * mm, 32 * mm],
            ),
            Spacer(1, 5 * mm),
            _scaled_image(FIGURES / "pitch_group_summary.png", 174 * mm, 54 * mm),
            _paragraph(
                "图 6. validation/180 s 指标箱线图。Classical 的状态覆盖与转换网络更大；Focus 在相同状态数中位数下比 Pop 有更多边和更高路径熵。",
                caption,
            ),
            _paragraph("通过 FDR 的主要指标", h2),
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
                    for row in significant[:8]
                ],
                [44 * mm, 24 * mm, 22 * mm, 22 * mm, 22 * mm, 27 * mm],
                7.2,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _paragraph("9. Betti 曲线与 Focus-Pop 差异", h1),
            _scaled_image(FIGURES / "pitch_betti_curves_by_group.png", 174 * mm, 68 * mm),
            _paragraph(
                "图 7. 敏感性过滤中的均值与标准误。H0 主要反映状态规模与连通过程；H1 在较低阈值短暂升高，且组内异质性明显。",
                caption,
            ),
            _paragraph(
                "Focus-vs-Pop 有 7/20 个指标通过成对 FDR：H0 观察持久量、H0 AUC、有向边数、平均 beta0、最大 beta0、H0 区间数和路径熵。秩二列相关为 0.287 至 0.330，方向均为 Focus 高于 Pop。状态数、互惠性、自转移比、复现度及全部 H1 指标未通过成对 FDR。",
                body,
            ),
            _paragraph("10. 组别解读", h1),
            _paragraph(
                "古典组：状态数中位数达到 13，边数与路径熵最大，表示节拍级主导音级覆盖更广、转换更分散。较高 beta0 部分来自顶点更多，不能孤立解释为更不连贯。",
                body,
            ),
            _paragraph(
                "专注组：与流行组的状态数中位数相同，但边数、路径熵和 H0 持久量更高，说明在相近状态字母表下，弱到中等强度转移的连接过程更丰富。它不等于注意力效果。",
                body,
            ),
            _paragraph(
                "流行组：路径熵和边数较低，自转移比与有向复现度较高，符合较少转移模式承担更大概率质量的描述；组内离散度仍然很大。",
                body,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _paragraph("11. 局限与解释边界", h1),
        ]
    )
    limitations = [
        "Chroma 折叠八度并混合旋律、和声与伴奏，不能替代音符级转录。",
        "节拍估计误差会改变池化边界；Classical 的节拍步数也高于 Focus 与 Pop。",
        "U 同时受复音性、低能量和主峰不明确影响，不是和弦或休止标签。",
        "研究批处理把 U 当作图顶点；若改为缺失值，必须重新计算全部图与统计。",
        "H0 与状态字母表大小强相关，应结合状态数、边数和归一化指标解释。",
        "H1 在主阈值严重零膨胀，更适合单曲与敏感性解释，而非稳定组中位数。",
        "三组比较为观察性证据；holdout 仅含 Focus，未用于三组检验。",
    ]
    for item in limitations:
        story.append(_paragraph(f"- {item}", bullet))
    story.extend(
        [
            _paragraph("12. 复现与产物", h1),
            _paragraph("PYTHONPATH=src;packages/pyglmy/src  python scripts/rerun_pitch_path_homology.py", formula),
            _paragraph("python scripts/render_pitch_path_report.py", formula),
            _paragraph(
                "数值产物：metadata/pitch_topology_segments.csv、pitch_topology_filtration.csv、pitch_topology_filtration_sensitivity.csv 与 pitch_topology_summary.json。",
                body,
            ),
            _paragraph("参考文献", h1),
            _paragraph("1. Mueller, M. (2015). Fundamentals of Music Processing. Springer.", body),
            _paragraph("2. Ellis, D. P. W., and Poliner, G. E. (2007). Chroma features and dynamic programming beat tracking. ICASSP.", body),
            _paragraph("3. Grigor'yan, A. et al. (2012). Homologies of path complexes and digraphs.", body),
            _paragraph("4. Chowdhury, S., and Memoli, F. (2018). Persistent path homology of directed networks. SODA.", body),
        ]
    )

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return OUTPUT


if __name__ == "__main__":
    print(build_pdf())
