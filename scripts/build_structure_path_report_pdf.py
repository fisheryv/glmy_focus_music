# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "path_homology_structure_analysis.pdf"
FIGURES = ROOT / "runs" / "structure_path_homology"


def register_fonts() -> None:
    font_path = Path(r"C:\Windows\Fonts\simhei.ttf")
    if not font_path.is_file():
        raise FileNotFoundError(f"Chinese font not found: {font_path}")
    pdfmetrics.registerFont(TTFont("SimHei", str(font_path)))
    pdfmetrics.registerFontFamily(
        "SimHei", normal="SimHei", bold="SimHei", italic="SimHei", boldItalic="SimHei"
    )


def page_footer(canvas, document) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setFont("SimHei", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(18 * mm, 10 * mm, "Path Homology 结构视角完整分析报告")
    canvas.drawRightString(192 * mm, 10 * mm, f"第 {document.page} 页")
    canvas.restoreState()


def scaled_image(path: Path, *, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def metric_table(data: list[list[str]], widths: list[float], *, font_size: float = 8) -> LongTable:
    table = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "SimHei"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C2CC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
            ]
        )
    )
    return table


def build_pdf() -> Path:
    register_fonts()
    topology_summary = json.loads(
        (ROOT / "metadata" / "topology_summary.json").read_text(encoding="utf-8")
    )
    feature_summary = json.loads(
        (ROOT / "metadata" / "feature_summary.json").read_text(encoding="utf-8")
    )
    statistics_summary = json.loads(
        (ROOT / "metadata" / "topology_statistics_summary.json").read_text(encoding="utf-8")
    )
    tests = pd.read_csv(ROOT / "metadata" / "topology_statistical_tests.csv")
    permanova = pd.read_csv(ROOT / "metadata" / "topology_permanova.csv")
    classification = pd.read_csv(ROOT / "metadata" / "classification_results.csv")
    topology = pd.read_csv(ROOT / "metadata" / "topology_segments.csv")
    structure = topology[
        (topology["view"] == "structure")
        & (topology["split"] == "validation")
        & (topology["scale_seconds"] == 180.0)
    ]
    nonzero = structure.groupby("group")["h1_betti_max"].apply(
        lambda values: int((values > 0).sum())
    )
    group_counts = structure.groupby("group").size()
    structure_tests = tests[
        (tests["analysis_set"] == "primary_validation_180") & (tests["view"] == "structure")
    ].sort_values(["p_fdr_bh", "epsilon_squared"])
    significant = structure_tests[structure_tests["p_fdr_bh"] <= 0.10]
    primary_permanova = permanova[permanova["analysis_set"] == "primary_validation_180"].iloc[0]
    primary_classification = classification[
        classification["analysis_set"] == "primary_validation_180"
    ].sort_values("macro_f1", ascending=False)

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ChineseTitle",
        parent=styles["Title"],
        fontName="SimHei",
        fontSize=24,
        leading=34,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#153E5C"),
        spaceAfter=14,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="SimHei",
        fontSize=11,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4B5563"),
    )
    heading1 = ParagraphStyle(
        "Heading1CN",
        parent=styles["Heading1"],
        fontName="SimHei",
        fontSize=16,
        leading=22,
        textColor=colors.HexColor("#153E5C"),
        spaceBefore=10,
        spaceAfter=8,
    )
    heading2 = ParagraphStyle(
        "Heading2CN",
        parent=styles["Heading2"],
        fontName="SimHei",
        fontSize=12,
        leading=18,
        textColor=colors.HexColor("#28536B"),
        spaceBefore=8,
        spaceAfter=5,
    )
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName="SimHei",
        fontSize=9.5,
        leading=16,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    formula = ParagraphStyle(
        "Formula",
        parent=body,
        fontSize=9,
        leading=15,
        leftIndent=8 * mm,
        rightIndent=8 * mm,
        backColor=colors.HexColor("#F1F5F9"),
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=8,
    )
    caption = ParagraphStyle(
        "Caption",
        parent=body,
        fontSize=8.5,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4B5563"),
        spaceBefore=4,
        spaceAfter=8,
    )
    bullet = ParagraphStyle(
        "BulletCN",
        parent=body,
        leftIndent=6 * mm,
        firstLineIndent=-3 * mm,
        bulletIndent=2 * mm,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Path Homology 结构视角完整分析报告",
        author="Focus Music GLMY research pipeline",
    )
    story: list[object] = []
    story.extend(
        [
            Spacer(1, 26 * mm),
            Paragraph("Path Homology 结构视角", title),
            Paragraph("自相似矩阵、宏观状态与持久有向拓扑的完整重分析", title),
            Spacer(1, 8 * mm),
            Paragraph("数据规模：1,600 个片段 / 800 首曲目 / 4 个状态视角", subtitle),
            Paragraph("生成日期：2026-08-01", subtitle),
            Spacer(1, 18 * mm),
        ]
    )
    cover_data = [
        ["核心产物", "结果"],
        ["宏观结构块", f"{feature_summary['quality']['structure_blocks']:,}"],
        ["片段-视图", f"{topology_summary['segment_views']:,}，零失败"],
        ["非零 H1 片段-视图", f"{topology_summary['h1_nonzero_segment_views']:,}"],
        ["主验证 PERMANOVA", f"pseudo-F={primary_permanova['pseudo_f']:.3f}, p={primary_permanova['p_value']:.3g}"],
        ["主检验 FDR 发现", f"{statistics_summary['primary_fdr_discoveries']}/{statistics_summary['primary_omnibus_tests']}"],
    ]
    cover_table = Table(cover_data, colWidths=[55 * mm, 85 * mm], hAlign="CENTER")
    cover_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "SimHei"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#153E5C")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B8C2CC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend(
        [
            cover_table,
            Spacer(1, 20 * mm),
            Paragraph(
                "解释边界：本报告描述三组音乐的声学结构与有向状态转移，不构成注意力提升、治疗或因果效果证据。",
                subtitle,
            ),
            PageBreak(),
            Paragraph("1. 分析目标与结构视角", heading1),
            Paragraph(
                "原有 pitch、rhythm、modulation 视角刻画局部音高、节奏与调制状态。新增 structure 视角把时间分辨率提升到宏观段落：先从短时声学向量构建自相似矩阵（SSM），以对角棋盘核检测边界，再将各段汇聚为高阶声学块并映射到冻结的共享原型。最终状态路径表示段落形态及其有向转换，而不是逐帧音色。",
                body,
            ),
            Paragraph(
                "完整链路：短时声学向量 -> SSM -> novelty -> 宏观边界 -> 块向量 -> 共享结构状态 -> 有向转移图 -> 持久 Path Homology。",
                formula,
            ),
            Paragraph("2. 数学原理", heading1),
            Paragraph("2.1 自相似矩阵", heading2),
            Paragraph(
                "对第 i 帧声学向量 x_i，以中位数与 MAD 做稳健标准化；加入偏置坐标并单位化，避免稳健中心处的向量退化为零。",
                body,
            ),
            Paragraph(
                "z_i = (x_i - med(x)) / (1.4826 MAD(x) + epsilon);  u_i = [z_i, 1] / ||[z_i, 1]||_2;  S_ij = (1 + u_i^T u_j) / 2.",
                formula,
            ),
            Paragraph(
                "S_ij 位于 [0,1]。对角附近高值表示局部连续性，远离对角线的重复亮块表示非相邻段落复现。",
                body,
            ),
            Paragraph("2.2 novelty 与边界", heading2),
            Paragraph(
                "令 L_t=[t-h,t)，R_t=[t,t+h)。棋盘核比较左右两侧内部相似度与跨侧相似度：",
                body,
            ),
            Paragraph(
                "nu(t) = [sum_LL S + sum_RR S - sum_LR S - sum_RL S] / (2 h^2).",
                formula,
            ),
            Paragraph(
                "实现使用 8 s 半窗、[0.25,0.5,0.25] 平滑、median + 1.5 MAD 峰值阈值，并约束段长 8-45 s；过长均质区间在局部最大 novelty 处分割。",
                body,
            ),
            Paragraph("2.3 宏观状态", heading2),
            Paragraph(
                "每个边界区间内的有效声学帧取均值得到块向量 q_k。仅用 discovery/180s 三组平衡样本拟合声学标准化、32 维 PCA 和 16 个 MiniBatch K-means 中心。validation 与 holdout 不参与状态定义。",
                body,
            ),
            Paragraph(
                "q_k = mean{x_i : i in block k};  s_k = argmin_m ||P D^(-1)(q_k - mu) - c_m||_2^2.",
                formula,
            ),
            Paragraph("2.4 有向图与 Path Homology", heading2),
            Paragraph(
                "转移计数 C_uv 归一化为出向概率 p_uv。过滤图 G_tau 保留 p_uv >= tau 的非自环边，每个源状态最多 top-6 条出边。阈值下降时只增边，形成嵌套有向图。",
                body,
            ),
            Paragraph(
                "partial e_(v0...vp) = sum_i (-1)^i e_(v0...vhat_i...vp);  Omega_p = {a in A_p : partial a in A_(p-1)};  H_p^path = ker(partial_p) / im(partial_(p+1)).",
                formula,
            ),
            Paragraph(
                "过滤图包含映射诱导持久模。报告 H0/H1 秩不变量、持久区间、barcode 和持久图。绘图使用递增过滤坐标 a=1-tau。",
                body,
            ),
            PageBreak(),
            Paragraph("3. SSM、边界与高阶状态示例", heading1),
            scaled_image(FIGURES / "structure_ssm_boundaries.png", max_width=174 * mm, max_height=205 * mm),
            Paragraph(
                "图 1. pop_jamendo_1045184__180s 的 SSM、novelty、检测边界与结构状态。状态路径为 6 -> 3 -> 10 -> 6 -> 3 -> 10 -> 6 -> 10 -> 10。",
                caption,
            ),
            PageBreak(),
            Paragraph("4. 有向状态图与持久过程", heading1),
            scaled_image(FIGURES / "structure_directed_state_graph.png", max_width=125 * mm, max_height=112 * mm),
            Paragraph(
                "图 2. 完整宏观状态转移图。边宽与标签表示出向转移概率。",
                caption,
            ),
            scaled_image(FIGURES / "structure_filtration_process.png", max_width=174 * mm, max_height=72 * mm),
            Paragraph(
                "图 3. tau=0.95 时 H1=0；tau=0.60 加入 6->3 后形成 6->3->10->6 的有向一维类；tau=0.30 加入 6->10 后新增允许 2-路径边界，H1 类死亡。",
                caption,
            ),
            Paragraph(
                "该 H1 类在阈值坐标中出生于 tau=0.60、死亡于 tau=0.30、寿命 0.30；在 a=1-tau 中对应 [0.40,0.70)。H0 从 tau=0.95 存活到观测终点，属于右删失区间。",
                body,
            ),
            PageBreak(),
            Paragraph("5. 持久图与 Barcode", heading1),
            scaled_image(FIGURES / "structure_persistence_diagram.png", max_width=120 * mm, max_height=110 * mm),
            Paragraph("图 4. 持久图。三角形为 H1；圆点为 H0，censored 表示在过滤终点仍存活。", caption),
            scaled_image(FIGURES / "structure_barcode.png", max_width=165 * mm, max_height=74 * mm),
            Paragraph("图 5. 同一示例的 persistent path barcode。叉号表示有限死亡，空心端点表示右删失。", caption),
            PageBreak(),
            Paragraph("6. 全量重分析结果", heading1),
        ]
    )
    results = [
        ["项目", "结果"],
        ["片段 / 曲目", "1,600 / 800"],
        ["视角", "pitch, rhythm, modulation, structure"],
        ["宏观结构块 / 边界", f"{feature_summary['quality']['structure_blocks']:,} / {feature_summary['quality']['structure_boundaries']:,}"],
        ["共享结构原型", "16，全部在数据中被使用"],
        ["片段-视图", f"{topology_summary['segment_views']:,}，零失败"],
        ["非零 H1 片段-视图", f"{topology_summary['h1_nonzero_segment_views']:,}"],
        ["结构 H1 非零 (validation/180s)", f"Classical {nonzero['classical']}/{group_counts['classical']}; Focus {nonzero['focus']}/{group_counts['focus']}; Pop {nonzero['pop']}/{group_counts['pop']}"],
        ["四视角 PERMANOVA", f"pseudo-F={primary_permanova['pseudo_f']:.3f}, p={primary_permanova['p_value']:.3g}, n={int(primary_permanova['n_tracks'])}"],
        ["FDR 发现", f"{statistics_summary['primary_fdr_discoveries']}/{statistics_summary['primary_omnibus_tests']} (q<=0.10)"],
    ]
    story.append(metric_table(results, [65 * mm, 105 * mm], font_size=8.5))
    story.extend(
        [
            Spacer(1, 6 * mm),
            Paragraph(
                "加入结构视角后，四视角 PERMANOVA 仍显著，但 pseudo-F 从旧三视角的 3.143 变为 2.365。维数增加改变协方差白化和伪 F 的尺度，不能把数值下降直接解释为模型变差。仅拓扑验证集 Macro-F1 从 0.776 上升到 0.801，显示结构信息提供了小幅增量判别力。",
                body,
            ),
            scaled_image(FIGURES / "structure_group_summary.png", max_width=174 * mm, max_height=62 * mm),
            Paragraph(
                "图 6. validation/180s 的结构视角组间摘要。Focus 的宏观自转移更高、状态数中位数更低；H1 明显零膨胀。",
                caption,
            ),
            Paragraph("6.1 结构视角通过 FDR 的指标", heading2),
        ]
    )
    structure_table = [
        ["指标", "Classical", "Focus", "Pop", "epsilon2", "FDR q"]
    ] + [
        [
            str(row.metric),
            f"{row.classical_median:.3f}",
            f"{row.focus_median:.3f}",
            f"{row.pop_median:.3f}",
            f"{row.epsilon_squared:.3f}",
            f"{row.p_fdr_bh:.3g}",
        ]
        for row in significant.itertuples(index=False)
    ]
    story.append(
        metric_table(
            structure_table,
            [48 * mm, 25 * mm, 23 * mm, 23 * mm, 23 * mm, 25 * mm],
            font_size=7.3,
        )
    )
    story.extend(
        [
            Spacer(1, 5 * mm),
            Paragraph(
                "H1 指标虽有若干秩检验通过 FDR，但三组中位数均为 0，属于明显零膨胀。稳妥解释是：结构视角主要增加了宏观自转移、状态数、边密度和方向互惠性信息；个别曲目出现可解释的 H1 生命周期。",
                body,
            ),
            Paragraph("6.2 验证集分类基线", heading2),
        ]
    )
    classification_table = [["特征集", "Macro-F1", "95% CI", "平衡准确率", "Macro-AUROC"]]
    for row in primary_classification.itertuples(index=False):
        classification_table.append(
            [
                str(row.feature_set),
                f"{row.macro_f1:.3f}",
                f"{row.macro_f1_ci_low:.3f}-{row.macro_f1_ci_high:.3f}",
                f"{row.balanced_accuracy:.3f}",
                f"{row.macro_auroc_ovr:.3f}",
            ]
        )
    story.append(
        metric_table(
            classification_table,
            [42 * mm, 27 * mm, 39 * mm, 34 * mm, 33 * mm],
            font_size=8,
        )
    )
    story.extend(
        [
            PageBreak(),
            Paragraph("7. 复现路径与数据产物", heading1),
            Paragraph("1) python -m features.batch backfill-structure --root . --workers 6", formula),
            Paragraph("2) python -m features.batch fit-states --root . --overwrite", formula),
            Paragraph("3) python -m features.batch transform-states --root . --workers 6 --overwrite", formula),
            Paragraph("4) python -m topology.batch model --root . --workers 6", formula),
            Paragraph("5) python -m topology.batch statistics --root .", formula),
            Paragraph("6) python scripts/render_structure_path_report.py", formula),
            Paragraph(
                "数值结果：metadata/topology_segments.csv、metadata/topology_filtration.csv、metadata/topology_filtration_sensitivity.csv、metadata/topology_statistical_tests.csv。",
                body,
            ),
            Paragraph(
                "结构特征：features/structure/；有向图：graphs/structure/；持久结果：homology/persistence/structure/ 与 homology/persistence_sensitivity/structure/。",
                body,
            ),
            Paragraph("8. 局限性与解释边界", heading1),
        ]
    )
    limitations = [
        "SSM 边界是算法性分段，不等同于人工曲式标注。",
        "16 个结构原型用于跨曲目可比性，但状态编号本身没有固定音乐语义。",
        "段落数较少导致 H1 零膨胀，单曲 barcode 比组中位数更适合解释结构环。",
        "当前最高报告维度为 H1；更高阶路径同调需要独立评估计算量与稳定性。",
        "统计比较是观察性描述；holdout 仅含 Focus 曲目，未用于三组检验。",
    ]
    for item in limitations:
        story.append(Paragraph(f"- {item}", bullet))
    story.extend(
        [
            Paragraph("参考文献", heading1),
            Paragraph("1. Foote, J. (2000). Automatic Audio Segmentation Using a Measure of Audio Novelty. ICME.", body),
            Paragraph("2. Müller, M. (2015). Fundamentals of Music Processing. Springer.", body),
            Paragraph("3. Grigor'yan, A., Lin, Y., Muranov, Y., and Yau, S.-T. (2012). Homologies of path complexes and digraphs.", body),
            Paragraph("4. Chowdhury, S., and Mémoli, F. (2018). Persistent path homology of directed networks. SODA.", body),
        ]
    )
    document.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return OUTPUT


if __name__ == "__main__":
    print(build_pdf())
