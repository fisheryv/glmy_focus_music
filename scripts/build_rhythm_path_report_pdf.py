# ruff: noqa: E501
from __future__ import annotations

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
OUTPUT = ROOT / "output" / "pdf" / "path_homology_rhythm_state_transition.pdf"
FIGURES = ROOT / "runs" / "rhythm_path_homology"
FONT = Path(r"C:\Windows\Fonts\simhei.ttf")


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text).replace("\n", "<br/>"), style)


def _image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def _table(data: list[list[str]], widths: list[float], font_size: float = 8.2) -> Table:
    cell = ParagraphStyle("CellCN", fontName="SimHei", fontSize=font_size, leading=font_size + 3, wordWrap="CJK")
    table = Table([[_p(str(value), cell) for value in row] for row in data], colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "SimHei"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#153E5C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#B8C2CC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6F8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _footer(canvas, document) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D5DCE2"))
    canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
    canvas.setFont("SimHei", 7.5)
    canvas.setFillColor(colors.HexColor("#59636E"))
    canvas.drawString(18 * mm, 8.5 * mm, "Focus Music GLMY - Path Homology rhythm")
    canvas.drawRightString(A4[0] - 18 * mm, 8.5 * mm, f"第 {document.page} 页")
    canvas.restoreState()


def build_pdf() -> Path:
    pdfmetrics.registerFont(TTFont("SimHei", str(FONT)))
    summary = json.loads((ROOT / "metadata" / "rhythm_analysis_summary.json").read_text(encoding="utf-8"))
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCN", parent=styles["Title"], fontName="SimHei", fontSize=22, leading=31, alignment=TA_CENTER, textColor=colors.HexColor("#153E5C"), spaceAfter=10)
    subtitle = ParagraphStyle("SubtitleCN", parent=styles["Normal"], fontName="SimHei", fontSize=10.2, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#59636E"))
    h1 = ParagraphStyle("H1CN", parent=styles["Heading1"], fontName="SimHei", fontSize=16, leading=22, textColor=colors.HexColor("#153E5C"), spaceBefore=4, spaceAfter=8)
    h2 = ParagraphStyle("H2CN", parent=styles["Heading2"], fontName="SimHei", fontSize=12.5, leading=18, textColor=colors.HexColor("#28536B"), spaceBefore=6, spaceAfter=5)
    body = ParagraphStyle("BodyCN", parent=styles["BodyText"], fontName="SimHei", fontSize=9.3, leading=15.2, alignment=TA_LEFT, wordWrap="CJK", spaceAfter=5)
    small = ParagraphStyle("SmallCN", parent=body, fontSize=7.8, leading=12, textColor=colors.HexColor("#59636E"), alignment=TA_CENTER)
    formula = ParagraphStyle("FormulaCN", parent=body, fontName="Courier", fontSize=8.0, leading=12.5, leftIndent=8, rightIndent=8, backColor=colors.HexColor("#EEF3F7"), borderPadding=6)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm, title="Path Homology Rhythm Analysis", author="Focus Music GLMY research pipeline")
    story: list[object] = []

    story.extend([
        Spacer(1, 28 * mm),
        _p("Path Homology 节奏视角", title),
        _p("八维局部节奏型、冻结码本与持续有向拓扑", title),
        Spacer(1, 12 * mm),
        _p("按照结构视角与音高视角框架完成的独立重跑", subtitle),
        _p("生成日期：2026-08-01", subtitle),
        Spacer(1, 14 * mm),
        _table([
            ["核心项目", "结果"],
            ["独立重跑", "1,600 个片段视图，0 失败"],
            ["状态码本", "Discovery/180 s；10 个全局节奏原型"],
            ["主比较集", "validation/180 s，n=195"],
            ["Omnibus FDR", f"{summary['primary_fdr_discoveries']}/20"],
            ["Focus-Pop FDR", f"{summary['focus_pop_fdr_discoveries']}/20"],
            ["H1 结论", "中位数全为零；阈值敏感"],
        ], [55 * mm, 115 * mm], 9.2),
        Spacer(1, 17 * mm),
        _p("证据等级：主结果来自 validation/180 s；300 s 只作尺度复核。该分析描述音乐来源类别，不构成注意力、治疗或因果证据。", small),
        PageBreak(),
    ])

    story.extend([
        _p("1. 节奏视角思想与八维特征", h1),
        _p("节奏视角回答局部起音活动、间隔规律和拍速共同形成哪些节奏型，以及这些节奏型沿时间如何有向转换。窗口长 1 s，步长 0.5 s。", body),
        _p("r_n = [onset mean, onset SD, onset max, onset rate, mean IOI, IOI SD, BPM, beat rate]^T", formula),
        _p("起音密度 rho_o=|O_n|/|I_n|；拍点密度 rho_b=|B_n|/|I_n|；局部 BPM=60/median(beat IOI)。缺失 IOI/BPM 使用 Discovery 中位数填补。", body),
        _p("2. Discovery-only 冻结码本", h1),
        _p("标准化后固定 V_rhythm=10，MiniBatch K-means 只在 discovery/180 s 拟合。Classical 和 Pop 各抽样上限 50,000 个窗口，Focus 使用全部 46,670 个窗口；这不是严格等量训练。", body),
        _p("r_tilde[n,j]=(r'[n,j]-mu_j)/sigma_j;  s_n=argmin_v ||r_tilde_n-mu_v||^2", formula),
        _image(FIGURES / "rhythm_codebook.png", 166 * mm, 128 * mm),
        _p("图 1. 十状态标准化质心及 Discovery 三组占用率。状态名后的两个特征仅是助记标签。", small),
        PageBreak(),
    ])

    story.extend([
        _p("3. 从冻结状态路径直接构建有向图", h1),
        _p("主分析不构造或读取 SSM。冻结状态路径直接通过相邻窗口状态计数进入有向图，因此移除 SSM 不改变任何拓扑数值。", body),
        _p("相邻状态计数按源状态归一化；每个源状态最多保留 6 条出边。自转移用于描述子，但不进入 Path Homology。未出现的全局状态不会作为人工孤立点加入单曲图。", body),
        _p("C_uv=#{n:s_n=u,s_(n+1)=v}; p_uv=C_uv/sum_w C_uw", formula),
        _image(FIGURES / "rhythm_directed_state_graph.png", 150 * mm, 128 * mm),
        _p("图 2. 示例六状态有向图。边宽表示转移概率，节点按冻结质心 PCA 角度排列。", small),
        _p("4. Path Homology", h1),
        _p("下降阈值图 G_tau 保留 p_uv>=tau 的非自环边。主阈值为 0.50-0.95，扩展敏感性降至 0.05。H0 表示连通合并过程；H1 表示未被允许二维路径边界填充的有向一维类。", body),
        _p("Omega_p={c in A_p : partial c in A_(p-1)}; H_p=ker(partial_p)/im(partial_(p+1)); a=1-tau", formula),
        PageBreak(),
    ])

    story.extend([
        _p("5. 持续同调示例", h1),
        _image(FIGURES / "rhythm_filtration_process.png", 169 * mm, 73 * mm),
        _p("图 3. tau=0.60 时 beta1=0；tau=0.50 时 H1 出生；tau=0.10 时加入反向边和捷径后死亡。", small),
        _p("主要区间为 [0.50,0.10)_tau，寿命 0.40；在 a=1-tau 中为 [0.50,0.90)。", body),
        _image(FIGURES / "rhythm_persistence_diagram.png", 124 * mm, 110 * mm),
        _p("图 4. 持久图。蓝色为 H0、红色为 H1、空心圈表示右删失。", small),
        PageBreak(),
    ])

    story.extend([
        _p("6. Barcode 与重跑审计", h1),
        _image(FIGURES / "rhythm_barcode.png", 168 * mm, 91 * mm),
        _p("图 5. 示例包含五条 H0 区间和一条有限 H1 区间。", small),
        Spacer(1, 7 * mm),
        _p("独立重跑完成 1,600/1,600 个片段视图，覆盖 800 首曲目，失败 0。主阈值 H1 非零片段共 36/1,600。", body),
        _p("全局十状态只保证跨曲目 ID 对齐，不强制每首曲目观察到全部状态。顶点数、H0 与状态覆盖率相关。", body),
        _p("模型 SHA256：047af2e4ddd32138f21873bfb9c41fbbdf9903a56e6efbdc0ed39d426a3ac0f2", small),
        PageBreak(),
    ])

    med = summary["validation_180_group_medians"]
    story.extend([
        _p("7. 三类音乐比较", h1),
        _table([
            ["指标", "Classical", "Focus", "Pop"],
            ["状态数", f"{med['vertex_count']['classical']:.0f}", f"{med['vertex_count']['focus']:.0f}", f"{med['vertex_count']['pop']:.0f}"],
            ["有向边数", f"{med['edge_count']['classical']:.1f}", f"{med['edge_count']['focus']:.1f}", f"{med['edge_count']['pop']:.1f}"],
            ["边密度", f"{med['edge_density']['classical']:.3f}", f"{med['edge_density']['focus']:.3f}", f"{med['edge_density']['pop']:.3f}"],
            ["自转移比例", f"{med['self_transition_ratio']['classical']:.3f}", f"{med['self_transition_ratio']['focus']:.3f}", f"{med['self_transition_ratio']['pop']:.3f}"],
            ["路径熵", f"{med['path_entropy']['classical']:.3f}", f"{med['path_entropy']['focus']:.3f}", f"{med['path_entropy']['pop']:.3f}"],
            ["互惠性", f"{med['reciprocity']['classical']:.3f}", f"{med['reciprocity']['focus']:.3f}", f"{med['reciprocity']['pop']:.3f}"],
            ["平均 beta0", f"{med['h0_betti_mean']['classical']:.3f}", f"{med['h0_betti_mean']['focus']:.3f}", f"{med['h0_betti_mean']['pop']:.3f}"],
        ], [52 * mm, 38 * mm, 38 * mm, 38 * mm]),
        Spacer(1, 5 * mm),
        _image(FIGURES / "rhythm_group_summary.png", 169 * mm, 61 * mm),
        _p("图 6. validation/180 s 的三组描述子分布。", small),
        _p("Omnibus：20 个 rhythm-only 指标中 14 个通过 FDR。最大效应来自 edge_count (epsilon^2=0.278)，其次为 H0 AUC、平均 beta0 和 H0 观测持久量。", body),
        _p("Focus-Pop：通过 pairwise FDR 的 5 项为边密度、互惠性、自转移比例、状态数和转移熵。边数不显著，说明 Focus 的差别是较少状态上的较高连接比例。", body),
        PageBreak(),
    ])

    h1_counts = summary["validation_180_h1_counts"]
    sensitivity_counts = summary["validation_180_sensitivity_h1_counts"]
    story.extend([
        _p("8. Betti 曲线与零膨胀 H1", h1),
        _image(FIGURES / "rhythm_betti_curves.png", 169 * mm, 71 * mm),
        _p("图 7. validation/180 s 的平均 Betti 曲线。H1 仅在低阈值附近出现小峰。", small),
        _table([
            ["组别", "主阈值非零 H1", "扩展阈值非零 H1"],
            ["Classical", f"{h1_counts['classical']['nonzero']}/{h1_counts['classical']['total']}", f"{sensitivity_counts['classical']['nonzero']}/{sensitivity_counts['classical']['total']}"],
            ["Focus", f"{h1_counts['focus']['nonzero']}/{h1_counts['focus']['total']}", f"{sensitivity_counts['focus']['nonzero']}/{sensitivity_counts['focus']['total']}"],
            ["Pop", f"{h1_counts['pop']['nonzero']}/{h1_counts['pop']['total']}", f"{sensitivity_counts['pop']['nonzero']}/{sensitivity_counts['pop']['total']}"],
        ], [58 * mm, 54 * mm, 54 * mm]),
        Spacer(1, 5 * mm),
        _p("六个 H1 指标均未通过主分析 FDR，三组中位数全为零。普通 rhythm 图不能支持 Focus 环更多的结论。", body),
        _p("Classical 的状态覆盖、边数和路径熵最高；Focus 状态较少但边密度与互惠性最高；Pop 自转移和有向复现度最高。", body),
        PageBreak(),
    ])

    story.extend([
        _p("9. 尺度复核与多表示关系", h1),
        _image(FIGURES / "rhythm_scale_sensitivity.png", 169 * mm, 67 * mm),
        _p("图 8. 180 s 与 300 s 的组中位数。Focus 相对 Pop 的边密度、互惠性和较低自转移方向一致。", small),
        _p("离散 rhythm 图、连续节奏点云和相位提升 Path Homology 不是同一个对象。离散图回答局部节奏型如何转换；连续点云 H0 回答轨迹几何是否紧凑；相位提升 H1 回答中尺度周期是否回归。", body),
        _p("综合既有验证结果，Focus 更接近：较紧凑的连续轨迹 + 较稠密且互惠的局部转换 + 更稳定的相位回归，而不是普通状态图 H1 更多。", body),
        _p("证据边界", h2),
        _p("连续节奏 H0 与相位 loop score 来自既有独立验证报告；它们用于解释关系，不是本次离散图重跑新增的发现。", body),
        PageBreak(),
    ])

    story.extend([
        _p("10. 局限性与复现", h1),
        _p("- 八维手工特征不能完整表达拍号、重音层级、切分和复节奏。\n- BPM 和 IOI 依赖事件检测，稀疏音乐可能不稳定。\n- Discovery 训练不是严格等量，需补做 46,670/组重采样。\n- K=10 尚未进行 K=8/12/16 的稳定性复核。\n- 顶点数、H0 与状态覆盖率相关。\n- top-k 和阈值对低发生率 H1 敏感。\n- rhythm-only FDR 不能与四视角联合 FDR 数字直接混用。\n- 来源类别差异不等于功能性或因果效应。", body),
        _p("复现命令", h2),
        _p("PYTHONPATH=src;packages/pyglmy/src python scripts/rerun_rhythm_path_homology.py\npython scripts/analyze_rhythm_results.py\npython scripts/render_rhythm_path_report.py", formula),
        _p("参考文献", h1),
        _p("1. Ellis, D. P. W. (2007). Beat tracking by dynamic programming.\n2. Muller, M. (2015). Fundamentals of Music Processing. Springer.\n3. Grigor'yan, A. et al. (2012). Homologies of path complexes and digraphs.\n4. Chowdhury, S. and Memoli, F. (2018). Persistent path homology of directed networks. SODA.", body),
    ])

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return OUTPUT


if __name__ == "__main__":
    print(build_pdf())
