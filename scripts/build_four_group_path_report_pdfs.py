# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
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
FIGURES = ROOT / "runs" / "four_group_path_homology"
OUTPUT = ROOT / "output" / "pdf"
FONT = Path(r"C:\Windows\Fonts\simhei.ttf")
GROUPS = ("classical", "focus", "focus_open", "pop")
GROUP_LABELS = ("Classical", "Focus", "Focus Open", "Pop")
VIEW_LABELS = {
    "structure": "结构视角",
    "pitch_v2": "音高视角（Tonnetz pitch_v2）",
    "rhythm": "节奏视角",
}
INPUT_FIGURES = {
    "structure": "structure_ssm_boundaries.png",
    "pitch_v2": "pitch_v2_state_sequence.png",
    "rhythm": "rhythm_state_sequence.png",
}
OUTPUT_NAMES = {
    "structure": "path_homology_structure_four_group_analysis.pdf",
    "pitch_v2": "path_homology_pitch_v2_four_group_analysis.pdf",
    "rhythm": "path_homology_rhythm_four_group_analysis.pdf",
}


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text).replace("\n", "<br/>"), style)


def _image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def _table(data: list[list[str]], widths: list[float], font_size: float = 8.0) -> Table:
    style = ParagraphStyle(
        "CellCN",
        fontName="SimHei",
        fontSize=font_size,
        leading=font_size + 3,
        wordWrap="CJK",
    )
    cells = [[_p(str(value), style) for value in row] for row in data]
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "SimHei"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#153E5C")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C2CC")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F3F6F8")],
                ),
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
    canvas.drawString(18 * mm, 8.5 * mm, "Focus Music GLMY - 四组 Path Homology")
    canvas.drawRightString(A4[0] - 18 * mm, 8.5 * mm, f"第 {document.page} 页")
    canvas.restoreState()


def _styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=styles["Title"],
            fontName="SimHei",
            fontSize=21,
            leading=31,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#153E5C"),
            spaceAfter=9,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN",
            parent=styles["Normal"],
            fontName="SimHei",
            fontSize=10,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#59636E"),
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "H1CN",
            parent=styles["Heading1"],
            fontName="SimHei",
            fontSize=15.5,
            leading=21,
            textColor=colors.HexColor("#153E5C"),
            spaceBefore=6,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "H2CN",
            parent=styles["Heading2"],
            fontName="SimHei",
            fontSize=11.5,
            leading=17,
            textColor=colors.HexColor("#28536B"),
            spaceBefore=4,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=styles["BodyText"],
            fontName="SimHei",
            fontSize=9.1,
            leading=14.6,
            alignment=TA_LEFT,
            wordWrap="CJK",
            spaceAfter=5,
        ),
        "formula": ParagraphStyle(
            "FormulaCN",
            parent=styles["BodyText"],
            fontName="SimHei",
            fontSize=8.1,
            leading=13,
            leftIndent=6 * mm,
            rightIndent=6 * mm,
            backColor=colors.HexColor("#F1F5F9"),
            borderPadding=5,
            spaceBefore=3,
            spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "CaptionCN",
            parent=styles["BodyText"],
            fontName="SimHei",
            fontSize=7.8,
            leading=11.5,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#59636E"),
            spaceBefore=3,
            spaceAfter=6,
        ),
    }


def _method(view: str) -> tuple[list[str], list[str]]:
    common = [
        "C_ij = sum_t 1[z_t=i, z_(t+1)=j];  P_ij = C_ij / sum_k C_ik.",
        "G_tau = (V, {(i,j): P_ij >= tau}); each source keeps at most top_k=6 edges.",
        "H_p^path(G) = ker(partial_p | Omega_p) / im(partial_(p+1) | Omega_(p+1)); beta_p = dim H_p.",
        "Persistence diagrams use a = 1 - tau. Primary thresholds are 0.50-0.95; lower thresholds are sensitivity only.",
    ]
    if view == "structure":
        paragraphs = [
            "结构视角把时间分辨率提升到宏观段落。短时声学向量先形成余弦自相似矩阵，棋盘核新颖度定位边界；段内池化后映射到冻结结构原型，最后用相邻段落状态构建有向图。",
            "SSM 是结构边界构造的一部分。结构状态模型仅由四组 Discovery/180s 等量样本拟合，Validation 与 300s 不参与拟合。",
        ]
        formulas = [
            "S_ij = <x_i,x_j> / (||x_i||_2 ||x_j||_2).",
            "nu(t) = sum_(i=-L)^L sum_(j=-L)^L K_L(i,j) S_(t+i,t+j).",
            "s_m = pool{x_t : b_m <= t < b_(m+1)}.",
            *common,
        ]
    elif view == "pitch_v2":
        paragraphs = [
            "音高视角从 12 维 Chroma 出发，映射到六维 Tonnetz 的五度、小三度和大三度圆坐标，再用四组 Discovery/180s 严格等量样本拟合 16 状态 K-means 谐波骨架码本。",
            "主图直接由相邻冻结音高状态构建，SSM 不参与建图。码本热图和 V_pitch 敏感性用于审计状态语义与稳定性。",
        ]
        formulas = [
            "c_tilde(k) = c(k) / (sum_r c(r) + epsilon).",
            "T(c) = sum_k c_tilde(k) [cos(7pi k/6), sin(7pi k/6), cos(3pi k/2), sin(3pi k/2), cos(2pi k/3), sin(2pi k/3)].",
            "z_t = argmin_(v=1..16) ||T(c_t) - mu_v||_2^2.",
            *common,
        ]
    else:
        paragraphs = [
            "节奏视角把每个时间窗表示为局部 onset、速度/节拍、IOI 与 tempogram 形态的多维向量。缺失维按 Discovery 中位数填补，标准化后用四组等量样本拟合冻结节奏码本。",
            "主图直接由相邻冻结节奏状态构建，SSM 不参与建图。该表示描述局部时间组织，而不是宏观段落边界。",
        ]
        formulas = [
            "r_fill(td) = r_td if valid, otherwise median_d;  r_tilde(td) = (r_fill(td)-mu_d)/sigma_d.",
            "z_t = argmin_v ||r_tilde_t - mu_v^(rhythm)||_2^2.",
            *common,
        ]
    return paragraphs, formulas


def _fmt(value: float) -> str:
    if value != 0 and abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def build_pdf(
    view: str,
    summary: dict[str, object],
    topology: pd.DataFrame,
    tests: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> Path:
    styles = _styles()
    output = OUTPUT / OUTPUT_NAMES[view]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"Four-group Path Homology {view}",
        author="Focus Music GLMY research pipeline",
    )
    primary = topology[
        (topology["view"] == view)
        & (topology["split"] == "validation")
        & (topology["scale_seconds"] == 180.0)
    ]
    metrics = (
        "vertex_count",
        "edge_count",
        "path_entropy",
        "directed_recurrence",
        "h0_betti_mean",
        "h1_betti_max",
    )
    medians = primary.groupby("group")[list(metrics)].median()
    view_tests = tests[
        (tests["view"] == view) & (tests["analysis_set"] == "primary_validation_180")
    ].sort_values(["p_fdr_bh", "epsilon_squared"])
    discoveries = int((view_tests["p_fdr_bh"] <= 0.10).sum())
    sensitivity_tests = tests[
        (tests["view"] == view)
        & (tests["analysis_set"] == "sensitivity_validation_300")
        & (tests["p_fdr_bh"] <= 0.10)
    ]
    replicated_metrics = sorted(
        set(view_tests.loc[view_tests["p_fdr_bh"] <= 0.10, "metric"])
        & set(sensitivity_tests["metric"])
    )
    key_pairs = pairwise[
        (pairwise["view"] == view)
        & (pairwise["analysis_set"] == "primary_validation_180")
        & (
            ((pairwise["group_a"] == "focus") & (pairwise["group_b"] == "focus_open"))
            | ((pairwise["group_a"] == "focus_open") & (pairwise["group_b"] == "pop"))
        )
    ].sort_values(["p_fdr_bh", "metric"])
    h1_counts = [
        int((primary.loc[primary["group"] == group, "h1_betti_max"] > 0).sum()) for group in GROUPS
    ]
    h1_text = "; ".join(
        f"{label} {count}" for label, count in zip(GROUP_LABELS, h1_counts, strict=True)
    )
    paragraphs, formulas = _method(view)
    story: list[object] = [
        Spacer(1, 25 * mm),
        _p("四组音乐 Path Homology", styles["title"]),
        _p(VIEW_LABELS[view], styles["title"]),
        _p("Focus / Focus Open / Pop / Classical", styles["subtitle"]),
        _p("结构、状态转移与持久有向拓扑完整重跑", styles["subtitle"]),
        Spacer(1, 15 * mm),
        _table(
            [
                ["项目", "结果"],
                ["独立数据组", "4 组；Focus Open 不与 Focus 合并"],
                ["曲目 / 片段", f"{summary['tracks']} / {summary['segments']}"],
                ["本视角片段", f"{summary['view_counts'][view]}，零失败"],  # type: ignore[index]
                ["主推断", f"Validation/180s，n={summary['primary_validation_n_per_view']}"],
                ["Omnibus FDR", f"{discoveries}/20 指标，统一三视角家族"],
                ["H1 非零", h1_text],
            ],
            [58 * mm, 98 * mm],
            8.7,
        ),
        Spacer(1, 14 * mm),
        _p(
            "证据边界：状态模型仅用 Discovery/180s 拟合；主比较冻结为 Validation/180s；300s 为敏感性。Holdout 未进入四组检验。本结果为观察性声学结构证据，不支持注意力、认知收益或因果解释。",
            styles["subtitle"],
        ),
        PageBreak(),
        _p("1. 视角构建思想与原理", styles["h1"]),
    ]
    for paragraph in paragraphs:
        story.append(_p(paragraph, styles["body"]))
    story.append(_p("公式链", styles["h2"]))
    for formula in formulas:
        story.append(_p(formula, styles["formula"]))
    story.extend(
        [
            _p("2. 输入表示与状态序列", styles["h1"]),
            _image(
                FIGURES / INPUT_FIGURES[view],
                174 * mm,
                (135 if view == "structure" else 190) * mm,
            ),
            _p(
                "图 1. 结构视角展示 SSM-新颖度-边界-状态链；音高和节奏视角展示直接状态序列，明确不以 SSM 建图。",
                styles["caption"],
            ),
        ]
    )
    if view == "pitch_v2":
        story.extend(
            [
                PageBreak(),
                _p("附加：四组 Discovery 音高码本", styles["h1"]),
                _image(FIGURES / "pitch_v2_codebook.png", 174 * mm, 205 * mm),
                _p("图 2. 16 状态 Chroma 原型与 V_pitch 诊断。", styles["caption"]),
            ]
        )
    story.extend(
        [
            PageBreak(),
            _p("3. 有向状态图与持续同调过程", styles["h1"]),
            _image(FIGURES / f"{view}_directed_state_graph.png", 125 * mm, 112 * mm),
            _p("图 3. Focus Open validation/180s 解释性示例的加权有向状态图。", styles["caption"]),
            _image(FIGURES / f"{view}_filtration_process.png", 174 * mm, 72 * mm),
            _p("图 4. 阈值下降时边的加入与 beta0/beta1 变化。", styles["caption"]),
            PageBreak(),
            _p("4. 持久图与 Barcode", styles["h1"]),
            _image(FIGURES / f"{view}_persistence_diagram.png", 120 * mm, 105 * mm),
            _p("图 5. 持久图；空心点表示在观测终点仍存活。", styles["caption"]),
            _image(FIGURES / f"{view}_barcode.png", 165 * mm, 78 * mm),
            _p("图 6. H0/H1 barcode。", styles["caption"]),
            PageBreak(),
            _p("5. 四组 Validation/180s 数值结果", styles["h1"]),
        ]
    )
    median_table = [["指标", *GROUP_LABELS]]
    for metric in metrics:
        median_table.append(
            [metric, *[_fmt(float(medians.loc[group, metric])) for group in GROUPS]]
        )
    story.extend(
        [
            _table(median_table, [49 * mm, 27 * mm, 27 * mm, 27 * mm, 27 * mm], 7.6),
            Spacer(1, 4 * mm),
            _p(
                f"本视角 {discoveries}/20 个 omnibus 指标通过统一 FDR q<=0.10。H1 中位数为零时，显著性只可解释为零膨胀发生率或尾部差异，不能表述为普遍有向环。",
                styles["body"],
            ),
            _p(
                f"Validation/300s 敏感性中 {len(sensitivity_tests)}/20 个指标通过独立 FDR；与主分析共同通过的指标为："
                + (", ".join(replicated_metrics) if replicated_metrics else "无"),
                styles["body"],
            ),
            _p("按 FDR 排序的前 12 个 omnibus 指标", styles["h2"]),
        ]
    )
    test_table = [["指标", "epsilon^2", "FDR q"]]
    for _, row in view_tests.head(12).iterrows():
        test_table.append(
            [
                str(row["metric"]),
                _fmt(float(row["epsilon_squared"])),
                _fmt(float(row["p_fdr_bh"])),
            ]
        )
    story.extend(
        [
            _table(test_table, [90 * mm, 33 * mm, 33 * mm], 7.8),
            PageBreak(),
            _p("6. Focus Open 的关键对比", styles["h1"]),
        ]
    )
    pair_table = [["对比", "指标", "rank-biserial", "FDR q"]]
    for _, row in key_pairs.head(16).iterrows():
        pair_table.append(
            [
                f"{row['group_a']} vs {row['group_b']}",
                str(row["metric"]),
                _fmt(float(row["rank_biserial_a_minus_b"])),
                _fmt(float(row["p_fdr_bh"])),
            ]
        )
    story.extend(
        [
            _table(pair_table, [42 * mm, 62 * mm, 27 * mm, 25 * mm], 7.3),
            Spacer(1, 5 * mm),
            _p(
                "正 rank-biserial 表示表中前一组更高。Focus Open 只能作为独立开放数据分布解读；即使与 Focus 接近，也不证明两者等价。统一 FDR 后未通过的对比标为不支持。",
                styles["body"],
            ),
            PageBreak(),
            _p("7. 分布与 Betti 曲线", styles["h1"]),
            _image(FIGURES / f"{view}_group_summary.png", 174 * mm, 72 * mm),
            _p("图 7. 五个主要描述量的四组分布。", styles["caption"]),
            _image(FIGURES / f"{view}_betti_curves.png", 174 * mm, 75 * mm),
            _p("图 8. 扩展过滤下的平均 beta0/beta1 与标准误。", styles["caption"]),
            PageBreak(),
            _p("8. 结论、限制与复现", styles["h1"]),
            _p(
                "结论应按三层读取：图描述量反映状态组织；H0 反映有向图连通分支；H1 只在实际非零曲目中支持环类。结构、音高和节奏互补，任何单一视角都不能代表完整音乐结构。",
                styles["body"],
            ),
            _p(
                "限制包括数据源差异、Focus Open 的标签代理性质、固定 top-k 对短结构序列的影响、H1 零膨胀，以及多重比较。敏感性曲线不替代冻结主分析。",
                styles["body"],
            ),
            _p(
                "复现：python scripts/run_four_group_path_homology.py --workers 6；随后运行 render_four_group_path_reports.py 和 build_four_group_path_report_pdfs.py。模型、特征、图、同调和报告均使用 four_group 隔离命名空间。",
                styles["formula"],
            ),
        ]
    )
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output


def main() -> None:
    if not FONT.is_file():
        raise RuntimeError(f"Chinese font not found: {FONT}")
    pdfmetrics.registerFont(TTFont("SimHei", str(FONT)))
    summary = json.loads(
        (ROOT / "metadata" / "four_group_path_homology_summary.json").read_text(encoding="utf-8")
    )
    topology = pd.read_csv(ROOT / "metadata" / "four_group_topology_segments.csv")
    tests = pd.read_csv(ROOT / "metadata" / "four_group_statistical_tests.csv")
    pairwise = pd.read_csv(ROOT / "metadata" / "four_group_pairwise_tests.csv")
    for view in VIEW_LABELS:
        path = build_pdf(view, summary, topology, tests, pairwise)
        print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
