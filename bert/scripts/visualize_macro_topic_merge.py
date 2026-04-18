from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


DEFAULT_MERGE_DIR = Path(
    "bert/artifacts/broad_analysis/topic_model_compare/O_outlier_manual_merge_v1"
)
DEFAULT_OUTPUT_FILENAME = "macro_topic_dashboard.html"

ACTION_COLORS = {
    "keep": "#1b9e77",
    "weak": "#e67e22",
    "omit": "#7f7f7f",
}
TOPIC_PALETTE = [
    "#d1495b",
    "#edae49",
    "#00798c",
    "#66a182",
    "#7f5539",
    "#9c6644",
    "#6d597a",
    "#386641",
    "#bc4749",
    "#8d99ae",
    "#588157",
    "#ef476f",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive HTML dashboard for a manual macro-topic merge result."
    )
    parser.add_argument(
        "--merge_dir",
        default=str(DEFAULT_MERGE_DIR),
        help="Directory containing macro_topic_overview.csv and macro_topic_share_by_period.csv.",
    )
    parser.add_argument(
        "--output_html",
        default=None,
        help="Optional output HTML path. Defaults to <merge_dir>/visualizations/macro_topic_dashboard.html.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=12,
        help="Number of keep macro topics to show in the trend line chart.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def normalize_inputs(overview: pd.DataFrame, trends: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overview = overview.copy()
    trends = trends.copy()

    for frame, numeric_columns in (
        (
            overview,
            [
                "topic_n",
                "doc_n",
                "share_of_clustered_docs_pct",
                "peak_doc_count",
                "peak_doc_share",
            ],
        ),
        (
            trends,
            ["doc_count", "period_total", "doc_share"],
        ),
    ):
        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

    overview["report_action"] = overview["report_action"].astype("string").fillna("unknown")
    trends["macro_report_action"] = trends["macro_report_action"].astype("string").fillna("unknown")

    overview["period_order"] = pd.to_datetime(overview["peak_period"].astype("string") + "-01", errors="coerce")
    trends["period_order"] = pd.to_datetime(trends["period_label"].astype("string") + "-01", errors="coerce")
    trends = trends.sort_values(["period_order", "macro_topic"], ascending=[True, True]).reset_index(drop=True)
    overview = overview.sort_values(["doc_n", "macro_topic"], ascending=[False, True]).reset_index(drop=True)
    return overview, trends


def format_pct(value: float) -> str:
    return f"{value:.2f}%"


def build_metric_cards(overview: pd.DataFrame) -> str:
    share_by_action = (
        overview.groupby("report_action", dropna=False)["share_of_clustered_docs_pct"].sum().to_dict()
    )
    total_docs = int(overview["doc_n"].sum())
    metric_specs = [
        ("宏观主题数", f"{len(overview)}", "本轮人工合并后的中层主题总数"),
        ("覆盖文本数", f"{total_docs:,}", "所有已聚类文本中落入宏观主题的文档量"),
        ("keep 占比", format_pct(float(share_by_action.get("keep", 0.0))), "可直接进入 substantive 报告主线"),
        ("weak 占比", format_pct(float(share_by_action.get("weak", 0.0))), "边缘场景或补充簇，适合辅助解释"),
        ("omit 占比", format_pct(float(share_by_action.get("omit", 0.0))), "平台模板、虚词混合或强噪声簇"),
    ]
    blocks = []
    for title, value, note in metric_specs:
        blocks.append(
            "\n".join(
                [
                    '<div class="metric-card">',
                    f'  <div class="metric-title">{html.escape(title)}</div>',
                    f'  <div class="metric-value">{html.escape(value)}</div>',
                    f'  <div class="metric-note">{html.escape(note)}</div>',
                    "</div>",
                ]
            )
        )
    return "\n".join(blocks)


def build_top_table(overview: pd.DataFrame, top_n: int = 15) -> str:
    display = overview[
        [
            "macro_topic",
            "report_action",
            "doc_n",
            "share_of_clustered_docs_pct",
            "peak_period",
            "peak_doc_share",
        ]
    ].head(top_n).copy()
    display["doc_n"] = display["doc_n"].map(lambda value: f"{int(value):,}")
    display["share_of_clustered_docs_pct"] = display["share_of_clustered_docs_pct"].map(format_pct)
    display["peak_doc_share"] = display["peak_doc_share"].map(format_pct)
    display = display.rename(
        columns={
            "macro_topic": "macro_topic",
            "report_action": "action",
            "doc_n": "doc_n",
            "share_of_clustered_docs_pct": "share",
            "peak_period": "peak_period",
            "peak_doc_share": "peak_share",
        }
    )
    header = "".join(f"<th>{html.escape(column)}</th>" for column in display.columns)
    rows: list[str] = []
    for _, row in display.iterrows():
        action = str(row["action"])
        cells = []
        for column, value in row.items():
            if column == "action":
                cells.append(
                    f'<td><span class="action-chip action-{html.escape(action)}">{html.escape(str(value))}</span></td>'
                )
            else:
                cells.append(f"<td>{html.escape(str(value))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(
        [
            '<table class="summary-table">',
            f"<thead><tr>{header}</tr></thead>",
            "<tbody>",
            *rows,
            "</tbody>",
            "</table>",
        ]
    )


def build_rank_figure(overview: pd.DataFrame) -> go.Figure:
    ordered = overview.sort_values(["share_of_clustered_docs_pct", "doc_n"], ascending=[True, True]).copy()
    ordered["hover_terms"] = ordered["example_terms"].fillna("").astype(str).str.slice(0, 220)
    figure = go.Figure()
    for action in ("keep", "weak", "omit"):
        subset = ordered[ordered["report_action"] == action]
        if subset.empty:
            continue
        figure.add_bar(
            x=subset["share_of_clustered_docs_pct"],
            y=subset["macro_topic"],
            orientation="h",
            name=action,
            marker_color=ACTION_COLORS.get(action, "#999999"),
            customdata=subset[
                [
                    "doc_n",
                    "topic_n",
                    "peak_period",
                    "peak_doc_share",
                    "hover_terms",
                ]
            ],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "share: %{x:.2f}%<br>"
                "docs: %{customdata[0]:,.0f}<br>"
                "topics merged: %{customdata[1]:,.0f}<br>"
                "peak period: %{customdata[2]}<br>"
                "peak share in month: %{customdata[3]:.2f}%<br>"
                "example terms: %{customdata[4]}<extra></extra>"
            ),
        )
    figure.update_layout(
        barmode="stack",
        title="宏观主题体量排序",
        xaxis_title="share of clustered docs (%)",
        yaxis_title="",
        height=max(720, len(ordered) * 24),
        margin=dict(l=20, r=20, t=60, b=40),
        legend_title_text="report_action",
    )
    return figure


def build_action_figure(trends: pd.DataFrame) -> go.Figure:
    summary = (
        trends.groupby(["period_label", "macro_report_action"], as_index=False)[["doc_count", "doc_share"]]
        .sum()
    )
    summary["period_order"] = pd.to_datetime(summary["period_label"].astype("string") + "-01", errors="coerce")
    summary = summary.sort_values(["period_order", "macro_report_action", "period_label"])
    figure = go.Figure()
    for action in ("keep", "weak", "omit"):
        subset = summary[summary["macro_report_action"] == action]
        if subset.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=subset["period_label"],
                y=subset["doc_share"] * 100,
                mode="lines",
                name=action,
                stackgroup="one",
                line=dict(width=1.5, color=ACTION_COLORS.get(action, "#999999")),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "period: %{x}<br>"
                    "share in month: %{y:.2f}%<br>"
                    "<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="keep / weak / omit 的月度占比结构",
        xaxis_title="period",
        yaxis_title="share in month (%)",
        hovermode="x unified",
        height=420,
        margin=dict(l=20, r=20, t=60, b=40),
    )
    return figure


def build_trend_figure(overview: pd.DataFrame, trends: pd.DataFrame, top_n: int) -> go.Figure:
    focus_topics = (
        overview[overview["report_action"] == "keep"]
        .sort_values(["doc_n", "macro_topic"], ascending=[False, True])
        .head(top_n)["macro_topic"]
        .tolist()
    )
    subset = trends[trends["macro_topic"].isin(focus_topics)].copy()
    figure = go.Figure()
    for index, topic in enumerate(focus_topics):
        topic_rows = subset[subset["macro_topic"] == topic]
        if topic_rows.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=topic_rows["period_label"],
                y=topic_rows["doc_share"] * 100,
                mode="lines",
                name=topic,
                line=dict(width=2, color=TOPIC_PALETTE[index % len(TOPIC_PALETTE)]),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "period: %{x}<br>"
                    "share in month: %{y:.3f}%<br>"
                    "doc count: %{customdata:,.0f}<extra></extra>"
                ),
                customdata=topic_rows["doc_count"],
            )
        )
    figure.update_layout(
        title=f"Top {len(focus_topics)} keep 主题的时间走势",
        xaxis_title="period",
        yaxis_title="share in month (%)",
        hovermode="x unified",
        height=520,
        margin=dict(l=20, r=20, t=60, b=40),
        legend_title_text="macro_topic",
    )
    return figure


def build_heatmap_figure(overview: pd.DataFrame, trends: pd.DataFrame) -> go.Figure:
    ordered_topics = overview.sort_values(
        ["report_action", "doc_n", "macro_topic"],
        ascending=[True, False, True],
    )["macro_topic"].tolist()
    pivot = (
        trends.pivot_table(
            index="macro_topic",
            columns="period_label",
            values="doc_share",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(ordered_topics)
        .fillna(0.0)
    )
    action_lookup = overview.set_index("macro_topic")["report_action"].to_dict()
    y_labels = [f"[{action_lookup.get(topic, '?')}] {topic}" for topic in pivot.index]
    figure = go.Figure(
        data=[
            go.Heatmap(
                z=pivot.to_numpy() * 100,
                x=pivot.columns.tolist(),
                y=y_labels,
                colorscale="YlOrRd",
                colorbar=dict(title="share %"),
                hovertemplate="period: %{x}<br>topic: %{y}<br>share: %{z:.3f}%<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title="宏观主题热力图",
        xaxis_title="period",
        yaxis_title="macro_topic",
        height=max(900, len(y_labels) * 26),
        margin=dict(l=20, r=20, t=60, b=40),
    )
    return figure


def render_html(
    overview: pd.DataFrame,
    output_html: Path,
    rank_figure: go.Figure,
    action_figure: go.Figure,
    trend_figure: go.Figure,
    heatmap_figure: go.Figure,
    top_n: int,
) -> None:
    keep_topics = (
        overview[overview["report_action"] == "keep"]
        .sort_values(["doc_n", "macro_topic"], ascending=[False, True])["macro_topic"]
        .head(5)
        .tolist()
    )
    highlights = " / ".join(keep_topics)
    cards_html = build_metric_cards(overview)
    table_html = build_top_table(overview, top_n=15)
    sections = [
        rank_figure.to_html(full_html=False, include_plotlyjs="inline", config={"displayModeBar": False}),
        action_figure.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False}),
        trend_figure.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False}),
        heatmap_figure.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False}),
    ]
    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Macro Topic Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f4ef;
      --panel: #fffdf8;
      --ink: #1f2933;
      --muted: #5b6773;
      --line: #d8d2c4;
      --keep: {ACTION_COLORS["keep"]};
      --weak: {ACTION_COLORS["weak"]};
      --omit: {ACTION_COLORS["omit"]};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.5;
    }}
    .page {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    .hero {{
      padding: 18px 0 10px;
    }}
    h1, h2 {{
      margin: 0 0 10px;
      font-weight: 700;
      line-height: 1.15;
    }}
    h1 {{ font-size: 32px; }}
    h2 {{ font-size: 22px; margin-top: 28px; }}
    p {{
      margin: 0 0 12px;
      color: var(--muted);
      max-width: 960px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 20px 0 14px;
    }}
    .metric-card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .metric-card {{
      padding: 14px 14px 12px;
      min-height: 126px;
    }}
    .metric-title {{
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .metric-value {{
      font-size: 28px;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .metric-note {{
      font-size: 13px;
      color: var(--muted);
    }}
    .panel {{
      padding: 14px;
      margin-top: 14px;
      overflow: hidden;
    }}
    .summary-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .summary-table th,
    .summary-table td {{
      text-align: left;
      padding: 9px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    .summary-table th {{
      color: var(--muted);
      font-weight: 600;
      font-size: 13px;
    }}
    .action-chip {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 6px;
      color: white;
      font-size: 12px;
      font-weight: 600;
      line-height: 1.6;
    }}
    .action-keep {{ background: var(--keep); }}
    .action-weak {{ background: var(--weak); }}
    .action-omit {{ background: var(--omit); }}
    .footnote {{
      margin-top: 14px;
      font-size: 13px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>O_outlier_manual_merge_v1 可视化</h1>
      <p>这份面板把人工合并后的宏观主题结果压缩成一眼能读的几层：总体规模、keep/weak/omit 结构、主干主题时间走势、以及全量主题热力图。顶部最值得先看的主干主题是：{html.escape(highlights)}。</p>
    </section>

    <section class="metrics">
      {cards_html}
    </section>

    <section class="panel">
      <h2>Top 宏观主题</h2>
      <p>先看哪几个 macro topic 真正占了体量，再回到时间图里看它们是持续存在，还是阶段性冲高。</p>
      {table_html}
    </section>

    <section class="panel">
      <h2>体量排序</h2>
      <p>横向堆叠条形图直接把 keep / weak / omit 的量级放在同一张图里，能很快看出主线主题和边缘簇之间的距离。</p>
      {sections[0]}
    </section>

    <section class="panel">
      <h2>结构占比</h2>
      <p>这一张看的是每个月里 keep、weak、omit 分别占了多少。它更适合回答“这套 merge 在不同时间点上到底干不干净”。</p>
      {sections[1]}
    </section>

    <section class="panel">
      <h2>主干走势</h2>
      <p>这里默认只放前 {top_n} 个 keep 主题，避免线太多看不清。它比较适合直接拿去写“哪些经验在什么时候更突出”。</p>
      {sections[2]}
    </section>

    <section class="panel">
      <h2>全量热力图</h2>
      <p>热力图保留所有宏观主题，左侧前缀标了 action。适合找阶段性突出的弱主题，也适合回头检查合并后有没有结构失衡。</p>
      {sections[3]}
      <div class="footnote">文件生成路径：{html.escape(str(output_html))}</div>
    </section>
  </main>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    merge_dir = Path(args.merge_dir).resolve()
    output_html = (
        Path(args.output_html).resolve()
        if args.output_html
        else (merge_dir / "visualizations" / DEFAULT_OUTPUT_FILENAME).resolve()
    )

    overview = load_csv(merge_dir / "macro_topic_overview.csv")
    trends = load_csv(merge_dir / "macro_topic_share_by_period.csv")
    overview, trends = normalize_inputs(overview, trends)

    rank_figure = build_rank_figure(overview)
    action_figure = build_action_figure(trends)
    trend_figure = build_trend_figure(overview, trends, top_n=max(1, args.top_n))
    heatmap_figure = build_heatmap_figure(overview, trends)

    render_html(
        overview=overview,
        output_html=output_html,
        rank_figure=rank_figure,
        action_figure=action_figure,
        trend_figure=trend_figure,
        heatmap_figure=heatmap_figure,
        top_n=max(1, args.top_n),
    )
    print(output_html)


if __name__ == "__main__":
    main()
