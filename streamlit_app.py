"""
HubGenes Streamlit app.
Run from repo root: streamlit run streamlit_app.py
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from hubgenes.pipeline import rank_genes_from_interactions


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --gp-accent: #15803d;
                --gp-accent-2: #0f766e;
                --gp-surface-light: rgba(255, 255, 255, 0.94);
                --gp-surface-dark: rgba(15, 23, 42, 0.68);
                --gp-shadow: 0 24px 60px rgba(15, 23, 42, 0.16);
                --gp-shadow-soft: 0 14px 36px rgba(15, 23, 42, 0.10);
                --gp-text-strong: #0f172a;
                --gp-text: #1e293b;
                --gp-text-muted: #475569;
                --gp-text-soft: #64748b;
                --gp-border-light: rgba(148, 163, 184, 0.22);
            }

            html, body, [class*="css"] {
                font-family: "SF Pro Display", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            }

            body, p, label, span, div {
                color: var(--gp-text);
            }

            .stMarkdown,
            [data-testid="stMarkdownContainer"],
            .stCaption,
            [data-testid="stCaptionContainer"],
            [data-testid="stMetricLabel"],
            [data-testid="stMetricValue"],
            .stAlert,
            .stSubheader,
            h1, h2, h3 {
                color: var(--gp-text-strong) !important;
            }

            [data-testid="stAppViewContainer"],
            [data-testid="stAppViewContainer"] p,
            [data-testid="stAppViewContainer"] label,
            [data-testid="stAppViewContainer"] span,
            [data-testid="stAppViewContainer"] div,
            [data-testid="stAppViewContainer"] li {
                color: var(--gp-text) !important;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(21, 128, 61, 0.14), transparent 30%),
                    radial-gradient(circle at top right, rgba(14, 165, 233, 0.14), transparent 34%),
                    linear-gradient(180deg, rgba(248, 250, 252, 0.96), rgba(241, 245, 249, 1));
            }

            @media (prefers-color-scheme: dark) {
                .stApp {
                    background:
                        radial-gradient(circle at top left, rgba(34, 197, 94, 0.14), transparent 28%),
                        radial-gradient(circle at top right, rgba(45, 212, 191, 0.12), transparent 30%),
                        linear-gradient(180deg, #020617 0%, #0f172a 54%, #111827 100%);
                }
            }

            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 3rem;
                max-width: 1180px;
            }

            .gp-hero {
                position: relative;
                overflow: hidden;
                padding: 2rem 2rem 1.8rem 2rem;
                margin-bottom: 1.2rem;
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 28px;
                background:
                    linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(248, 250, 252, 0.88)),
                    linear-gradient(120deg, rgba(21, 128, 61, 0.10), rgba(8, 145, 178, 0.08));
                box-shadow: var(--gp-shadow);
                backdrop-filter: blur(18px);
            }

            @media (prefers-color-scheme: dark) {
                .gp-hero {
                    background:
                        linear-gradient(135deg, rgba(15, 23, 42, 0.86), rgba(15, 23, 42, 0.70)),
                        linear-gradient(120deg, rgba(21, 128, 61, 0.16), rgba(13, 148, 136, 0.14));
                    border-color: rgba(148, 163, 184, 0.14);
                    box-shadow: 0 24px 60px rgba(2, 6, 23, 0.40);
                }
            }

            .gp-hero-grid {
                display: grid;
                grid-template-columns: minmax(0, 1.65fr) minmax(260px, 0.8fr);
                gap: 1.25rem;
                align-items: center;
            }

            .gp-kicker {
                display: inline-block;
                padding: 0.42rem 0.78rem;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.14em;
                text-transform: uppercase;
                color: var(--gp-accent);
                margin-bottom: 0.8rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid rgba(21, 128, 61, 0.18);
                color: var(--gp-accent) !important;
            }

            .gp-hero h1 {
                font-size: clamp(2rem, 4vw, 3.1rem);
                font-weight: 800;
                letter-spacing: -0.03em;
                color: #0f172a !important;
                margin: 0 0 0.55rem 0;
                line-height: 1.05;
                opacity: 1 !important;
                text-shadow: none !important;
            }

            .gp-hero p {
                margin: 0;
                font-size: 1rem;
                color: #334155 !important;
                line-height: 1.7;
                max-width: 48rem;
                opacity: 1 !important;
            }

            .gp-hero-panel {
                padding: 1.1rem 1.15rem;
                border-radius: 22px;
                border: 1px solid rgba(148, 163, 184, 0.18);
                background: rgba(255, 255, 255, 0.86);
                box-shadow: var(--gp-shadow-soft);
            }

            .gp-panel-label {
                font-size: 0.74rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                font-weight: 700;
                color: var(--gp-text-muted) !important;
                margin-bottom: 0.45rem;
            }

            .gp-panel-value {
                font-size: 1.3rem;
                font-weight: 800;
                line-height: 1.15;
                color: var(--gp-text-strong) !important;
                margin-bottom: 0.35rem;
            }

            .gp-panel-copy {
                font-size: 0.92rem;
                color: var(--gp-text-muted) !important;
                line-height: 1.55;
                margin: 0;
            }

            .gp-section-label {
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.11em;
                text-transform: uppercase;
                color: #64748b !important;
                margin-bottom: 0.42rem;
                opacity: 1 !important;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: var(--gp-surface-light);
                border: 1px solid var(--gp-border-light) !important;
                border-radius: 22px !important;
                box-shadow: var(--gp-shadow-soft);
                backdrop-filter: blur(16px);
            }

            .stButton button[kind="primary"] {
                background: linear-gradient(135deg, var(--gp-accent), var(--gp-accent-2));
                color: #ffffff;
                border: none;
                font-weight: 700;
                padding: 0.75rem 1.25rem;
                border-radius: 14px;
                box-shadow: 0 16px 32px rgba(21, 128, 61, 0.24);
            }

            .stDownloadButton button {
                border-radius: 14px;
                font-weight: 700;
                border: 1px solid rgba(21, 128, 61, 0.18);
                background: rgba(255, 255, 255, 0.96);
                color: var(--gp-text-strong) !important;
            }

            .stFileUploader label,
            .stRadio label,
            .stMetric label,
            .stMetric div {
                color: var(--gp-text-strong) !important;
            }

            .stButton button,
            .stDownloadButton button,
            .stRadio p,
            .stFileUploader,
            .stFileUploader small,
            [data-testid="stExpanderToggleIcon"],
            [data-testid="stExpander"] summary,
            [data-testid="stMetricLabel"],
            [data-testid="stMetricValue"],
            [data-testid="stAlertContainer"],
            [data-testid="stAlertContainer"] *,
            [data-testid="stMarkdownContainer"] *,
            [data-testid="stCaptionContainer"] *,
            .stDataFrame,
            .stDataFrame * {
                opacity: 1 !important;
            }

            [data-testid="stMetricLabel"] {
                color: #64748b !important;
                font-weight: 600 !important;
                opacity: 1 !important;
            }

            [data-testid="stMetricValue"] {
                color: #0f172a !important;
                font-weight: 800 !important;
                opacity: 1 !important;
            }

            .stCaption,
            [data-testid="stCaptionContainer"],
            [data-testid="stCaptionContainer"] * {
                color: #475569 !important;
                opacity: 1 !important;
            }

            [data-testid="stExpander"] summary,
            [data-testid="stExpander"] summary * {
                color: #0f172a !important;
                opacity: 1 !important;
            }

            [data-testid="stFileUploaderDropzoneInstructions"],
            [data-testid="stFileUploaderDropzoneInstructions"] *,
            [data-testid="stBaseButton-secondary"],
            [data-testid="stWidgetLabel"],
            [data-testid="stWidgetLabel"] * {
                color: #0f172a !important;
                opacity: 1 !important;
            }

            [data-testid="stAlertContainer"] {
                border-radius: 16px !important;
            }

            [data-testid="stAlertContainer"] p,
            [data-testid="stAlertContainer"] span,
            [data-testid="stAlertContainer"] div {
                color: #ecfdf5 !important;
                font-weight: 600 !important;
            }

            [data-testid="stDataFrame"] {
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid rgba(148, 163, 184, 0.22);
            }

            .gp-footnote {
                font-size: 0.84rem;
                color: var(--gp-text-muted);
                line-height: 1.5;
                margin-top: 2rem;
                padding: 1rem 1.1rem 0 1.1rem;
                border-top: 1px solid rgba(148, 163, 184, 0.18);
                text-align: center;
            }

            @media (prefers-color-scheme: dark) {
                [data-testid="stAppViewContainer"],
                [data-testid="stAppViewContainer"] p,
                [data-testid="stAppViewContainer"] label,
                [data-testid="stAppViewContainer"] span,
                [data-testid="stAppViewContainer"] div,
                [data-testid="stAppViewContainer"] li,
                .stMarkdown,
                [data-testid="stMarkdownContainer"],
                .stCaption,
                [data-testid="stCaptionContainer"],
                .stSubheader,
                h1, h2, h3 {
                    color: #f1f5f9 !important;
                }

                .gp-kicker {
                    color: #86efac !important;
                    background: rgba(15, 23, 42, 0.64);
                    border-color: rgba(134, 239, 172, 0.18);
                }

                .gp-hero h1, .gp-panel-value {
                    color: #ffffff !important;
                }

                .gp-hero p, .gp-panel-copy, .gp-panel-label, .gp-footnote,
                .stMarkdown, [data-testid="stMarkdownContainer"],
                .stCaption, [data-testid="stCaptionContainer"],
                [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
                .stFileUploader label, .stRadio label, .stSubheader, h1, h2, h3 {
                    color: #f1f5f9 !important;
                }

                .gp-section-label {
                    color: #dbeafe !important;
                }

                .gp-hero-panel,
                div[data-testid="stVerticalBlockBorderWrapper"] {
                    background: var(--gp-surface-dark);
                }

                .stDownloadButton button {
                    background: rgba(15, 23, 42, 0.72);
                    border-color: rgba(134, 239, 172, 0.16);
                    color: #f8fafc !important;
                }

                [data-testid="stMetricLabel"] {
                    color: #e2e8f0 !important;
                }

                [data-testid="stMetricValue"] {
                    color: #ffffff !important;
                }

                .stCaption,
                [data-testid="stCaptionContainer"],
                [data-testid="stCaptionContainer"] * {
                    color: #e2e8f0 !important;
                }

                [data-testid="stExpander"] summary,
                [data-testid="stExpander"] summary * {
                    color: #f8fafc !important;
                }

                [data-testid="stFileUploaderDropzoneInstructions"],
                [data-testid="stFileUploaderDropzoneInstructions"] *,
                [data-testid="stBaseButton-secondary"],
                [data-testid="stWidgetLabel"],
                [data-testid="stWidgetLabel"] * {
                    color: #f8fafc !important;
                }

                [data-testid="stAlertContainer"] p,
                [data-testid="stAlertContainer"] span,
                [data-testid="stAlertContainer"] div {
                    color: #ffffff !important;
                }

                .stDownloadButton button,
                .stButton button,
                .stRadio p,
                .stRadio label,
                .stFileUploader,
                .stFileUploader small,
                .stDataFrame,
                .stDataFrame * {
                    color: #f8fafc !important;
                }
            }

            @media (max-width: 900px) {
                .gp-hero-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_uploaded_table(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    raw = uploaded.getvalue()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    if name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(io.BytesIO(raw), sep="\t")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    raise ValueError("Unsupported file type. Use .tsv, .csv, or .xlsx.")


st.set_page_config(
    page_title="HubGenes Ranking",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_inject_styles()

st.markdown(
    """
    <div class="gp-hero">
        <div class="gp-hero-grid">
            <div>
                <span class="gp-kicker">STRING network ranking</span>
                <h1>HubGenes Ranking</h1>
                <p>
                    Upload a STRING interaction file and rank genes using the workflow:
                    build the graph, compute betweenness, closeness, degree, and MCC, assign median-based
                    labels, then score genes with LASSO logistic regression, SVM-RFE, and Random Forest.
                </p>
            </div>
            <div class="gp-hero-panel">
                <div class="gp-panel-label">Current method</div>
                <div class="gp-panel-value">4 topology measures</div>
                <p class="gp-panel-copy">
                    Graph-native feature extraction, standardized model inputs, composite ranking, and export-ready outputs.
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown('<p class="gp-section-label">Shortlist size</p>', unsafe_allow_html=True)
    k = st.radio(
        "Number of top genes to display and export",
        options=[10, 20, 50],
        horizontal=True,
        index=0,
        label_visibility="collapsed",
    )

with st.container(border=True):
    st.markdown('<p class="gp-section-label">Upload STRING interactions</p>', unsafe_allow_html=True)
    st.caption(
        "Accepted: `.tsv`, `.csv`, `.xlsx`. Required edge columns can be `#node1` / `node2`, "
        "`protein1` / `protein2`, or similar source-target pairs. Extra columns like `combined_score` are allowed."
    )
    uploaded = st.file_uploader(
        "STRING interaction file",
        type=["tsv", "txt", "csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

interaction_df: pd.DataFrame | None = None

if uploaded is not None:
    try:
        interaction_df = _load_uploaded_table(uploaded)
        st.success(f"Loaded **{len(interaction_df):,}** interaction rows from `{uploaded.name}`.")
        with st.expander("Preview uploaded table", expanded=False):
            st.dataframe(interaction_df.head(20), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(str(e))

st.markdown("<br/>", unsafe_allow_html=True)
run = st.button(
    "Build topology + generate ranking",
    type="primary",
    disabled=interaction_df is None,
    use_container_width=True,
)

if run and interaction_df is not None:
    try:
        ranking, topology_df, info = rank_genes_from_interactions(interaction_df)
        top = ranking.head(int(k))
        graph_summary = info["graph_summary"]
        medians = info["medians"]
        evaluation = info["evaluation"]

        st.markdown('<p class="gp-section-label">Network summary</p>', unsafe_allow_html=True)
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Nodes", f"{graph_summary['nodes']:,}")
        g2.metric("Edges", f"{graph_summary['edges']:,}")
        g3.metric("Components", f"{graph_summary['connected_components']:,}")
        g4.metric("Density", f"{graph_summary['density']:.4f}")

        st.markdown('<p class="gp-section-label">Median thresholds</p>', unsafe_allow_html=True)
        median_df = pd.DataFrame(
            {
                "measure": ["betweenness", "closeness", "degree", "mcc"],
                "median_value": [
                    medians["betweenness"],
                    medians["closeness"],
                    medians["degree"],
                    medians["mcc"],
                ],
            }
        )
        st.dataframe(median_df, use_container_width=True, hide_index=True)

        st.markdown('<p class="gp-section-label">Results</p>', unsafe_allow_html=True)
        st.subheader(f"Top {k} genes")
        st.dataframe(
            top,
            use_container_width=True,
            hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", format="%d"),
                "name": st.column_config.TextColumn("Gene"),
                "betweenness": st.column_config.NumberColumn("Betweenness", format="%.6f"),
                "closeness": st.column_config.NumberColumn("Closeness", format="%.6f"),
                "degree": st.column_config.NumberColumn("Degree", format="%d"),
                "mcc": st.column_config.NumberColumn("MCC", format="%.3e"),
                "label": st.column_config.NumberColumn("Label", format="%d"),
                "composite_score": st.column_config.NumberColumn("Composite", format="%.6f"),
            },
        )

        b1, b2, b3 = st.columns(3)
        with b1:
            st.download_button(
                label=f"Download top {k} ranking (CSV)",
                data=top.to_csv(index=False).encode("utf-8"),
                file_name=f"hubgenes_top_{k}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with b2:
            st.download_button(
                label="Download full ranking (CSV)",
                data=ranking.to_csv(index=False).encode("utf-8"),
                file_name="hubgenes_full_ranking.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with b3:
            st.download_button(
                label="Download topology table (CSV)",
                data=topology_df.to_csv(index=False).encode("utf-8"),
                file_name="hubgenes_topology_features.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.caption(
            f"Train set: **{info['n_train']}** / {info['n_total']} · "
            f"Stratified split: **{info['used_stratify']}** · "
            f"SVM-RFE features: **{', '.join(info['rfe_features'])}**"
        )

        if not evaluation.empty:
            st.markdown('<p class="gp-section-label">Model evaluation</p>', unsafe_allow_html=True)
            st.dataframe(
                evaluation,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "test_accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
                    "test_roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
                },
            )

        with st.expander("Topology feature table", expanded=False):
            st.dataframe(
                topology_df.sort_values("mcc", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Full ranking", expanded=False):
            st.dataframe(ranking, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(str(e))

st.markdown(
    '<p class="gp-footnote">Ⓒ IIT BHU</p>',
    unsafe_allow_html=True,
)
