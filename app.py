"""
LLM-Powered Unstructured Data Enrichment Pipeline
Streamlit Dashboard — Data Science Portfolio Project 2
Author: Abhishek Singh Bhadouria | Técnico Lisboa
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
import os
from pathlib import Path

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Enrichment Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.main { background-color: #0a0e1a; }
.stApp { background-color: #0a0e1a; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2234 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #63b3ed;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.78rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* Severity badges */
.badge-high   { background:#7f1d1d; color:#fca5a5; border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:500; }
.badge-medium { background:#78350f; color:#fcd34d; border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:500; }
.badge-low    { background:#064e3b; color:#6ee7b7; border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:500; }

/* Pipeline header */
.pipeline-header {
    background: linear-gradient(90deg, #0f172a, #1e3a5f, #0f172a);
    border-bottom: 1px solid #2d3748;
    padding: 1rem 2rem;
    margin-bottom: 1.5rem;
}
.section-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4a90d9;
    margin-bottom: 8px;
    font-weight: 500;
}

/* Record detail box */
.record-box {
    background: #111827;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.7;
    color: #a0aec0;
}
.key { color: #63b3ed; }
.val-str { color: #68d391; }
.val-num { color: #f6ad55; }
.val-bool-t { color: #fc8181; }
.val-bool-f { color: #718096; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    base = Path(__file__).parent
    paths = [
        base / "data" / "enriched_sample.csv",
        base / "enriched_sample.csv",
        "enriched_sample.csv",
        "data/enriched_sample.csv",
    ]
    for p in paths:
        if Path(p).exists():
            df = pd.read_csv(p)
            # Normalise column names
            df.columns = [c.strip() for c in df.columns]
            return df
    st.error("enriched_sample.csv not found. Run `python src/llm_enricher.py` first.")
    st.stop()


df = load_data()

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Controls")
    st.divider()

    st.markdown("### Filter Records")
    sev_filter = st.multiselect(
        "Severity", ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )
    fault_options = sorted(df["llm_fault_type"].dropna().unique().tolist())
    fault_filter = st.multiselect("Fault Type", fault_options, default=fault_options)
    action_filter = st.checkbox("Action Required Only", False)
    conf_min = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)

    st.divider()
    st.markdown("### About")
    st.markdown("""
    **Project 2** of a Data Science portfolio.  
    Uses **Claude API** to extract structured fields
    from free-text railway maintenance notes.
    
    *Stack:* Python · Anthropic SDK · Streamlit · Plotly
    """)
    st.markdown("---")
    st.caption("Abhi Bhadouria · Técnico Lisboa · 2024")


# ──────────────────────────────────────────────
# Filtered data
# ──────────────────────────────────────────────
fdf = df.copy()
fdf = fdf[fdf["llm_severity"].isin(sev_filter)]
fdf = fdf[fdf["llm_fault_type"].isin(fault_filter)]
if action_filter:
    fdf = fdf[fdf["llm_action_required"] == True]
fdf = fdf[fdf["llm_confidence"] >= conf_min]


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:1.5rem;'>
  <h1 style='font-size:1.8rem;font-weight:600;color:#e2e8f0;margin:0;'>
    🔬 LLM-Powered Unstructured Data Enrichment Pipeline
  </h1>
  <p style='color:#718096;margin-top:0.3rem;font-size:0.9rem;'>
    Railway Maintenance Notes → Structured Intelligence via Claude API
  </p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# KPI Row
# ──────────────────────────────────────────────
def eval_metrics(d):
    """Quick accuracy on available data."""
    if "true_severity" not in d.columns:
        return None
    match = (d["true_severity"] == d["llm_severity"]).sum()
    return round(match / len(d) * 100, 1) if len(d) > 0 else 0

acc = eval_metrics(fdf)
action_pct = round(fdf["llm_action_required"].mean() * 100, 1) if len(fdf) else 0
mean_conf = round(fdf["llm_confidence"].mean() * 100, 1) if len(fdf) else 0
high_sev_count = (fdf["llm_severity"] == "High").sum()

c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    (len(fdf), "Records Enriched"),
    (f"{acc}%" if acc is not None else "—", "Severity Accuracy"),
    (f"{mean_conf}%", "Mean Confidence"),
    (f"{action_pct}%", "Action Required"),
    (high_sev_count, "High Severity"),
]
for col, (val, label) in zip([c1, c2, c3, c4, c5], cards):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-val">{val}</div>
      <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Pipeline Architecture Diagram (static visual)
# ──────────────────────────────────────────────
with st.expander("📐 Pipeline Architecture", expanded=False):
    arch_cols = st.columns([1, 1, 1, 1, 1])
    stages = [
        ("📄", "Raw Notes", "Free-text technician notes (unstructured)"),
        ("⚙️", "Enricher", "LLM prompt → JSON extraction per record"),
        ("🧠", "Claude API", "claude-haiku-4-5 · structured JSON output"),
        ("✅", "Validator", "Schema validation + heuristic fallback"),
        ("📊", "Dashboard", "Streamlit + Plotly visualisation"),
    ]
    for col, (icon, title, desc) in zip(arch_cols, stages):
        col.markdown(f"""
        <div style='text-align:center;padding:1rem;background:#111827;border-radius:10px;border:1px solid #2d3748;'>
          <div style='font-size:1.8rem;'>{icon}</div>
          <div style='font-size:0.85rem;font-weight:600;color:#e2e8f0;margin-top:0.3rem;'>{title}</div>
          <div style='font-size:0.72rem;color:#718096;margin-top:0.3rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)


st.divider()

# ──────────────────────────────────────────────
# Tab layout
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Analytics", "🧪 Evaluation", "🔍 Record Explorer", "📝 Live Demo"
])


# ──────────────────────────────────────────────
# TAB 1: Analytics
# ──────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    # Severity distribution
    with col_a:
        st.markdown('<div class="section-title">Severity Distribution (LLM)</div>', unsafe_allow_html=True)
        sev_counts = fdf["llm_severity"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
        fig_sev = go.Figure(go.Bar(
            x=sev_counts.index.tolist(),
            y=sev_counts.values.tolist(),
            marker_color=["#fc8181", "#f6ad55", "#68d391"],
            text=sev_counts.values.tolist(),
            textposition="outside",
        ))
        fig_sev.update_layout(
            plot_bgcolor="#111827", paper_bgcolor="#111827",
            font_color="#a0aec0", height=300, margin=dict(t=20, b=20, l=20, r=20),
            xaxis=dict(gridcolor="#2d3748"), yaxis=dict(gridcolor="#2d3748"),
            showlegend=False
        )
        st.plotly_chart(fig_sev, use_container_width=True)

    # Fault type distribution
    with col_b:
        st.markdown('<div class="section-title">Fault Type Distribution</div>', unsafe_allow_html=True)
        ft_counts = fdf["llm_fault_type"].value_counts()
        colors = px.colors.qualitative.Pastel
        fig_ft = go.Figure(go.Pie(
            labels=ft_counts.index.tolist(),
            values=ft_counts.values.tolist(),
            hole=0.4,
            marker_colors=colors[:len(ft_counts)],
        ))
        fig_ft.update_layout(
            plot_bgcolor="#111827", paper_bgcolor="#111827",
            font_color="#a0aec0", height=300, margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(bgcolor="#111827", font_color="#a0aec0")
        )
        st.plotly_chart(fig_ft, use_container_width=True)

    col_c, col_d = st.columns(2)

    # Confidence histogram
    with col_c:
        st.markdown('<div class="section-title">LLM Confidence Distribution</div>', unsafe_allow_html=True)
        fig_conf = go.Figure(go.Histogram(
            x=fdf["llm_confidence"].dropna().tolist(),
            nbinsx=20,
            marker_color="#63b3ed",
            opacity=0.8
        ))
        fig_conf.update_layout(
            plot_bgcolor="#111827", paper_bgcolor="#111827",
            font_color="#a0aec0", height=280, margin=dict(t=20, b=20, l=20, r=20),
            xaxis=dict(title="Confidence", gridcolor="#2d3748"),
            yaxis=dict(title="Count", gridcolor="#2d3748"),
            bargap=0.05
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    # Fault × Severity heatmap
    with col_d:
        st.markdown('<div class="section-title">Fault Type × Severity Heatmap</div>', unsafe_allow_html=True)
        cross = pd.crosstab(fdf["llm_fault_type"], fdf["llm_severity"]).reindex(
            columns=["Low", "Medium", "High"], fill_value=0)
        fig_heat = go.Figure(go.Heatmap(
            z=cross.values.tolist(),
            x=["Low", "Medium", "High"],
            y=cross.index.tolist(),
            colorscale="Blues",
            text=cross.values.tolist(),
            texttemplate="%{text}",
        ))
        fig_heat.update_layout(
            plot_bgcolor="#111827", paper_bgcolor="#111827",
            font_color="#a0aec0", height=280, margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Urgency days strip
    st.markdown('<div class="section-title">Urgency Days by Severity</div>', unsafe_allow_html=True)
    urg_df = fdf[fdf["llm_urgency_days"].notna()].copy()
    if not urg_df.empty:
        fig_urg = px.strip(
            urg_df, x="llm_urgency_days", y="llm_severity",
            color="llm_severity",
            color_discrete_map={"High": "#fc8181", "Medium": "#f6ad55", "Low": "#68d391"},
        )
        fig_urg.update_layout(
            plot_bgcolor="#111827", paper_bgcolor="#111827",
            font_color="#a0aec0", height=260, margin=dict(t=10, b=20, l=20, r=20),
            showlegend=False,
            xaxis=dict(title="Days to Next Required Action", gridcolor="#2d3748"),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_urg, use_container_width=True)
    else:
        st.info("No urgency data available in current filter.")


# ──────────────────────────────────────────────
# TAB 2: Evaluation
# ──────────────────────────────────────────────
with tab2:
    st.markdown("### 🧪 LLM vs Ground Truth Evaluation")

    if "true_severity" not in fdf.columns:
        st.warning("Ground truth column `true_severity` not found.")
    else:
        eval_df = fdf.dropna(subset=["true_severity", "llm_severity"])

        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, accuracy_score

        labels = ["Low", "Medium", "High"]
        y_true = eval_df["true_severity"]
        y_pred = eval_df["llm_severity"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        col_ev1, col_ev2 = st.columns([1.2, 1])

        with col_ev1:
            st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
            fig_cm = go.Figure(go.Heatmap(
                z=cm.tolist(),
                x=labels, y=labels,
                colorscale="Blues",
                text=cm.tolist(),
                texttemplate="%{text}",
                showscale=False,
            ))
            fig_cm.update_layout(
                plot_bgcolor="#111827", paper_bgcolor="#111827",
                font_color="#a0aec0", height=360,
                margin=dict(t=30, b=30, l=30, r=30),
                xaxis=dict(title="Predicted"),
                yaxis=dict(title="True", autorange="reversed"),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_ev2:
            st.markdown('<div class="section-title">Classification Metrics</div>', unsafe_allow_html=True)
            report = classification_report(y_true, y_pred, labels=labels,
                                           output_dict=True, zero_division=0)
            kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)
            exact_acc = accuracy_score(y_true, y_pred)

            st.metric("Exact Accuracy", f"{exact_acc:.1%}")
            st.metric("Quadratic Kappa (κ)", f"{kappa:.4f}")
            st.metric("Mean Confidence", f"{eval_df['llm_confidence'].mean():.1%}")

            st.markdown("**Per-class F1 Scores**")
            for cls in labels:
                f1 = report.get(cls, {}).get("f1-score", 0)
                st.progress(f1, text=f"{cls}: {f1:.2f}")

        # Calibration scatter
        st.markdown('<div class="section-title">Confidence vs Correctness</div>', unsafe_allow_html=True)
        eval_df2 = eval_df.copy()
        eval_df2["correct"] = (eval_df2["true_severity"] == eval_df2["llm_severity"]).astype(int)
        eval_df2["jitter"] = eval_df2["correct"] + np.random.uniform(-0.05, 0.05, len(eval_df2))

        fig_cal = px.scatter(
            eval_df2, x="llm_confidence", y="jitter",
            color="llm_severity",
            color_discrete_map={"High": "#fc8181", "Medium": "#f6ad55", "Low": "#68d391"},
            hover_data=["record_id", "llm_fault_type"],
        )
        fig_cal.update_layout(
            plot_bgcolor="#111827", paper_bgcolor="#111827",
            font_color="#a0aec0", height=280, margin=dict(t=10, b=20, l=20, r=20),
            xaxis=dict(title="LLM Confidence", gridcolor="#2d3748"),
            yaxis=dict(title="Correct (1) / Wrong (0)", tickvals=[0, 1],
                       ticktext=["Wrong", "Correct"], gridcolor="#2d3748"),
            legend=dict(bgcolor="#111827")
        )
        st.plotly_chart(fig_cal, use_container_width=True)


# ──────────────────────────────────────────────
# TAB 3: Record Explorer
# ──────────────────────────────────────────────
with tab3:
    st.markdown("### 🔍 Record Explorer")

    col_sel, col_srt = st.columns([2, 1])
    with col_sel:
        rec_ids = fdf["record_id"].tolist()
        selected_id = st.selectbox("Select Record", rec_ids)
    with col_srt:
        sort_col = st.selectbox("Sort table by", ["record_id", "llm_severity", "llm_confidence", "llm_fault_type"])

    # Detail card
    row = fdf[fdf["record_id"] == selected_id].iloc[0]

    sev_badge = {
        "High": '<span class="badge-high">HIGH</span>',
        "Medium": '<span class="badge-medium">MEDIUM</span>',
        "Low": '<span class="badge-low">LOW</span>',
    }.get(str(row.get("llm_severity", "")), "")

    action_icon = "🔴 YES" if row.get("llm_action_required") else "🟢 NO"

    col_d1, col_d2 = st.columns([1.6, 1])

    with col_d1:
        st.markdown("**Original Maintenance Note**")
        st.info(str(row.get("maintenance_note", "")))

    with col_d2:
        st.markdown("**LLM Extraction Output**")
        urgency_str = f'{int(row["llm_urgency_days"])} days' if pd.notna(row.get("llm_urgency_days")) else "—"
        st.markdown(f"""
        <div class="record-box">
          <span class="key">fault_type   </span><span class="val-str">"{row.get('llm_fault_type','')}"</span><br>
          <span class="key">severity     </span><span class="val-str">"{row.get('llm_severity','')}"</span><br>
          <span class="key">component    </span><span class="val-str">"{row.get('llm_component','')}"</span><br>
          <span class="key">action_req   </span><span class="{'val-bool-t' if row.get('llm_action_required') else 'val-bool-f'}">{str(row.get('llm_action_required','')).lower()}</span><br>
          <span class="key">urgency_days </span><span class="val-num">{urgency_str}</span><br>
          <span class="key">confidence   </span><span class="val-num">{row.get('llm_confidence',''):.2f}</span><br>
          <span class="key">reasoning    </span><span class="val-str">"{str(row.get('llm_reasoning',''))[:80]}…"</span>
        </div>
        """, unsafe_allow_html=True)

        if "true_severity" in row:
            match = row.get("true_severity") == row.get("llm_severity")
            st.markdown(
                f"**Ground Truth:** `{row.get('true_severity')}` → "
                f"{'✅ Match' if match else '❌ Mismatch'}"
            )

    st.markdown("---")
    st.markdown("**Full Enriched Dataset**")
    display_cols = [c for c in [
        "record_id", "asset_type", "date", "llm_fault_type",
        "llm_severity", "llm_action_required", "llm_urgency_days",
        "llm_confidence", "true_severity"
    ] if c in fdf.columns]

    sorted_df = fdf.sort_values(sort_col, ascending=(sort_col == "record_id"))
    st.dataframe(
        sorted_df[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=380
    )


# ──────────────────────────────────────────────
# TAB 4: Live Demo
# ──────────────────────────────────────────────
with tab4:
    st.markdown("### 📝 Live Prompt Demo")
    st.markdown(
        "Type any maintenance note below. The app will call **Claude API** to extract "
        "structured fields in real time. Set your `ANTHROPIC_API_KEY` env variable."
    )

    demo_note = st.text_area(
        "Maintenance Note",
        value="Driver reported excessive vibration above 90 km/h. "
              "Investigated left wheelset — flange thickness measured at 24.2 mm, "
              "approaching lower limit of 22 mm. Recommend reprofiling within 5 days.",
        height=110,
    )
    demo_asset = st.selectbox("Asset Type", [
        "Wheelset", "Bogie Frame", "Axle Box", "Brake System",
        "Pantograph", "Traction Motor"
    ])
    demo_date = st.date_input("Date", value=pd.Timestamp("2024-01-15"))

    if st.button("🚀 Run Extraction", type="primary"):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            st.error("ANTHROPIC_API_KEY not set. Showing mock output for demo.")
            mock = {
                "fault_type": "wear",
                "severity": "Medium",
                "component": "wheelset flange",
                "action_required": True,
                "urgency_days": 5,
                "confidence": 0.93,
                "reasoning": "Flange approaching lower limit — reprofiling required within 5 days; Medium severity as limit not yet breached."
            }
            with st.spinner("Calling LLM extraction engine..."):
                time.sleep(1.5)
            st.success("Extraction complete (mock output)")
            st.json(mock)
        else:
            try:
                import sys
                sys.path.append(str(Path(__file__).parent / "src"))
                from llm_enricher import LLMEnricher
                enricher = LLMEnricher(api_key=api_key)
                with st.spinner("Calling Claude API..."):
                    result = enricher.extract(
                        note=demo_note,
                        asset_type=demo_asset,
                        date=str(demo_date)
                    )
                st.success("Extraction complete")
                st.json(result)
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("**System Prompt (used for every record)**")
    st.code("""You are a railway maintenance data analyst. Extract structured fields
from free-text maintenance notes. Return ONLY valid JSON with keys:
fault_type | severity | component | action_required | urgency_days | confidence | reasoning
""", language="text")

    st.markdown("**Example API call (Python)**")
    st.code("""import anthropic, json

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=300,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": note_text}]
)
extracted = json.loads(response.content[0].text)
""", language="python")
