# FactoryLens | Streamlit Dashboard
# Manufacturing Intelligence & Predictive Maintenance Platform

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="FactoryLens | Manufacturing Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------
# Industrial UI Theme
# ----------------------------------------
st.markdown(
    """
    <style>
    /* Global */
    .stApp {
        background: #0B1220;
        color: #E5E7EB;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* Hide default Streamlit chrome for a cleaner product look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0A1020;
        border-right: 1px solid #1F2937;
    }

    [data-testid="stSidebarContent"] {
        padding-top: 2rem;
    }

    /* Typography */
    h1, h2, h3, h4 {
        color: #F8FAFC !important;
        letter-spacing: -0.02em;
    }

    p, label, .stMarkdown, .stCaption {
        color: #CBD5E1;
    }

    /* Product header */
    .hero {
        background: linear-gradient(135deg, #111827 0%, #0F172A 60%, #10233C 100%);
        border: 1px solid #243244;
        border-radius: 20px;
        padding: 26px 30px;
        margin-bottom: 22px;
        box-shadow: 0 14px 35px rgba(0, 0, 0, 0.20);
    }

    .hero-kicker {
        color: #38BDF8;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .hero-title {
        color: #F8FAFC;
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1.05;
        margin: 0;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 1rem;
        margin-top: 10px;
    }

    .status-pill {
        display: inline-block;
        margin-top: 16px;
        padding: 6px 11px;
        border-radius: 999px;
        background: #0C2638;
        border: 1px solid #174A63;
        color: #7DD3FC;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    /* KPI cards */
    .kpi-card {
        background: #111827;
        border: 1px solid #243244;
        border-radius: 16px;
        padding: 18px 18px 16px 18px;
        min-height: 122px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.14);
    }

    .kpi-label {
        color: #94A3B8;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        font-weight: 800;
    }

    .kpi-value {
        color: #F8FAFC;
        font-size: 1.75rem;
        font-weight: 800;
        margin-top: 8px;
        line-height: 1.0;
    }

    .kpi-note {
        color: #64748B;
        font-size: 0.78rem;
        margin-top: 7px;
    }

    /* Section headers */
    .section-kicker {
        color: #38BDF8;
        font-size: 0.70rem;
        font-weight: 800;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 3px;
    }

    .section-title {
        color: #F8FAFC;
        font-size: 1.45rem;
        font-weight: 800;
        margin-bottom: 2px;
    }

    .section-description {
        color: #94A3B8;
        font-size: 0.90rem;
        margin-bottom: 12px;
    }

    /* Generic panels */
    .panel {
        background: #111827;
        border: 1px solid #243244;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.14);
    }

    .panel-title {
        color: #F8FAFC;
        font-size: 0.98rem;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .panel-copy {
        color: #CBD5E1;
        font-size: 0.88rem;
        line-height: 1.65;
    }

    /* Insight badges */
    .insight-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        margin-bottom: 10px;
    }

    .badge-blue {
        color: #7DD3FC;
        background: #082638;
        border: 1px solid #16465E;
    }

    .badge-red {
        color: #FCA5A5;
        background: #35141A;
        border: 1px solid #6A2730;
    }

    .badge-amber {
        color: #FCD34D;
        background: #36270B;
        border: 1px solid #6C4C10;
    }

    /* Risk cards */
    .risk-card {
        border-radius: 18px;
        padding: 24px;
        text-align: center;
        border: 1px solid #243244;
        background: #111827;
        margin-bottom: 14px;
    }

    .risk-probability {
        font-size: 3.4rem;
        font-weight: 900;
        line-height: 1;
        color: #F8FAFC;
        margin: 4px 0 5px;
    }

    .risk-caption {
        color: #94A3B8;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        font-weight: 800;
    }

    .risk-level {
        font-size: 0.95rem;
        font-weight: 900;
        letter-spacing: 0.10em;
        margin-top: 14px;
    }

    .risk-high { color: #F87171; }
    .risk-medium { color: #FBBF24; }
    .risk-low { color: #4ADE80; }

    .recommendation {
        background: #0F172A;
        border: 1px solid #243244;
        border-radius: 14px;
        padding: 15px 16px;
        color: #CBD5E1;
        font-size: 0.84rem;
        line-height: 1.55;
        text-align: left;
    }

    /* Derived metric blocks */
    .derived {
        background: #0F172A;
        border: 1px solid #243244;
        border-radius: 14px;
        padding: 13px 15px;
    }

    .derived-label {
        color: #64748B;
        font-size: 0.70rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
    }

    .derived-value {
        color: #E2E8F0;
        font-size: 1.05rem;
        font-weight: 800;
        margin-top: 4px;
    }

    /* Small explanatory text */
    .muted {
        color: #64748B;
        font-size: 0.76rem;
    }

    /* Streamlit widget adjustments */
    [data-testid="stMetric"] {
        background: transparent;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #0F172A;
        border: 1px solid #243244;
        border-radius: 10px;
        padding: 8px 13px;
        color: #94A3B8;
    }

    .stTabs [aria-selected="true"] {
        background: #10283A;
        color: #7DD3FC !important;
        border-color: #1B5874;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background: #111827;
        border-radius: 12px;
        color: #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------
# Paths
# ----------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "MyFactorylens_dataset.csv"
MODEL_PATH = BASE_DIR / "factorylens_model.pkl"
FEATURES_PATH = BASE_DIR / "factorylens_features.pkl"

# ----------------------------------------
# Data + Model Loading
# ----------------------------------------
@st.cache_data

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource

def load_model(path: str):
    return joblib.load(path)


@st.cache_data

def load_features(path: str):
    return joblib.load(path)


@st.cache_resource

def build_shap_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data

def calculate_global_shap(_model, feature_names: tuple, data_sample: pd.DataFrame):
    """Calculate a compact global SHAP profile for dashboard display."""
    explainer = shap.TreeExplainer(_model)
    values = explainer.shap_values(data_sample)

    # SHAP API differs across versions. For binary tree models, handle
    # both (n_samples, n_features, 2) and list-of-arrays formats.
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    elif isinstance(values, np.ndarray) and values.ndim == 3:
        values = values[:, :, 1]

    values = np.asarray(values)
    mean_abs = np.abs(values).mean(axis=0)

    importance = (
        pd.DataFrame({
            "Feature": list(feature_names),
            "Mean SHAP Value": mean_abs
        })
        .sort_values("Mean SHAP Value", ascending=False)
        .reset_index(drop=True)
    )
    return importance


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not FEATURES_PATH.exists():
    st.error(f"Feature file not found: {FEATURES_PATH}")
    st.stop()


df = load_data(str(DATA_PATH))
model = load_model(str(MODEL_PATH))
feature_names = load_features(str(FEATURES_PATH))
feature_names = list(feature_names)

# ----------------------------------------
# Derived / Validation Columns
# ----------------------------------------
df = df.copy()
df["power_kw"] = (
    df["torque_nm"] * df["rotational_speed_rpm"]
) / 9549

df["temp_diff"] = df["process_temp_k"] - df["air_temp_k"]

wear_map = {
    "New": 0,
    "Mid-life": 1,
    "Aging": 2,
    "Critical": 3
}

df["tool_wear_category_num"] = df["machine_age_category"].map(wear_map)

df["failure_label"] = df["machine_failure"].map({0: "No Failure", 1: "Failure"})

# ----------------------------------------
# Sidebar
# ----------------------------------------
with st.sidebar:
    st.markdown("### 🏭 FACTORYLENS")
    st.caption("Manufacturing Intelligence Platform")
    st.markdown("---")

    st.markdown("**Dashboard Modules**")
    st.markdown("• Operations Overview")
    st.markdown("• Risk Driver Analysis")
    st.markdown("• Failure Risk Predictor")
    st.markdown("• Explainable ML")

    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("Random Forest Classifier")
    st.markdown("17 predictive features")

    st.markdown("---")
    st.markdown("**Data**")
    st.markdown(f"{len(df):,} machine observations")
    st.markdown("AI4I 2020 + operational enrichment")

    st.markdown("---")
    st.caption("FactoryLens is an analytical prototype. Operational variables such as shift, operator experience, and machine age are simulated enrichment fields.")

# ----------------------------------------
# Header
# ----------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">MANUFACTURING INTELLIGENCE</div>
        <div class="hero-title">🏭 FactoryLens</div>
        <div class="hero-subtitle">Predict • Diagnose • Prevent machine failures with data-driven operations analytics.</div>
        <div class="status-pill">● MODEL CONNECTED &nbsp;|&nbsp; RANDOM FOREST &nbsp;|&nbsp; 17 FEATURES</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------
# Operations Overview
# ----------------------------------------
st.markdown('<div class="section-kicker">01 · OPERATIONS OVERVIEW</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Plant Health Snapshot</div>', unsafe_allow_html=True)
st.markdown('<div class="section-description">A compact view of failure exposure, shift risk, and machine condition.</div>', unsafe_allow_html=True)

# Calculate KPI values dynamically
total_observations = len(df)
overall_failure_rate = df["machine_failure"].mean() * 100
shift_failure_rates = df.groupby("shift")["machine_failure"].mean() * 100
worst_shift = shift_failure_rates.idxmax()
worst_shift_rate = shift_failure_rates.max()
critical_pct = (df["machine_age_category"] == "Critical").mean() * 100

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Observations</div>
            <div class="kpi-value">{total_observations:,}</div>
            <div class="kpi-note">Machine records analyzed</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Overall Failure Rate</div>
            <div class="kpi-value">{overall_failure_rate:.2f}%</div>
            <div class="kpi-note">Historical failure incidence</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Highest Risk Shift</div>
            <div class="kpi-value">{worst_shift}</div>
            <div class="kpi-note">{worst_shift_rate:.2f}% failure rate</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Critical Machines</div>
            <div class="kpi-value">{critical_pct:.1f}%</div>
            <div class="kpi-note">Highest tool-wear state</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ----------------------------------------
# Operational Risk Overview
# ----------------------------------------
chart_col, insight_col = st.columns([1.65, 1], gap="large")

with chart_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Failure Rate by Shift</div>', unsafe_allow_html=True)

    shift_order = [s for s in ["Morning", "Evening", "Night"] if s in shift_failure_rates.index]
    shift_plot = shift_failure_rates.reindex(shift_order).dropna()

    fig, ax = plt.subplots(figsize=(6.2, 3.15))
    bars = ax.bar(
        shift_plot.index,
        shift_plot.values,
        width=0.58,
        color=["#334155", "#38BDF8", "#475569"]
    )

    for bar, value in zip(bars, shift_plot.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(shift_plot.values) * 0.035,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#E2E8F0",
            fontweight="bold"
        )

    ax.set_ylabel("Failure Rate (%)", color="#94A3B8", fontsize=9)
    ax.set_xlabel("")
    ax.tick_params(colors="#94A3B8", labelsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.grid(axis="y", alpha=0.14, color="#64748B")
    ax.set_axisbelow(True)
    ax.set_facecolor("#111827")
    fig.patch.set_facecolor("#111827")
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with insight_col:
    st.markdown(
        f"""
        <div class="panel">
            <div class="insight-badge badge-blue">OPERATIONAL SIGNAL</div>
            <div class="panel-title">Shift concentration</div>
            <div class="panel-copy">
                <strong style="color:#F8FAFC">{worst_shift}</strong> shows the highest observed failure rate at
                <strong style="color:#7DD3FC">{worst_shift_rate:.2f}%</strong>.
                The dashboard surfaces this as a monitoring priority rather than treating every shift equally.
            </div>
            <div style="height:14px"></div>
            <div class="derived">
                <div class="derived-label">Lowest-risk shift</div>
                <div class="derived-value">{shift_failure_rates.idxmin()} · {shift_failure_rates.min():.2f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------------------
# Business Drivers
# ----------------------------------------
st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-kicker">02 · RISK DRIVER ANALYSIS</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">What Drives Machine Failure?</div>', unsafe_allow_html=True)
st.markdown('<div class="section-description">Explore the six operational questions built into the original FactoryLens analysis.</div>', unsafe_allow_html=True)

# Tabs make each analysis screenshot-friendly and compact.
tab_shift, tab_tool, tab_product, tab_operator, tab_temp, tab_age = st.tabs([
    "Shift",
    "Tool Wear",
    "Product Type",
    "Operator",
    "Temperature",
    "Machine Age"
])


def styled_bar_chart(data, x_col, y_col, title, y_label, palette, figsize=(5.8, 3.15), order=None):
    fig, ax = plt.subplots(figsize=figsize)
    labels = data[x_col].astype(str).tolist()
    values = data[y_col].astype(float).tolist()
    colors = (palette * ((len(values) // len(palette)) + 1))[:len(values)]
    bars = ax.bar(labels, values, color=colors, width=0.58)

    offset = max(values) * 0.035 if max(values) > 0 else 0.1
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            color="#E2E8F0",
            fontsize=9,
            fontweight="bold"
        )

    ax.set_title(title, color="#F8FAFC", fontsize=11, fontweight="bold", pad=12, loc="left")
    ax.set_ylabel(y_label, color="#94A3B8", fontsize=8)
    ax.set_xlabel("")
    ax.tick_params(colors="#94A3B8", labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.grid(axis="y", alpha=0.13, color="#64748B")
    ax.set_axisbelow(True)
    ax.set_facecolor("#111827")
    fig.patch.set_facecolor("#111827")
    fig.tight_layout()
    return fig

with tab_shift:
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        data = (
            df.groupby("shift")["machine_failure"]
            .mean()
            .mul(100)
            .reset_index(name="Failure Rate (%)")
        )
        data["shift_order"] = pd.Categorical(
            data["shift"], categories=["Morning", "Evening", "Night"], ordered=True
        )
        data = data.sort_values("shift_order").drop(columns="shift_order")
        fig = styled_bar_chart(
            data,
            "shift",
            "Failure Rate (%)",
            "Failure Rate by Shift",
            "Failure Rate (%)",
            ["#475569", "#38BDF8", "#64748B"]
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        rate = data.set_index("shift")["Failure Rate (%)"]
        st.markdown(
            f"""
            <div class="panel">
                <div class="insight-badge badge-blue">SHIFT ANALYSIS</div>
                <div class="panel-title">Highest observed exposure</div>
                <div class="panel-copy">
                    <strong style="color:#F8FAFC">{rate.idxmax()}</strong> records the highest failure rate at
                    <strong style="color:#7DD3FC">{rate.max():.2f}%</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

with tab_tool:
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        grouped = (
            df.groupby("failure_label")["tool_wear_min"]
            .median()
            .reindex(["No Failure", "Failure"])
        )
        fig, ax = plt.subplots(figsize=(5.8, 3.15))
        bars = ax.bar(grouped.index, grouped.values, color=["#475569", "#F87171"], width=0.58)
        for bar, value in zip(bars, grouped.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(grouped.values) * 0.03,
                f"{value:.0f} min",
                ha="center",
                va="bottom",
                color="#E2E8F0",
                fontsize=9,
                fontweight="bold"
            )
        ax.set_title("Median Tool Wear by Machine Status", color="#F8FAFC", fontsize=11, fontweight="bold", loc="left", pad=12)
        ax.set_ylabel("Tool Wear (minutes)", color="#94A3B8", fontsize=8)
        ax.tick_params(colors="#94A3B8", labelsize=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#334155")
        ax.grid(axis="y", alpha=0.13, color="#64748B")
        ax.set_axisbelow(True)
        ax.set_facecolor("#111827")
        fig.patch.set_facecolor("#111827")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        delta = grouped["Failure"] - grouped["No Failure"]
        st.markdown(
            f"""
            <div class="panel">
                <div class="insight-badge badge-red">WEAR SIGNAL</div>
                <div class="panel-title">Tool wear separates failures</div>
                <div class="panel-copy">
                    Failed machines show a median tool-wear level approximately
                    <strong style="color:#FCA5A5">{delta:.0f} minutes higher</strong>
                    than non-failures in this dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

with tab_product:
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        data = (
            df.groupby("product_type")["machine_failure"]
            .mean()
            .mul(100)
            .reset_index(name="Failure Rate (%)")
            .sort_values("Failure Rate (%)", ascending=False)
        )
        fig = styled_bar_chart(
            data,
            "product_type",
            "Failure Rate (%)",
            "Failure Rate by Product Type",
            "Failure Rate (%)",
            ["#F87171", "#38BDF8", "#64748B"]
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        top = data.iloc[0]
        low = data.iloc[-1]
        st.markdown(
            f"""
            <div class="panel">
                <div class="insight-badge badge-red">PRODUCT EXPOSURE</div>
                <div class="panel-title">Highest-risk product type</div>
                <div class="panel-copy">
                    <strong style="color:#F8FAFC">{top['product_type']}</strong> has the highest failure rate at
                    <strong style="color:#FCA5A5">{top['Failure Rate (%)']:.2f}%</strong>,
                    versus {low['Failure Rate (%)']:.2f}% for the lowest-risk type.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

with tab_operator:
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        data = (
            df.groupby("operator_experience")["machine_failure"]
            .mean()
            .mul(100)
            .reset_index(name="Failure Rate (%)")
            .sort_values("Failure Rate (%)", ascending=False)
        )
        fig = styled_bar_chart(
            data,
            "operator_experience",
            "Failure Rate (%)",
            "Failure Rate by Operator Experience",
            "Failure Rate (%)",
            ["#F59E0B", "#64748B", "#38BDF8"]
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        top = data.iloc[0]
        st.markdown(
            f"""
            <div class="panel">
                <div class="insight-badge badge-amber">WORKFORCE SIGNAL</div>
                <div class="panel-title">Experience is one factor</div>
                <div class="panel-copy">
                    <strong style="color:#F8FAFC">{top['operator_experience']}</strong> operators show the highest
                    observed failure rate at <strong style="color:#FCD34D">{top['Failure Rate (%)']:.2f}%</strong>.
                    The result should be interpreted alongside machine conditions rather than as a standalone causal effect.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

with tab_temp:
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        temp = df.groupby("failure_label")["temp_diff"].median().reindex(["No Failure", "Failure"])
        fig, ax = plt.subplots(figsize=(5.8, 3.15))
        bars = ax.bar(temp.index, temp.values, color=["#475569", "#38BDF8"], width=0.58)
        for bar, value in zip(bars, temp.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(temp.values) * 0.02,
                f"{value:.2f} K",
                ha="center",
                va="bottom",
                color="#E2E8F0",
                fontsize=9,
                fontweight="bold"
            )
        ax.set_title("Median Process–Air Temperature Difference", color="#F8FAFC", fontsize=11, fontweight="bold", loc="left", pad=12)
        ax.set_ylabel("Temperature Difference (K)", color="#94A3B8", fontsize=8)
        ax.tick_params(colors="#94A3B8", labelsize=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#334155")
        ax.grid(axis="y", alpha=0.13, color="#64748B")
        ax.set_axisbelow(True)
        ax.set_facecolor("#111827")
        fig.patch.set_facecolor("#111827")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="insight-badge badge-blue">THERMAL SIGNAL</div>
                <div class="panel-title">Temperature separation matters</div>
                <div class="panel-copy">
                    Failure observations show a lower median process–air temperature difference,
                    consistent with the project's heat-dissipation interpretation.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

with tab_age:
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        category_order = ["New", "Mid-life", "Aging", "Critical"]
        data = (
            df.groupby("machine_age_category", observed=True)["machine_failure"]
            .mean()
            .mul(100)
            .reindex(category_order)
            .reset_index(name="Failure Rate (%)")
            .dropna()
        )
        fig = styled_bar_chart(
            data,
            "machine_age_category",
            "Failure Rate (%)",
            "Failure Rate by Machine Age Category",
            "Failure Rate (%)",
            ["#475569", "#FBBF24", "#F59E0B", "#F87171"],
            figsize=(6.5, 3.15)
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        critical_rate = float(data.loc[data["machine_age_category"] == "Critical", "Failure Rate (%)"].iloc[0]) if "Critical" in data["machine_age_category"].values else np.nan
        noncritical = data.loc[data["machine_age_category"] != "Critical", "Failure Rate (%)"]
        multiplier = critical_rate / noncritical.mean() if len(noncritical) and noncritical.mean() else np.nan
        st.markdown(
            f"""
            <div class="panel">
                <div class="insight-badge badge-red">AGE / WEAR RISK</div>
                <div class="panel-title">Critical state needs attention</div>
                <div class="panel-copy">
                    Critical machines show a failure rate of
                    <strong style="color:#FCA5A5">{critical_rate:.2f}%</strong>.
                    """
            + (f"That is roughly <strong style='color:#FCA5A5'>{multiplier:.1f}×</strong> the average of the other age categories." if np.isfinite(multiplier) else "")
            + """
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----------------------------------------
# ML Explainability + Predictor
# ----------------------------------------
st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-kicker">03 · PREDICTIVE MAINTENANCE</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Failure Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="section-description">Adjust machine conditions, estimate failure probability, and inspect the factors influencing the prediction.</div>', unsafe_allow_html=True)

input_col, result_col = st.columns([1.05, 1], gap="large")

with input_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Machine Parameters</div>', unsafe_allow_html=True)

    air_temp = st.slider(
        "Air Temperature (K)",
        min_value=295.0,
        max_value=305.0,
        value=300.0,
        step=0.1
    )

    process_temp = st.slider(
        "Process Temperature (K)",
        min_value=305.0,
        max_value=315.0,
        value=310.0,
        step=0.1
    )

    rot_speed = st.slider(
        "Rotational Speed (RPM)",
        min_value=1168,
        max_value=2886,
        value=1500,
        step=10
    )

    torque = st.slider(
        "Torque (Nm)",
        min_value=3.8,
        max_value=76.6,
        value=40.0,
        step=0.1
    )

    tool_wear = st.slider(
        "Tool Wear (minutes)",
        min_value=0,
        max_value=253,
        value=100,
        step=1
    )

    product_type = st.selectbox(
        "Product Type",
        options=["L", "M", "H"]
    )

    shift = st.selectbox(
        "Shift",
        options=["Morning", "Evening", "Night"]
    )

    operator_exp = st.selectbox(
        "Operator Experience",
        options=["Junior", "Senior", "Expert"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------
# Feature Engineering for Prediction
# ----------------------------------------
power_kw = (torque * rot_speed) / 9549
temp_diff = process_temp - air_temp

if tool_wear <= 100:
    wear_cat_num = 0
elif tool_wear <= 160:
    wear_cat_num = 1
elif tool_wear <= 200:
    wear_cat_num = 2
else:
    wear_cat_num = 3

wear_label = ["New", "Mid-life", "Aging", "Critical"][wear_cat_num]

input_dict = {
    "air_temp_k": air_temp,
    "process_temp_k": process_temp,
    "rotational_speed_rpm": rot_speed,
    "torque_nm": torque,
    "tool_wear_min": tool_wear,
    "power_kw": power_kw,
    "temp_diff": temp_diff,
    "tool_wear_category_num": wear_cat_num,
    "product_type_H": 1 if product_type == "H" else 0,
    "product_type_L": 1 if product_type == "L" else 0,
    "product_type_M": 1 if product_type == "M" else 0,
    "shift_Evening": 1 if shift == "Evening" else 0,
    "shift_Morning": 1 if shift == "Morning" else 0,
    "shift_Night": 1 if shift == "Night" else 0,
    "operator_experience_Expert": 1 if operator_exp == "Expert" else 0,
    "operator_experience_Junior": 1 if operator_exp == "Junior" else 0,
    "operator_experience_Senior": 1 if operator_exp == "Senior" else 0
}

input_df = pd.DataFrame([input_dict])[feature_names]
failure_prob = float(model.predict_proba(input_df)[0][1])
failure_pct = failure_prob * 100

with result_col:
    if failure_prob > 0.35:
        risk_level = "HIGH RISK"
        risk_class = "risk-high"
        risk_action = "Schedule immediate inspection within 24 hours."
        accent = "#F87171"
        progress_fill = "#EF4444"
    elif failure_prob > 0.15:
        risk_level = "MEDIUM RISK"
        risk_class = "risk-medium"
        risk_action = "Monitor closely and inspect within 72 hours."
        accent = "#FBBF24"
        progress_fill = "#F59E0B"
    else:
        risk_level = "LOW RISK"
        risk_class = "risk-low"
        risk_action = "Continue production with the next scheduled maintenance."
        accent = "#4ADE80"
        progress_fill = "#22C55E"

    progress_width = max(2, min(100, failure_pct))

    st.markdown(
        f"""
        <div class="risk-card">
            <div class="risk-caption">Current Failure Probability</div>
            <div class="risk-probability">{failure_pct:.1f}%</div>
            <div style="height:10px"></div>
            <div style="height:9px;background:#1F2937;border-radius:99px;overflow:hidden;">
                <div style="height:100%;width:{progress_width}%;background:{progress_fill};border-radius:99px;"></div>
            </div>
            <div class="risk-level {risk_class}">{risk_level}</div>
            <div style="height:14px"></div>
            <div class="recommendation">
                <strong style="color:#F8FAFC">Recommended action</strong><br>
                {risk_action}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(
            f"""
            <div class="derived">
                <div class="derived-label">Mechanical Power</div>
                <div class="derived-value">{power_kw:.2f} kW</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with d2:
        st.markdown(
            f"""
            <div class="derived">
                <div class="derived-label">Thermal Gap</div>
                <div class="derived-value">{temp_diff:.2f} K</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with d3:
        st.markdown(
            f"""
            <div class="derived">
                <div class="derived-label">Machine State</div>
                <div class="derived-value">{wear_label}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----------------------------------------
# Per-prediction SHAP Explanation
# ----------------------------------------
st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
shap_col, global_col = st.columns([1.45, 1], gap="large")

with shap_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Why did FactoryLens make this prediction?</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">SHAP contribution for the current machine profile. Positive values push the prediction toward failure; negative values push it away.</div>', unsafe_allow_html=True)

    try:
        explainer = build_shap_explainer(model)
        shap_values = explainer.shap_values(input_df)

        if isinstance(shap_values, list):
            local_values = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])[0]
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                local_values = arr[0, :, 1]
            else:
                local_values = arr[0]

        explanation = (
            pd.DataFrame({
                "Feature": feature_names,
                "Contribution": local_values,
                "Value": input_df.iloc[0].values
            })
            .assign(AbsContribution=lambda x: np.abs(x["Contribution"]))
            .sort_values("AbsContribution", ascending=False)
            .head(7)
            .sort_values("Contribution")
        )

        fig, ax = plt.subplots(figsize=(6.8, 3.9))
        bars = ax.barh(
            explanation["Feature"],
            explanation["Contribution"],
            color=["#F87171" if v > 0 else "#38BDF8" for v in explanation["Contribution"]]
        )

        limit = max(np.abs(explanation["Contribution"]).max() * 1.25, 0.01)
        ax.set_xlim(-limit, limit)
        ax.axvline(0, color="#64748B", linewidth=1)
        ax.tick_params(axis="y", colors="#CBD5E1", labelsize=8)
        ax.tick_params(axis="x", colors="#64748B", labelsize=8)
        ax.set_xlabel("SHAP contribution", color="#94A3B8", fontsize=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#334155")
        ax.grid(axis="x", alpha=0.12, color="#64748B")
        ax.set_facecolor("#111827")
        fig.patch.set_facecolor("#111827")
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown(
            "<div class='muted'>Red bars increase modelled failure risk. Blue bars decrease it for this specific prediction.</div>",
            unsafe_allow_html=True
        )
    except Exception as shap_error:
        st.warning(f"SHAP explanation could not be rendered for this model build: {shap_error}")

    st.markdown('</div>', unsafe_allow_html=True)

with global_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Global Failure Drivers</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Mean absolute SHAP value across a compact sample of the dataset.</div>', unsafe_allow_html=True)

    try:
        sample_size = min(1000, len(df))

        global_input = df[feature_names].copy()
        sample = global_input.sample(sample_size, random_state=42)
        global_importance = calculate_global_shap(
            model,
            tuple(feature_names),
            sample
        ).head(7).sort_values("Mean SHAP Value")

        fig, ax = plt.subplots(figsize=(5.5, 3.9))
        bars = ax.barh(
            global_importance["Feature"],
            global_importance["Mean SHAP Value"],
            color="#38BDF8"
        )
        ax.tick_params(axis="y", colors="#CBD5E1", labelsize=8)
        ax.tick_params(axis="x", colors="#64748B", labelsize=8)
        ax.set_xlabel("Mean |SHAP|", color="#94A3B8", fontsize=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#334155")
        ax.grid(axis="x", alpha=0.12, color="#64748B")
        ax.set_axisbelow(True)
        ax.set_facecolor("#111827")
        fig.patch.set_facecolor("#111827")
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        top_driver = global_importance.iloc[-1]["Feature"]
        st.markdown(
            f"<div class='muted'>Top global driver in this model: <strong style='color:#7DD3FC'>{top_driver}</strong>.</div>",
            unsafe_allow_html=True
        )
    except Exception as shap_error:
        st.warning(f"Global SHAP profile could not be rendered: {shap_error}")

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------
# Model / Project Notes
# ----------------------------------------
st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
with st.expander("Model and project notes"):
    st.markdown(
        """
        **FactoryLens uses a saved Random Forest classifier trained on 17 engineered / encoded features.**

        The predictive layer uses three domain-informed engineered variables:
        `power_kw = torque × RPM / 9549`, `temp_diff = process temperature − air temperature`,
        and ordered tool-wear state encoding.

        The dashboard is intentionally decision-oriented: analytical patterns are shown beside concise business interpretations,
        while the predictor exposes both a probability and an explainability layer.

        The underlying project uses the AI4I 2020 predictive maintenance dataset with simulated operational enrichment fields for
        shift, operator experience, machine age category, and maintenance status.
        """
    )

st.caption("FactoryLens · Manufacturing Operations Analytics · Built for portfolio and decision-support demonstration")
