"""
AI Threat Detection & Smart Surveillance System
Professional enterprise-grade dashboard built with Streamlit, Plotly and scikit-learn.
"""

import random
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Threat Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---- main background ---- */
    .stApp { background-color: #0D1117; }

    /* ---- metric cards ---- */
    .metric-card {
        background: linear-gradient(135deg, #161B22 0%, #1C2128 100%);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card h3 { color: #8B949E; font-size: 13px; margin-bottom: 8px; letter-spacing: 1px; }
    .metric-card h2 { color: #C9D1D9; font-size: 28px; font-weight: 700; margin: 0; }

    /* ---- status badges ---- */
    .status-safe     { background:#0D4429; color:#3FB950; border:1px solid #238636; border-radius:20px; padding:6px 18px; font-weight:700; }
    .status-warning  { background:#3D2B00; color:#D29922; border:1px solid #BB8009; border-radius:20px; padding:6px 18px; font-weight:700; }
    .status-critical { background:#3D0B0B; color:#F85149; border:1px solid #DA3633; border-radius:20px; padding:6px 18px; font-weight:700; }

    /* ---- section headers ---- */
    .section-header {
        background: linear-gradient(90deg, #E63946 0%, #C1121F 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 16px;
        letter-spacing: 0.5px;
    }

    /* ---- sidebar ---- */
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }

    /* ---- hide Streamlit branding ---- */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Synthetic dataset & model (cached) ───────────────────────────────────────
@st.cache_resource(show_spinner="Training AI model…")
def train_model():
    """Generate synthetic surveillance data and train a RandomForestClassifier."""
    np.random.seed(42)
    n = 2000

    crowd_density   = np.random.randint(0, 101, n)
    motion_intensity = np.random.randint(0, 101, n)
    time_of_day     = np.random.randint(0, 24, n)
    location_risk   = np.random.choice([0, 1, 2], n)   # 0=Low 1=Med 2=High
    unauth_access   = np.random.choice([0, 1], n)

    # Rule-based threat labelling
    threat_score = (
        crowd_density * 0.30
        + motion_intensity * 0.25
        + (time_of_day < 6).astype(int) * 20
        + location_risk * 15
        + unauth_access * 25
    )
    threat_level = np.where(threat_score < 35, 0,
                   np.where(threat_score < 60, 1, 2))   # 0=Low 1=Med 2=High

    df = pd.DataFrame({
        "crowd_density":    crowd_density,
        "motion_intensity": motion_intensity,
        "time_of_day":      time_of_day,
        "location_risk":    location_risk,
        "unauth_access":    unauth_access,
        "threat_level":     threat_level,
    })

    X = df.drop("threat_level", axis=1)
    y = df["threat_level"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    return clf, X_test, y_test, df


# ── Helper utilities ──────────────────────────────────────────────────────────
THREAT_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
THREAT_COLORS = {0: "#3FB950", 1: "#D29922", 2: "#F85149"}
STATUS_CLASS  = {0: "status-safe", 1: "status-warning", 2: "status-critical"}
STATUS_EMOJI  = {0: "✅", 1: "⚠️", 2: "🚨"}


def predict(model, crowd, motion, hour, loc_risk, unauth):
    loc_map  = {"Low": 0, "Medium": 1, "High": 2}
    auth_map = {"No": 0, "Yes": 1}
    feat = pd.DataFrame([[crowd, motion, hour, loc_map[loc_risk], auth_map[unauth]]],
                        columns=["crowd_density", "motion_intensity", "time_of_day",
                                 "location_risk", "unauth_access"])
    pred  = model.predict(feat)[0]
    proba = model.predict_proba(feat)[0]
    return int(pred), proba


def risk_gauge(score: float, title: str = "Risk Score") -> go.Figure:
    color = "#3FB950" if score < 35 else "#D29922" if score < 65 else "#F85149"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"color": "#C9D1D9", "size": 16}},
        number={"suffix": "%", "font": {"color": color, "size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8B949E"},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#161B22",
            "bordercolor": "#30363D",
            "steps": [
                {"range": [0, 35],  "color": "#0D4429"},
                {"range": [35, 65], "color": "#3D2B00"},
                {"range": [65, 100],"color": "#3D0B0B"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": score},
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0D1117", font_color="#C9D1D9",
        height=260, margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding:10px 0 20px;'>"
        "<span style='font-size:40px'>🛡️</span><br>"
        "<span style='color:#E63946;font-size:18px;font-weight:700;'>AI THREAT DETECTION</span><br>"
        "<span style='color:#8B949E;font-size:11px;'>Smart Surveillance System</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#30363D;margin:0 0 16px;'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📊 Dashboard Overview", "🔍 Threat Prediction", "📈 Risk Analytics", "🤖 AI Model Insights"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#30363D;margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#8B949E;font-size:11px;text-align:center;'>"
        "Powered by RandomForest AI<br>© 2025 AI Surveillance Corp"
        "</div>",
        unsafe_allow_html=True,
    )

# Load model once
model, X_test, y_test, df = train_model()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – DASHBOARD OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard Overview":
    st.markdown(
        "<h1 style='color:#E63946;margin-bottom:4px;'>🛡️ AI Threat Detection Dashboard</h1>"
        "<p style='color:#8B949E;margin-top:0;'>Real-time surveillance monitoring & threat assessment</p>",
        unsafe_allow_html=True,
    )

    # ── Simulated live metrics ────────────────────────────────────────────────
    if "sim_data" not in st.session_state:
        st.session_state.sim_data = {
            "crowd": random.randint(20, 80),
            "motion": random.randint(10, 90),
            "hour": time.localtime().tm_hour,
            "loc": random.choice(["Low", "Medium", "High"]),
            "unauth": random.choice(["No", "No", "No", "Yes"]),
        }

    col_refresh = st.columns([6, 1])
    with col_refresh[1]:
        if st.button("🔄 Refresh", width="stretch"):
            st.session_state.sim_data = {
                "crowd":  random.randint(10, 95),
                "motion": random.randint(5, 95),
                "hour":   time.localtime().tm_hour,
                "loc":    random.choice(["Low", "Medium", "High"]),
                "unauth": random.choice(["No", "No", "No", "Yes"]),
            }

    sd = st.session_state.sim_data
    threat_idx, proba = predict(model, sd["crowd"], sd["motion"], sd["hour"], sd["loc"], sd["unauth"])
    risk_score = round(float(proba[threat_idx]) * 100, 1)
    threat_label = THREAT_LABELS[threat_idx]
    status_class  = STATUS_CLASS[threat_idx]
    status_emoji  = STATUS_EMOJI[threat_idx]

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='metric-card'><h3>THREAT LEVEL</h3>"
            f"<h2 style='color:{THREAT_COLORS[threat_idx]};'>{threat_label}</h2></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='metric-card'><h3>RISK SCORE</h3>"
            f"<h2 style='color:{THREAT_COLORS[threat_idx]};'>{risk_score}%</h2></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='metric-card'><h3>STATUS</h3>"
            f"<h2><span class='{status_class}'>{status_emoji} {threat_label}</span></h2></div>",
            unsafe_allow_html=True,
        )
    with c4:
        cam_status = "🟢 ONLINE" if threat_idx < 2 else "🔴 ALERT"
        st.markdown(
            f"<div class='metric-card'><h3>CAMERAS</h3>"
            f"<h2 style='font-size:20px;'>{cam_status}</h2></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge + Current parameters ────────────────────────────────────────────
    g1, g2 = st.columns([1, 1])
    with g1:
        st.markdown("<div class='section-header'>📡 Live Risk Gauge</div>", unsafe_allow_html=True)
        st.plotly_chart(risk_gauge(risk_score), width="stretch")

    with g2:
        st.markdown("<div class='section-header'>📋 Current Sensor Readings</div>", unsafe_allow_html=True)
        sensor_df = pd.DataFrame({
            "Parameter":  ["Crowd Density", "Motion Intensity", "Time of Day", "Location Risk", "Unauth. Access"],
            "Value":      [f"{sd['crowd']}%", f"{sd['motion']}%", f"{sd['hour']:02d}:00",
                           sd["loc"], sd["unauth"]],
        })
        st.dataframe(sensor_df, width="stretch", hide_index=True)

        alert_color = THREAT_COLORS[threat_idx]
        st.markdown(
            f"<div style='background:{alert_color}22;border-left:4px solid {alert_color};"
            f"padding:12px;border-radius:6px;margin-top:12px;color:{alert_color};font-weight:600;'>"
            f"{status_emoji} Threat Assessment: <strong>{threat_label}</strong> — "
            f"Confidence {risk_score}%</div>",
            unsafe_allow_html=True,
        )

    # ── Simulated real-time timeline ──────────────────────────────────────────
    st.markdown("<div class='section-header'>📉 Real-Time Monitoring Simulation</div>", unsafe_allow_html=True)
    ts_len = 60
    np.random.seed(int(time.time()) % 1000)
    timestamps = [f"{i:02d}:00" for i in range(ts_len)]
    sim_risk = np.clip(
        np.cumsum(np.random.randn(ts_len) * 3) + 40 + threat_idx * 15, 0, 100
    )

    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=timestamps, y=sim_risk,
        mode="lines", fill="tozeroy",
        line=dict(color=THREAT_COLORS[threat_idx], width=2),
        fillcolor=f"rgba({int(THREAT_COLORS[threat_idx][1:3],16)},{int(THREAT_COLORS[threat_idx][3:5],16)},{int(THREAT_COLORS[threat_idx][5:7],16)},0.2)",
        name="Risk Level",
    ))
    fig_timeline.add_hline(y=35, line_dash="dash", line_color="#3FB950",  annotation_text="Safe",    annotation_font_color="#3FB950")
    fig_timeline.add_hline(y=65, line_dash="dash", line_color="#D29922",  annotation_text="Warning", annotation_font_color="#D29922")
    fig_timeline.update_layout(
        paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        font_color="#C9D1D9", height=280,
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(title="Risk %", range=[0, 100], gridcolor="#21262D"),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(bgcolor="#161B22"),
    )
    st.plotly_chart(fig_timeline, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – THREAT PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Threat Prediction":
    st.markdown(
        "<h1 style='color:#E63946;margin-bottom:4px;'>🔍 Threat Prediction Engine</h1>"
        "<p style='color:#8B949E;'>Configure input parameters and run AI-powered threat analysis</p>",
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        st.markdown("<div class='section-header'>⚙️ Input Parameters</div>", unsafe_allow_html=True)
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            crowd    = st.slider("👥 Crowd Density", 0, 100, 50, help="Estimated crowd density (0–100)")
        with r1c2:
            motion   = st.slider("🏃 Motion Intensity", 0, 100, 40, help="Detected motion level (0–100)")
        with r1c3:
            hour     = st.slider("🕐 Time of Day (hour)", 0, 23, 12, help="Hour in 24h format")

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            loc_risk = st.selectbox("📍 Location Risk Level", ["Low", "Medium", "High"])
        with r2c2:
            unauth   = st.selectbox("🔒 Unauthorized Access Detected", ["No", "Yes"])

        submitted = st.form_submit_button("🚀 Analyze Threat", width="stretch")

    if submitted:
        threat_idx, proba = predict(model, crowd, motion, hour, loc_risk, unauth)
        risk_score = round(float(proba[threat_idx]) * 100, 1)
        color = THREAT_COLORS[threat_idx]
        emoji = STATUS_EMOJI[threat_idx]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:{color}22;border:2px solid {color};border-radius:12px;"
            f"padding:20px;text-align:center;'>"
            f"<div style='font-size:48px;'>{emoji}</div>"
            f"<div style='color:{color};font-size:32px;font-weight:700;'>{THREAT_LABELS[threat_idx]} THREAT</div>"
            f"<div style='color:#C9D1D9;font-size:18px;margin-top:8px;'>Confidence: {risk_score}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        pr1, pr2 = st.columns(2)

        with pr1:
            st.markdown("<div class='section-header'>📊 Probability Distribution</div>", unsafe_allow_html=True)
            prob_df = pd.DataFrame({
                "Threat Level": ["LOW", "MEDIUM", "HIGH"],
                "Probability":  [round(p * 100, 1) for p in proba],
                "Color":        ["#3FB950", "#D29922", "#F85149"],
            })
            fig_bar = px.bar(
                prob_df, x="Threat Level", y="Probability",
                color="Threat Level",
                color_discrete_map={"LOW": "#3FB950", "MEDIUM": "#D29922", "HIGH": "#F85149"},
                text="Probability",
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar.update_layout(
                paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
                font_color="#C9D1D9", height=320, showlegend=False,
                yaxis=dict(title="Probability (%)", gridcolor="#21262D"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_bar, width="stretch")

        with pr2:
            st.markdown("<div class='section-header'>🎯 Risk Gauge</div>", unsafe_allow_html=True)
            st.plotly_chart(risk_gauge(risk_score), width="stretch")

        # Recommendations
        st.markdown("<div class='section-header'>💡 AI Recommendations</div>", unsafe_allow_html=True)
        recs = {
            0: ["✅ Continue standard monitoring protocols",
                "✅ Maintain regular patrol schedules",
                "✅ Keep all surveillance systems active"],
            1: ["⚠️ Increase camera monitoring frequency",
                "⚠️ Deploy additional security personnel to the area",
                "⚠️ Issue advisory alert to nearby units"],
            2: ["🚨 Immediate security response required",
                "🚨 Activate emergency protocols and lock-down procedures",
                "🚨 Notify law enforcement and emergency services"],
        }
        for rec in recs[threat_idx]:
            st.markdown(
                f"<div style='background:#161B22;border-left:3px solid {color};"
                f"padding:10px 14px;border-radius:4px;margin:6px 0;color:#C9D1D9;'>{rec}</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – RISK ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Risk Analytics":
    st.markdown(
        "<h1 style='color:#E63946;margin-bottom:4px;'>📈 Risk Analytics</h1>"
        "<p style='color:#8B949E;'>Statistical analysis of threat patterns across the surveillance dataset</p>",
        unsafe_allow_html=True,
    )

    label_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    df_plot = df.copy()
    df_plot["threat_label"] = df_plot["threat_level"].map(label_map)
    df_plot["location_label"] = df_plot["location_risk"].map({0: "Low", 1: "Medium", 2: "High"})

    # ── Row 1: distribution + heatmap ────────────────────────────────────────
    a1, a2 = st.columns(2)

    with a1:
        st.markdown("<div class='section-header'>📊 Threat Level Distribution</div>", unsafe_allow_html=True)
        dist = df_plot["threat_label"].value_counts().reset_index()
        dist.columns = ["Threat Level", "Count"]
        fig_pie = px.pie(
            dist, names="Threat Level", values="Count",
            color="Threat Level",
            color_discrete_map={"LOW": "#3FB950", "MEDIUM": "#D29922", "HIGH": "#F85149"},
            hole=0.45,
        )
        fig_pie.update_layout(
            paper_bgcolor="#0D1117", font_color="#C9D1D9", height=340,
            legend=dict(bgcolor="#161B22"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_pie, width="stretch")

    with a2:
        st.markdown("<div class='section-header'>🌡️ Crowd Density vs Motion Intensity</div>", unsafe_allow_html=True)
        fig_scatter = px.scatter(
            df_plot.sample(500, random_state=42),
            x="crowd_density", y="motion_intensity",
            color="threat_label",
            color_discrete_map={"LOW": "#3FB950", "MEDIUM": "#D29922", "HIGH": "#F85149"},
            opacity=0.7,
            labels={"crowd_density": "Crowd Density", "motion_intensity": "Motion Intensity"},
        )
        fig_scatter.update_layout(
            paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
            font_color="#C9D1D9", height=340,
            xaxis=dict(gridcolor="#21262D"), yaxis=dict(gridcolor="#21262D"),
            legend=dict(bgcolor="#161B22"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_scatter, width="stretch")

    # ── Row 2: hourly pattern + location risk ─────────────────────────────────
    b1, b2 = st.columns(2)

    with b1:
        st.markdown("<div class='section-header'>🕐 Hourly Threat Frequency</div>", unsafe_allow_html=True)
        hourly = df_plot.groupby(["time_of_day", "threat_label"]).size().reset_index(name="count")
        fig_hourly = px.bar(
            hourly, x="time_of_day", y="count", color="threat_label",
            color_discrete_map={"LOW": "#3FB950", "MEDIUM": "#D29922", "HIGH": "#F85149"},
            labels={"time_of_day": "Hour of Day", "count": "Incident Count"},
        )
        fig_hourly.update_layout(
            paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
            font_color="#C9D1D9", height=320,
            xaxis=dict(gridcolor="#21262D"), yaxis=dict(gridcolor="#21262D"),
            legend=dict(bgcolor="#161B22"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_hourly, width="stretch")

    with b2:
        st.markdown("<div class='section-header'>📍 Threat Level by Location Risk</div>", unsafe_allow_html=True)
        loc_threat = df_plot.groupby(["location_label", "threat_label"]).size().reset_index(name="count")
        fig_loc = px.bar(
            loc_threat, x="location_label", y="count", color="threat_label",
            barmode="group",
            color_discrete_map={"LOW": "#3FB950", "MEDIUM": "#D29922", "HIGH": "#F85149"},
            labels={"location_label": "Location Risk", "count": "Count"},
            category_orders={"location_label": ["Low", "Medium", "High"]},
        )
        fig_loc.update_layout(
            paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
            font_color="#C9D1D9", height=320,
            xaxis=dict(gridcolor="#21262D"), yaxis=dict(gridcolor="#21262D"),
            legend=dict(bgcolor="#161B22"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_loc, width="stretch")

    # ── Row 3: summary statistics ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>📋 Dataset Summary Statistics</div>", unsafe_allow_html=True)
    summary = df[["crowd_density", "motion_intensity", "time_of_day"]].describe().round(2)
    st.dataframe(summary, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – AI MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Model Insights":
    st.markdown(
        "<h1 style='color:#E63946;margin-bottom:4px;'>🤖 AI Model Insights</h1>"
        "<p style='color:#8B949E;'>RandomForestClassifier — performance metrics & interpretability</p>",
        unsafe_allow_html=True,
    )

    y_pred = model.predict(X_test)
    acc    = round((y_pred == y_test).mean() * 100, 2)

    # ── Top KPIs ──────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='metric-card'><h3>ACCURACY</h3><h2 style='color:#3FB950;'>{acc}%</h2></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card'><h3>ESTIMATORS</h3><h2 style='color:#58A6FF;'>150</h2></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-card'><h3>MAX DEPTH</h3><h2 style='color:#58A6FF;'>8</h2></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='metric-card'><h3>TRAINING SIZE</h3><h2 style='color:#58A6FF;'>1,600</h2></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature importance + confusion matrix ─────────────────────────────────
    i1, i2 = st.columns(2)

    with i1:
        st.markdown("<div class='section-header'>🔑 Feature Importance</div>", unsafe_allow_html=True)
        feat_names = ["Crowd Density", "Motion Intensity", "Time of Day", "Location Risk", "Unauth. Access"]
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
        imp_df = imp_df.sort_values("Importance", ascending=True)

        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#21262D", "#E63946"],
            text=imp_df["Importance"].round(3),
        )
        fig_imp.update_traces(textposition="outside")
        fig_imp.update_layout(
            paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
            font_color="#C9D1D9", height=360, coloraxis_showscale=False,
            xaxis=dict(gridcolor="#21262D"), yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_imp, width="stretch")

    with i2:
        st.markdown("<div class='section-header'>🔲 Confusion Matrix</div>", unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            x=["LOW", "MEDIUM", "HIGH"],
            y=["LOW", "MEDIUM", "HIGH"],
            color_continuous_scale=["#0D1117", "#E63946"],
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )
        fig_cm.update_layout(
            paper_bgcolor="#0D1117", font_color="#C9D1D9", height=360,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_cm, width="stretch")

    # ── Classification report ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📊 Classification Report</div>", unsafe_allow_html=True)
    report_dict = classification_report(
        y_test, y_pred,
        target_names=["LOW", "MEDIUM", "HIGH"],
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).T.round(3)
    report_df = report_df.drop(["accuracy"], errors="ignore")
    st.dataframe(report_df, width="stretch")

    # ── Model architecture ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🏗️ Model Architecture</div>", unsafe_allow_html=True)
    arch_cols = st.columns(3)
    arch_info = [
        ("Algorithm",   "Random Forest Classifier"),
        ("Input Features", "5 (crowd, motion, time, location, unauth)"),
        ("Output Classes", "3 (Low / Medium / High)"),
        ("Ensemble Size",  "150 decision trees"),
        ("Max Tree Depth", "8 levels"),
        ("Train/Test Split","80% / 20%"),
    ]
    for idx, (k, v) in enumerate(arch_info):
        with arch_cols[idx % 3]:
            st.markdown(
                f"<div style='background:#161B22;border:1px solid #30363D;border-radius:8px;"
                f"padding:14px;margin-bottom:12px;'>"
                f"<div style='color:#8B949E;font-size:11px;letter-spacing:1px;'>{k.upper()}</div>"
                f"<div style='color:#C9D1D9;font-weight:600;margin-top:4px;'>{v}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
