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
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---- main background ---- */
    .stApp { background-color: #F8F9FA; }

    /* ---- metric cards ---- */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #DADCE0;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(60,64,67,0.1);
    }
    .metric-card h3 { color: #5F6368; font-size: 12px; margin-bottom: 8px; letter-spacing: 1px; text-transform: uppercase; }
    .metric-card h2 { color: #202124; font-size: 28px; font-weight: 600; margin: 0; }

    /* ---- status badges ---- */
    .status-safe     { background:#E6F4EA; color:#137333; border:1px solid #A8D5B5; border-radius:4px; padding:4px 12px; font-weight:600; font-size:13px; }
    .status-warning  { background:#FEF7E0; color:#B06000; border:1px solid #F9C15E; border-radius:4px; padding:4px 12px; font-weight:600; font-size:13px; }
    .status-critical { background:#FCE8E6; color:#C5221F; border:1px solid #F4B8B4; border-radius:4px; padding:4px 12px; font-weight:600; font-size:13px; }

    /* ---- section headers ---- */
    .section-header {
        color: #202124;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid #DADCE0;
        letter-spacing: 0.2px;
    }

    /* ---- sidebar ---- */
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #DADCE0; }

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
THREAT_COLORS = {0: "#34A853", 1: "#F9AB00", 2: "#EA4335"}
STATUS_CLASS  = {0: "status-safe", 1: "status-warning", 2: "status-critical"}


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
    color = "#34A853" if score < 35 else "#F9AB00" if score < 65 else "#EA4335"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"color": "#5F6368", "size": 14}},
        number={"suffix": "%", "font": {"color": color, "size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#5F6368"},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#FFFFFF",
            "bordercolor": "#DADCE0",
            "steps": [
                {"range": [0, 35],  "color": "#E6F4EA"},
                {"range": [35, 65], "color": "#FEF7E0"},
                {"range": [65, 100],"color": "#FCE8E6"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": score},
        },
    ))
    fig.update_layout(
        paper_bgcolor="#FFFFFF", font_color="#202124",
        height=260, margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:16px 0 20px;'>"
        "<div style='color:#1A73E8;font-size:16px;font-weight:700;letter-spacing:0.3px;'>AI THREAT DETECTION</div>"
        "<div style='color:#5F6368;font-size:13px;margin-top:2px;'>Smart Surveillance System</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#DADCE0;margin:0 0 16px;'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Dashboard Overview", "Threat Prediction", "Risk Analytics", "AI Model Insights"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#DADCE0;margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#5F6368;font-size:12px;'>"
        "Powered by RandomForest AI<br>© 2025 AI Surveillance Corp"
        "</div>",
        unsafe_allow_html=True,
    )

# Load model once
model, X_test, y_test, df = train_model()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – DASHBOARD OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard Overview":
    st.markdown(
        "<h1 style='color:#202124;font-size:24px;font-weight:600;margin-bottom:4px;'>AI Threat Detection Dashboard</h1>"
        "<p style='color:#5F6368;margin-top:0;font-size:14px;'>Real-time surveillance monitoring and threat assessment</p>",
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
        if st.button("Refresh", width="stretch"):
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
            f"<h2><span class='{status_class}'>{threat_label}</span></h2></div>",
            unsafe_allow_html=True,
        )
    with c4:
        cam_status = "ONLINE" if threat_idx < 2 else "ALERT"
        cam_color  = "#34A853" if threat_idx < 2 else "#EA4335"
        st.markdown(
            f"<div class='metric-card'><h3>CAMERAS</h3>"
            f"<h2 style='font-size:20px;color:{cam_color};'>{cam_status}</h2></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge + Current parameters ────────────────────────────────────────────
    g1, g2 = st.columns([1, 1])
    with g1:
        st.markdown("<div class='section-header'>Live Risk Gauge</div>", unsafe_allow_html=True)
        st.plotly_chart(risk_gauge(risk_score), width="stretch")

    with g2:
        st.markdown("<div class='section-header'>Current Sensor Readings</div>", unsafe_allow_html=True)
        sensor_df = pd.DataFrame({
            "Parameter":  ["Crowd Density", "Motion Intensity", "Time of Day", "Location Risk", "Unauth. Access"],
            "Value":      [f"{sd['crowd']}%", f"{sd['motion']}%", f"{sd['hour']:02d}:00",
                           sd["loc"], sd["unauth"]],
        })
        st.dataframe(sensor_df, width="stretch", hide_index=True)

        alert_color = THREAT_COLORS[threat_idx]
        st.markdown(
            f"<div style='background:{alert_color}18;border-left:3px solid {alert_color};"
            f"padding:10px 14px;border-radius:4px;margin-top:12px;color:{alert_color};font-weight:600;font-size:13px;'>"
            f"Threat Assessment: <strong>{threat_label}</strong> — "
            f"Confidence {risk_score}%</div>",
            unsafe_allow_html=True,
        )

    # ── Simulated real-time timeline ──────────────────────────────────────────
    st.markdown("<div class='section-header'>Real-Time Monitoring Simulation</div>", unsafe_allow_html=True)
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
    fig_timeline.add_hline(y=35, line_dash="dash", line_color="#34A853",  annotation_text="Safe",    annotation_font_color="#34A853")
    fig_timeline.add_hline(y=65, line_dash="dash", line_color="#F9AB00",  annotation_text="Warning", annotation_font_color="#F9AB00")
    fig_timeline.update_layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F8F9FA",
        font_color="#202124", height=280,
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(title="Risk %", range=[0, 100], gridcolor="#DADCE0"),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(bgcolor="#FFFFFF"),
    )
    st.plotly_chart(fig_timeline, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – THREAT PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Threat Prediction":
    st.markdown(
        "<h1 style='color:#202124;font-size:24px;font-weight:600;margin-bottom:4px;'>Threat Prediction Engine</h1>"
        "<p style='color:#5F6368;font-size:14px;'>Configure input parameters and run AI-powered threat analysis</p>",
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        st.markdown("<div class='section-header'>Input Parameters</div>", unsafe_allow_html=True)
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            crowd    = st.slider("Crowd Density", 0, 100, 50, help="Estimated crowd density (0–100)")
        with r1c2:
            motion   = st.slider("Motion Intensity", 0, 100, 40, help="Detected motion level (0–100)")
        with r1c3:
            hour     = st.slider("Time of Day (hour)", 0, 23, 12, help="Hour in 24h format")

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            loc_risk = st.selectbox("Location Risk Level", ["Low", "Medium", "High"])
        with r2c2:
            unauth   = st.selectbox("Unauthorized Access Detected", ["No", "Yes"])

        submitted = st.form_submit_button("Analyze Threat", width="stretch")

    if submitted:
        threat_idx, proba = predict(model, crowd, motion, hour, loc_risk, unauth)
        risk_score = round(float(proba[threat_idx]) * 100, 1)
        color = THREAT_COLORS[threat_idx]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:{color}12;border:1px solid {color}40;border-radius:8px;"
            f"padding:20px;text-align:center;'>"
            f"<div style='color:{color};font-size:22px;font-weight:700;letter-spacing:0.5px;'>{THREAT_LABELS[threat_idx]} THREAT</div>"
            f"<div style='color:#5F6368;font-size:14px;margin-top:6px;'>Confidence: {risk_score}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        pr1, pr2 = st.columns(2)

        with pr1:
            st.markdown("<div class='section-header'>Probability Distribution</div>", unsafe_allow_html=True)
            prob_df = pd.DataFrame({
                "Threat Level": ["LOW", "MEDIUM", "HIGH"],
                "Probability":  [round(p * 100, 1) for p in proba],
                "Color":        ["#34A853", "#F9AB00", "#EA4335"],
            })
            fig_bar = px.bar(
                prob_df, x="Threat Level", y="Probability",
                color="Threat Level",
                color_discrete_map={"LOW": "#34A853", "MEDIUM": "#F9AB00", "HIGH": "#EA4335"},
                text="Probability",
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar.update_layout(
                paper_bgcolor="#FFFFFF", plot_bgcolor="#F8F9FA",
                font_color="#202124", height=320, showlegend=False,
                yaxis=dict(title="Probability (%)", gridcolor="#DADCE0"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_bar, width="stretch")

        with pr2:
            st.markdown("<div class='section-header'>Risk Gauge</div>", unsafe_allow_html=True)
            st.plotly_chart(risk_gauge(risk_score), width="stretch")

        # Recommendations
        st.markdown("<div class='section-header'>AI Recommendations</div>", unsafe_allow_html=True)
        recs = {
            0: ["Continue standard monitoring protocols",
                "Maintain regular patrol schedules",
                "Keep all surveillance systems active"],
            1: ["Increase camera monitoring frequency",
                "Deploy additional security personnel to the area",
                "Issue advisory alert to nearby units"],
            2: ["Immediate security response required",
                "Activate emergency protocols and lock-down procedures",
                "Notify law enforcement and emergency services"],
        }
        for rec in recs[threat_idx]:
            st.markdown(
                f"<div style='background:#FFFFFF;border:1px solid #DADCE0;border-left:3px solid {color};"
                f"padding:10px 14px;border-radius:4px;margin:6px 0;color:#202124;font-size:13px;'>{rec}</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – RISK ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Risk Analytics":
    st.markdown(
        "<h1 style='color:#202124;font-size:24px;font-weight:600;margin-bottom:4px;'>Risk Analytics</h1>"
        "<p style='color:#5F6368;font-size:14px;'>Statistical analysis of threat patterns across the surveillance dataset</p>",
        unsafe_allow_html=True,
    )

    label_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    df_plot = df.copy()
    df_plot["threat_label"] = df_plot["threat_level"].map(label_map)
    df_plot["location_label"] = df_plot["location_risk"].map({0: "Low", 1: "Medium", 2: "High"})

    # ── Row 1: distribution + heatmap ────────────────────────────────────────
    a1, a2 = st.columns(2)

    with a1:
        st.markdown("<div class='section-header'>Threat Level Distribution</div>", unsafe_allow_html=True)
        dist = df_plot["threat_label"].value_counts().reset_index()
        dist.columns = ["Threat Level", "Count"]
        fig_pie = px.pie(
            dist, names="Threat Level", values="Count",
            color="Threat Level",
            color_discrete_map={"LOW": "#34A853", "MEDIUM": "#F9AB00", "HIGH": "#EA4335"},
            hole=0.45,
        )
        fig_pie.update_layout(
            paper_bgcolor="#FFFFFF", font_color="#202124", height=340,
            legend=dict(bgcolor="#FFFFFF"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_pie, width="stretch")

    with a2:
        st.markdown("<div class='section-header'>Crowd Density vs Motion Intensity</div>", unsafe_allow_html=True)
        fig_scatter = px.scatter(
            df_plot.sample(500, random_state=42),
            x="crowd_density", y="motion_intensity",
            color="threat_label",
            color_discrete_map={"LOW": "#34A853", "MEDIUM": "#F9AB00", "HIGH": "#EA4335"},
            opacity=0.7,
            labels={"crowd_density": "Crowd Density", "motion_intensity": "Motion Intensity"},
        )
        fig_scatter.update_layout(
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F8F9FA",
            font_color="#202124", height=340,
            xaxis=dict(gridcolor="#DADCE0"), yaxis=dict(gridcolor="#DADCE0"),
            legend=dict(bgcolor="#FFFFFF"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_scatter, width="stretch")

    # ── Row 2: hourly pattern + location risk ─────────────────────────────────
    b1, b2 = st.columns(2)

    with b1:
        st.markdown("<div class='section-header'>Hourly Threat Frequency</div>", unsafe_allow_html=True)
        hourly = df_plot.groupby(["time_of_day", "threat_label"]).size().reset_index(name="count")
        fig_hourly = px.bar(
            hourly, x="time_of_day", y="count", color="threat_label",
            color_discrete_map={"LOW": "#34A853", "MEDIUM": "#F9AB00", "HIGH": "#EA4335"},
            labels={"time_of_day": "Hour of Day", "count": "Incident Count"},
        )
        fig_hourly.update_layout(
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F8F9FA",
            font_color="#202124", height=320,
            xaxis=dict(gridcolor="#DADCE0"), yaxis=dict(gridcolor="#DADCE0"),
            legend=dict(bgcolor="#FFFFFF"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_hourly, width="stretch")

    with b2:
        st.markdown("<div class='section-header'>Threat Level by Location Risk</div>", unsafe_allow_html=True)
        loc_threat = df_plot.groupby(["location_label", "threat_label"]).size().reset_index(name="count")
        fig_loc = px.bar(
            loc_threat, x="location_label", y="count", color="threat_label",
            barmode="group",
            color_discrete_map={"LOW": "#34A853", "MEDIUM": "#F9AB00", "HIGH": "#EA4335"},
            labels={"location_label": "Location Risk", "count": "Count"},
            category_orders={"location_label": ["Low", "Medium", "High"]},
        )
        fig_loc.update_layout(
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F8F9FA",
            font_color="#202124", height=320,
            xaxis=dict(gridcolor="#DADCE0"), yaxis=dict(gridcolor="#DADCE0"),
            legend=dict(bgcolor="#FFFFFF"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_loc, width="stretch")

    # ── Row 3: summary statistics ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>Dataset Summary Statistics</div>", unsafe_allow_html=True)
    summary = df[["crowd_density", "motion_intensity", "time_of_day"]].describe().round(2)
    st.dataframe(summary, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – AI MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "AI Model Insights":
    st.markdown(
        "<h1 style='color:#202124;font-size:24px;font-weight:600;margin-bottom:4px;'>AI Model Insights</h1>"
        "<p style='color:#5F6368;font-size:14px;'>RandomForestClassifier — performance metrics and interpretability</p>",
        unsafe_allow_html=True,
    )

    y_pred = model.predict(X_test)
    acc    = round((y_pred == y_test).mean() * 100, 2)

    # ── Top KPIs ──────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='metric-card'><h3>ACCURACY</h3><h2 style='color:#34A853;'>{acc}%</h2></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card'><h3>ESTIMATORS</h3><h2 style='color:#1A73E8;'>150</h2></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-card'><h3>MAX DEPTH</h3><h2 style='color:#1A73E8;'>8</h2></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='metric-card'><h3>TRAINING SIZE</h3><h2 style='color:#1A73E8;'>1,600</h2></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature importance + confusion matrix ─────────────────────────────────
    i1, i2 = st.columns(2)

    with i1:
        st.markdown("<div class='section-header'>Feature Importance</div>", unsafe_allow_html=True)
        feat_names = ["Crowd Density", "Motion Intensity", "Time of Day", "Location Risk", "Unauth. Access"]
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
        imp_df = imp_df.sort_values("Importance", ascending=True)

        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#E8F0FE", "#1A73E8"],
            text=imp_df["Importance"].round(3),
        )
        fig_imp.update_traces(textposition="outside")
        fig_imp.update_layout(
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F8F9FA",
            font_color="#202124", height=360, coloraxis_showscale=False,
            xaxis=dict(gridcolor="#DADCE0"), yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_imp, width="stretch")

    with i2:
        st.markdown("<div class='section-header'>Confusion Matrix</div>", unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            x=["LOW", "MEDIUM", "HIGH"],
            y=["LOW", "MEDIUM", "HIGH"],
            color_continuous_scale=["#E8F0FE", "#1A73E8"],
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )
        fig_cm.update_layout(
            paper_bgcolor="#FFFFFF", font_color="#202124", height=360,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_cm, width="stretch")

    # ── Classification report ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Classification Report</div>", unsafe_allow_html=True)
    report_dict = classification_report(
        y_test, y_pred,
        target_names=["LOW", "MEDIUM", "HIGH"],
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).T.round(3)
    report_df = report_df.drop(["accuracy"], errors="ignore")
    st.dataframe(report_df, width="stretch")

    # ── Model architecture ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Model Architecture</div>", unsafe_allow_html=True)
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
                f"<div style='background:#FFFFFF;border:1px solid #DADCE0;border-radius:8px;"
                f"padding:14px;margin-bottom:12px;box-shadow:0 1px 3px rgba(60,64,67,0.08);'>"
                f"<div style='color:#5F6368;font-size:12px;letter-spacing:1px;text-transform:uppercase;'>{k}</div>"
                f"<div style='color:#202124;font-weight:600;margin-top:4px;font-size:13px;'>{v}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
