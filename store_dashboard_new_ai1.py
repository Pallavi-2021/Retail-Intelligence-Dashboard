"""
Retail Cross-Sell Intelligence Platform
Run:     streamlit run retail_intelligence.py
Install: pip install streamlit plotly pandas numpy mlxtend networkx openpyxl scikit-learn
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
import os

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Cross-Sell Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── GOOGLE GEMINI API KEY ──────────────────────────────────────
# Free API key — no credit card needed.
# Get yours at: https://aistudio.google.com  →  "Get API Key"
# Then paste it below between the quotes.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── CONSTANTS ──────────────────────────────────────────────────
PRODUCTS = ["Milk","Bread","Cheese","Yogurt","Butter","Cream","IceCream","Ghee","Paneer"]
QTY_COLS = [f"{p}_Qty" for p in PRODUCTS]
PAL = ["#2563EB","#F59E0B","#10B981","#8B5CF6","#EF4444",
       "#0EA5E9","#84CC16","#F97316","#EC4899","#14B8A6"]
PROD_COL = {p: PAL[i % len(PAL)] for i, p in enumerate(PRODUCTS)}

# ── CHART HELPER ───────────────────────────────────────────────
def fig_layout(fig, title="", h=320, ml=60, mr=25, mt=48, mb=55,
               xtitle="", ytitle="", xgrid=True, ygrid=True,
               xtickangle=0, xrange=None, yrange=None,
               yreversed=False, showlegend=True, barmode=None):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>",
                   font=dict(family="Inter", size=13, color="#111827"),
                   x=0, xanchor="left"),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(family="Inter", color="#374151", size=11),
        height=h, margin=dict(l=ml, r=mr, t=mt, b=mb),
        showlegend=showlegend,
        legend=dict(font=dict(color="#111827", size=11),
                    bgcolor="#FFFFFF", bordercolor="#E5E7EB", borderwidth=1),
        hoverlabel=dict(bgcolor="#1F2937", font_color="#FFFFFF",
                        font=dict(family="Inter", size=12)),
    )
    xa = dict(
        title=dict(text=xtitle, font=dict(color="#374151", size=12, family="Inter")),
        showgrid=xgrid, gridcolor="#F3F4F6", gridwidth=1,
        linecolor="#D1D5DB", linewidth=1, showline=True,
        tickfont=dict(color="#374151", size=11, family="Inter"),
        zeroline=False, ticks="outside", tickcolor="#D1D5DB", tickangle=xtickangle,
    )
    if xrange:
        xa["range"] = xrange
    fig.update_xaxes(**xa)
    ya = dict(
        title=dict(text=ytitle, font=dict(color="#374151", size=12, family="Inter")),
        showgrid=ygrid, gridcolor="#F3F4F6", gridwidth=1,
        linecolor="#D1D5DB", linewidth=1, showline=True,
        tickfont=dict(color="#374151", size=11, family="Inter"),
        zeroline=False, ticks="outside", tickcolor="#D1D5DB",
    )
    if yrange:
        ya["range"] = yrange
    if yreversed:
        ya["autorange"] = "reversed"
    fig.update_yaxes(**ya)
    if barmode:
        fig.update_layout(barmode=barmode)
    return fig

def cb(title_text):
    """Correct colorbar — uses title=dict(...)"""
    return dict(
        title=dict(text=title_text,
                   font=dict(color="#111827", size=12, family="Inter")),
        tickfont=dict(color="#111827", size=10, family="Inter"),
        outlinewidth=0, thickness=13,
    )

def explain(text):
    """Render a plain-language explanation box below a chart."""
    st.markdown(
        f'<div style="background:#F8FAFF;border-left:3px solid #93C5FD;border-radius:0 8px 8px 0;'
        f'padding:0.6rem 1rem;margin:0.3rem 0 1.2rem;font-size:0.82rem;color:#374151;line-height:1.65;">'
        f'💬 {text}</div>',
        unsafe_allow_html=True
    )

# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif;background:#F9FAFB;}
.main .block-container{padding:0 2rem 3rem;max-width:1500px;}
#MainMenu,footer,header{visibility:hidden;}
section[data-testid="stSidebar"]{display:none!important;}

/* Header */
.pg-hdr{background:linear-gradient(135deg,#1E3A5F 0%,#2563EB 100%);
  border-radius:0 0 20px 20px;padding:1.5rem 2.2rem 1.3rem;
  margin:0 -2rem 1.8rem;display:flex;align-items:center;gap:1.2rem;}
.pg-hdr h1{font-size:1.55rem;font-weight:800;color:#FFFFFF!important;margin:0 0 2px;}
.pg-hdr p{color:#BFDBFE!important;font-size:0.83rem;margin:0;}
.pg-hdr .tg{display:inline-block;background:rgba(255,255,255,0.15);
  border:1px solid rgba(255,255,255,0.25);color:#E0F2FE!important;
  padding:2px 10px;border-radius:20px;font-size:0.65rem;font-weight:600;
  text-transform:uppercase;letter-spacing:0.4px;margin:7px 3px 0 0;}

/* KPI */
.krow{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:1.5rem;}
.kc{background:#FFFFFF;border-radius:12px;padding:1rem 1.2rem;
  box-shadow:0 1px 6px rgba(0,0,0,0.07);border-top:4px solid var(--kc,#2563EB);position:relative;}
.ki{position:absolute;right:0.9rem;top:0.9rem;font-size:1.4rem;opacity:0.13;}
.kl{font-size:0.66rem;font-weight:700;text-transform:uppercase;letter-spacing:0.9px;
  color:#6B7280;margin-bottom:4px;}
.kv{font-size:1.7rem;font-weight:800;color:#111827;line-height:1.1;}
.ks{font-size:0.69rem;color:#9CA3AF;margin-top:3px;}

/* Section label */
.sec{font-size:0.66rem;font-weight:700;text-transform:uppercase;letter-spacing:2px;
  color:#2563EB;border-bottom:2px solid #DBEAFE;padding-bottom:5px;margin:1.5rem 0 1rem;}

/* Cards */
.card{background:#FFFFFF;border-radius:12px;padding:1.2rem 1.4rem;
  box-shadow:0 1px 6px rgba(0,0,0,0.07);margin-bottom:0.8rem;}
.ct{font-size:0.66rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;
  color:#6B7280;border-bottom:1px solid #F3F4F6;padding-bottom:6px;margin-bottom:0.8rem;}

/* Upload */
.upz{background:#FFFFFF;border-radius:14px;padding:2rem;text-align:center;
  border:2px dashed #BFDBFE;}
.upz h3{color:#1E3A5F;font-size:1rem;font-weight:700;margin-bottom:4px;}
.upz p{color:#6B7280;font-size:0.78rem;margin:0;}

/* Rec cards */
.rec{background:#F9FAFB;border-left:5px solid var(--rc,#2563EB);border-radius:10px;
  padding:0.75rem 0.9rem;margin-bottom:0.45rem;display:flex;align-items:flex-start;gap:9px;}
.rk{font-size:1.1rem;font-weight:800;color:#D1D5DB;min-width:24px;}
.rn{font-size:0.88rem;font-weight:700;color:#111827;}
.rm{font-size:0.73rem;color:#6B7280;margin-top:1px;}
.rb{height:5px;background:#E5E7EB;border-radius:3px;margin-top:4px;}
.rbf{height:100%;border-radius:3px;}
.bh{display:inline-block;padding:1px 8px;border-radius:12px;font-size:0.61rem;
  font-weight:700;margin-left:5px;background:#FEF2F2;color:#DC2626;border:1px solid #FECACA;}
.bm{display:inline-block;padding:1px 8px;border-radius:12px;font-size:0.61rem;
  font-weight:700;margin-left:5px;background:#FFFBEB;color:#D97706;border:1px solid #FDE68A;}
.bl{display:inline-block;padding:1px 8px;border-radius:12px;font-size:0.61rem;
  font-weight:700;margin-left:5px;background:#EFF6FF;color:#2563EB;border:1px solid #BFDBFE;}

/* Insight */
.insight{background:#EFF6FF;border:1.5px solid #BFDBFE;border-left:5px solid #2563EB;
  border-radius:10px;padding:0.9rem 1.1rem;font-size:0.84rem;
  color:#1E3A5F;line-height:1.72;margin-top:0.7rem;}
.insight strong{color:#1E3A5F;}

/* Cluster mini */
.clc{background:#F9FAFB;border-left:5px solid var(--cc,#2563EB);
  border-radius:8px;padding:0.55rem 0.8rem;margin-bottom:0.38rem;}
.clcn{font-weight:700;font-size:0.86rem;color:#111827;}
.clcm{font-size:0.71rem;color:#6B7280;margin-top:1px;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:#EFF6FF;border-radius:10px;
  padding:4px;gap:2px;border-bottom:none!important;}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:8px;
  color:#374151!important;font-weight:600;font-size:0.79rem;
  padding:7px 15px;border:none!important;}
.stTabs [aria-selected="true"]{background:#1E3A5F!important;color:#FFFFFF!important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.2rem;}

/* Tables */
[data-testid="stDataFrame"] td{color:#111827!important;font-size:0.82rem!important;}
[data-testid="stDataFrame"] th{color:#111827!important;font-weight:700!important;
  background:#F3F4F6!important;font-size:0.78rem!important;}

/* Misc */
.stMarkdown p,.element-container p{color:#374151;}
[data-testid="stAlert"] p{color:#111827!important;}
.stButton>button{background:#2563EB!important;color:#FFFFFF!important;
  border:none!important;border-radius:8px!important;font-weight:700!important;
  font-size:0.86rem!important;padding:0.52rem 1.3rem!important;}
.stButton>button:hover{background:#1D4ED8!important;}
.stSelectbox>div>div,.stMultiSelect>div>div{background:#FFFFFF!important;
  border:1.5px solid #E5E7EB!important;border-radius:8px!important;color:#111827!important;}
label{color:#374151!important;font-weight:600!important;}

/* File uploader — filename and browse text must be dark */
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] div,
[data-testid="stFileDropzone"] span,
[data-testid="stFileDropzone"] p,
[data-testid="stFileDropzone"] small {
  color:#111827!important;
}

/* Expander header and content text — must be dark */
.streamlit-expanderHeader,
.streamlit-expanderHeader p,
.streamlit-expanderHeader span,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {
  color:#111827!important; font-weight:600!important;
}
[data-testid="stExpander"] div,
[data-testid="stExpander"] p,
[data-testid="stExpander"] span {
  color:#374151!important;
}

/* Success / info / warning message text */
[data-testid="stNotification"] p,
[data-testid="stNotification"] span,
.stSuccess p, .stInfo p, .stWarning p {
  color:#111827!important;
}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────────
for k, v in [("df",None),("df_raw",None),("log",[]),
             ("rules",None),("mba_ready",False),("preprocessed",False),
             ("sel_clusters", None), ("ai_messages", [])]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── PREPROCESS FUNCTION ────────────────────────────────────────
def preprocess(raw):
    df = raw.copy()
    log = []
    df.columns = [c.strip() for c in df.columns]
    n = len(df); df = df.drop_duplicates()
    if len(df) < n:
        log.append(f"✅ Removed {n-len(df)} duplicate rows")
    num_cols = (["Total_Visits","Avg_Call_Duration","Salesperson_Count",
                 "StoreLatitude","StoreLongitude","GeoCluster"]
                + PRODUCTS + QTY_COLS)
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in PRODUCTS + QTY_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(float)
    drop_cols = [c for c in ["StoreLatitude","StoreLongitude","GeoCluster"] if c in df.columns]
    before = len(df); df = df.dropna(subset=drop_cols)
    removed = before-len(df)
    if removed:
        log.append(f"✅ Removed {removed} rows with missing coordinates/cluster")
    if "GeoCluster" in df.columns:
        df["GeoCluster"] = df["GeoCluster"].astype(int)
    prod_c = [p for p in PRODUCTS if p in df.columns]
    qty_c  = [q for q in QTY_COLS  if q in df.columns]
    if prod_c:
        df["_n_products"] = df[prod_c].astype(int).sum(axis=1)
    if qty_c:
        df["_total_qty"]  = df[qty_c].sum(axis=1)
    log.append(f"✅ {len(df):,} stores ready for analysis")
    return df, log

# ── MBA FUNCTION ───────────────────────────────────────────────
def run_mba(df):
    prod_c = [p for p in PRODUCTS if p in df.columns]
    if not prod_c:
        return pd.DataFrame()
    basket = df[prod_c].astype(bool)
    try:
        freq  = apriori(basket, min_support=0.03, use_colnames=True)
        if freq.empty:
            return pd.DataFrame()
        rules = association_rules(freq, metric="confidence", min_threshold=0.05)
        rules["from_str"] = rules["antecedents"].apply(lambda x: " + ".join(sorted(x)))
        rules["to_str"]   = rules["consequents"].apply(lambda x: " + ".join(sorted(x)))
        return rules
    except Exception as e:
        st.error(f"MBA error: {e}")
        return pd.DataFrame()

def get_recs(product, rules_df, top_n=5):
    if rules_df is None or rules_df.empty or not product:
        return []
    f = rules_df[rules_df["antecedents"].apply(lambda x: product in x)].copy()
    f = f.sort_values("confidence", ascending=False)
    seen, out = set(), []
    for _, row in f.iterrows():
        cons = " + ".join(sorted(row["consequents"]))
        if cons not in seen:
            seen.add(cons)
            out.append({
                "product":    cons,
                "confidence": round(row["confidence"]*100, 1),
                "lift":       round(row["lift"],           2),
                "support":    round(row["support"]*100,    1),
            })
        if len(out) >= top_n:
            break
    return out

# ══════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="pg-hdr">
  <div style="font-size:2.5rem">📊</div>
  <div>
    <h1>Retail Cross-Sell Intelligence Platform</h1>
    <p>Market Basket Analysis · Geographic Intelligence · Store Performance</p>
    <span class="tg">Cross-Sell Engine</span>
    <span class="tg">Geo Mapping</span>
    <span class="tg">Store Rankings</span>
    <span class="tg">Product Strategy</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD & PREPROCESS
# ══════════════════════════════════════════════════════════════
st.markdown('<p class="sec">📂 STEP 1 — UPLOAD & PREPROCESS YOUR DATA</p>',
            unsafe_allow_html=True)
up_c, info_c = st.columns([1.5, 1])

with up_c:
    uploaded = st.file_uploader("Upload CSV or Excel file",
                                type=["csv","xlsx","xls"])
    if uploaded:
        try:
            raw = (pd.read_csv(uploaded)
                   if uploaded.name.lower().endswith(".csv")
                   else pd.read_excel(uploaded))
            st.session_state["df_raw"]        = raw
            st.session_state["preprocessed"]  = False
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if st.session_state["df_raw"] is not None:
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("⚙️  Preprocess Data", use_container_width=True):
                df_clean, log = preprocess(st.session_state["df_raw"])
                st.session_state.update({
                    "df": df_clean, "log": log,
                    "preprocessed": True,
                    "rules": None, "mba_ready": False,
                })
        with bc2:
            if st.session_state["preprocessed"] and st.session_state["df"] is not None:
                st.success(f"✅ {len(st.session_state['df']):,} stores ready!")

    if st.session_state["log"]:
        with st.expander("🔬 View preprocessing log"):
            for msg in st.session_state["log"]:
                st.write(msg)

with info_c:
    st.markdown("""
    <div class="upz">
      <div style="font-size:2rem;margin-bottom:0.5rem">📋</div>
      <h3>Expected Data Columns</h3>
      <p>StoreID · Total_Visits · Avg_Call_Duration · Salesperson_Count ·
         StoreLatitude · StoreLongitude · GeoCluster ·
         Product flags (Milk, Bread, Cheese … as 0/1) ·
         Product quantities (Milk_Qty, Bread_Qty …)</p>
    </div>
    """, unsafe_allow_html=True)

if st.session_state["df"] is None:
    st.info("👆 Upload your dataset and click **⚙️ Preprocess Data** to begin.")
    st.stop()

# ══════════════════════════════════════════════════════════════
# DATA IS READY
# ══════════════════════════════════════════════════════════════
df        = st.session_state["df"]
prod_cols = [p for p in PRODUCTS if p in df.columns]
qty_cols  = [q for q in QTY_COLS  if q in df.columns]

# ── KPI ROW ────────────────────────────────────────────────────
avg_vis = f"{round(df['Total_Visits'].mean())} visits"       if "Total_Visits"       in df.columns else "—"
avg_dur = f"{round(df['Avg_Call_Duration'].mean())} min"     if "Avg_Call_Duration"  in df.columns else "—"
tot_qty = f"{int(df['_total_qty'].sum()):,} units"           if "_total_qty"         in df.columns else "—"
avg_p   = f"{round(df['_n_products'].mean())} products"      if "_n_products"        in df.columns else "—"
n_cl    = df["GeoCluster"].nunique()                         if "GeoCluster"         in df.columns else "—"

st.markdown(f"""
<div class="krow">
  <div class="kc" style="--kc:#2563EB"><div class="ki">🏪</div>
    <div class="kl">Total Stores</div><div class="kv">{len(df):,}</div>
    <div class="ks">{n_cl} geographic clusters</div></div>
  <div class="kc" style="--kc:#F59E0B"><div class="ki">📞</div>
    <div class="kl">Avg Visits / Store</div><div class="kv">{avg_vis}</div>
    <div class="ks">sales visits per store</div></div>
  <div class="kc" style="--kc:#10B981"><div class="ki">⏱️</div>
    <div class="kl">Avg Call Duration</div><div class="kv">{avg_dur}</div>
    <div class="ks">minutes per visit</div></div>
  <div class="kc" style="--kc:#8B5CF6"><div class="ki">📦</div>
    <div class="kl">Total Units Sold</div><div class="kv">{tot_qty}</div>
    <div class="ks">across all products</div></div>
  <div class="kc" style="--kc:#EF4444"><div class="ki">🛒</div>
    <div class="kl">Avg Products / Store</div><div class="kv">{avg_p}</div>
    <div class="ks">out of {len(prod_cols)} tracked</div></div>
</div>
""", unsafe_allow_html=True)

# ── CLUSTER-FILTERED DATAFRAME ──────────────────────────────────
# All tabs (cross-sell, products, performance) use this.
# It reflects whatever clusters the user selected in the Store Map tab.
_all_cl   = sorted(df["GeoCluster"].unique()) if "GeoCluster" in df.columns else []
_sel_cl   = st.session_state["sel_clusters"]
# If nothing saved yet (first load), use all clusters
if _sel_cl is None or len(_sel_cl) == 0:
    _sel_cl = _all_cl
# Keep only valid cluster values
_sel_cl = [c for c in _sel_cl if c in _all_cl] or _all_cl
dff = df[df["GeoCluster"].isin(_sel_cl)].copy() if "GeoCluster" in df.columns else df.copy()

# Show a small active-filter badge if clusters are filtered
if set(_sel_cl) != set(_all_cl):
    cl_names = ", ".join([f"Cluster {int(c)}" for c in sorted(_sel_cl)])
    st.markdown(
        f'<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;'
        f'padding:0.4rem 1rem;margin-bottom:1rem;font-size:0.8rem;color:#1E3A5F;">'
        f'🔍 <strong>Active cluster filter:</strong> {cl_names} &nbsp;·&nbsp; '
        f'<strong>{len(dff):,} stores</strong> shown across all tabs. '
        f'Change selection in the <strong>Store Map</strong> tab.</div>',
        unsafe_allow_html=True
    )

# ── TABS ───────────────────────────────────────────────────────
tab_ov, tab_map, tab_cs, tab_prod, tab_perf, tab_ai = st.tabs([
    "📊  Overview",
    "🌍  Store Map",
    "🎯  Cross-Sell Analysis",
    "📦  Product Performance",
    "🏪  Store Performance",
    "🤖  AI Advisor",
])

# ════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════
with tab_ov:
    # ── Cluster explanation banner ──────────────────────────────
    st.markdown('<p class="sec">🗺️ HOW ARE STORES GROUPED INTO CLUSTERS?</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#FFFFFF;border-radius:12px;padding:1.1rem 1.4rem;
         box-shadow:0 1px 6px rgba(0,0,0,0.07);margin-bottom:1.2rem;
         border-left:5px solid #F59E0B;">
      <p style="font-size:0.88rem;color:#111827;font-weight:700;margin:0 0 6px;">
        📌 What is a "Cluster" and why does it matter?</p>
      <p style="font-size:0.83rem;color:#374151;line-height:1.7;margin:0;">
        A <strong>cluster</strong> is a group of stores that are located
        <strong>geographically close to each other</strong>. We used a technique
        called <em>geo-clustering</em> (based on store latitude and longitude) to
        automatically divide all stores into <strong>5 natural groups</strong> —
        Cluster 0 through Cluster 4 — based purely on where they are on the map.<br><br>
        Think of it like dividing a country into <strong>sales regions</strong>:
        stores in the same cluster are in the same region, face similar local demand,
        and can be served by the same sales team or distributor.
        Analysing clusters separately helps you spot which regions sell more of
        certain products, where cross-sell opportunities are strongest, and
        where your sales team should focus next.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sec">DISTRIBUTION OVERVIEW</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:  # Cluster donut
        if "GeoCluster" in df.columns:
            cdata  = df["GeoCluster"].value_counts().sort_index()
            labels = [f"Cluster {int(x)}" for x in cdata.index]
            fig = go.Figure(go.Pie(
                labels=labels, values=cdata.values.tolist(),
                hole=0.56,
                marker=dict(colors=PAL[:len(cdata)],
                            line=dict(color="#FFFFFF", width=2)),
                textfont=dict(color="#111827", size=11, family="Inter"),
                hovertemplate="<b>%{label}</b><br>%{value:,} stores (%{percent})<extra></extra>",
                textinfo="percent+label",
            ))
            fig_layout(fig, "Stores by Geo Cluster", h=300,
                       ml=20, mr=20, mt=45, mb=15, showlegend=False)
            fig.update_layout(annotations=[dict(
                text=f"<b>{len(df):,}</b><br>stores",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#111827", family="Inter"),
            )])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            explain("This donut chart shows how your stores are distributed across geographic regions (clusters). "
                    "A larger slice means more stores in that region. Use this to quickly see if your store network "
                    "is concentrated in one area or spread evenly across the country.")

    with c2:  # Visit histogram (using Bar + numpy for guaranteed labels)
        if "Total_Visits" in df.columns:
            v      = df["Total_Visits"].dropna()
            cap    = int(v.quantile(0.95))
            v_cap  = v.clip(upper=cap)
            counts, edges = np.histogram(v_cap, bins=min(18, cap))
            bin_lbl = [str(int(edges[i])) for i in range(len(edges)-1)]

            fig = go.Figure(go.Bar(
                x=bin_lbl, y=counts.tolist(),
                marker=dict(color="#2563EB", line=dict(color="#FFFFFF", width=0.8)),
                text=[f"{c:,}" for c in counts],
                textposition="outside",
                textfont=dict(color="#374151", size=9, family="Inter"),
                hovertemplate="Visits: %{x}<br>Number of stores: %{y:,}<extra></extra>",
            ))
            fig_layout(fig, "How Often Are Stores Visited?", h=300,
                       xtitle="Number of Visits (per store, all time)",
                       ytitle="Number of Stores",
                       xgrid=False, mb=50)
            fig.update_layout(bargap=0.08)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            explain("Each bar shows how many stores received a certain number of visits from your sales team. "
                    "Tall bars on the left mean most stores are visited only a few times. "
                    "If many stores cluster at low visit counts, there may be untapped sales opportunities "
                    "simply by increasing visit frequency.")

    with c3:  # Product presence
        if prod_cols:
            pct = (df[prod_cols].sum() / len(df) * 100).sort_values()
            fig = go.Figure(go.Bar(
                x=pct.values.tolist(), y=pct.index.tolist(), orientation="h",
                marker=dict(color=[PROD_COL.get(p,"#2563EB") for p in pct.index],
                            line=dict(width=0)),
                text=[f"{round(v)}%" for v in pct.values],
                textposition="outside",
                textfont=dict(color="#374151", size=10, family="Inter"),
                hovertemplate="<b>%{y}</b>: stocked in %{x:.0f}% of stores<extra></extra>",
            ))
            fig_layout(fig, "Which Products Are Most Widely Stocked?", h=300,
                       xtitle="% of Stores", ygrid=False, ml=80, xrange=[0,115])
            fig.update_xaxes(ticksuffix="%")
            fig.update_layout(bargap=0.22)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            explain("Each bar shows what percentage of stores currently stock that product. "
                    "Products with shorter bars are in fewer stores — these are your biggest distribution "
                    "expansion opportunities. A product in only 30% of stores could potentially double "
                    "its revenue just by being introduced to the remaining 70%.")

    st.markdown('<p class="sec">💡 EXECUTIVE SUMMARY</p>', unsafe_allow_html=True)
    if prod_cols and "_n_products" in df.columns:
        pcts   = df[prod_cols].sum() / len(df) * 100
        top_p  = pcts.idxmax();  top_pct = round(pcts.max())
        low_p  = pcts.idxmin();  low_pct = round(pcts.min())
        avg_np = round(df["_n_products"].mean())
        tot_u  = int(df["_total_qty"].sum()) if "_total_qty" in df.columns else 0
        st.markdown(f"""
        <div class="insight">
            We are analysing <strong>{len(df):,} retail stores</strong> across
            <strong>{n_cl} geographic clusters</strong>.<br><br>
            🏆 <strong>{top_p}</strong> is the most widely stocked product —
            present in <strong>{top_pct}%</strong> of stores, making it the
            strongest anchor for cross-sell campaigns.<br><br>
            📈 <strong>{low_p}</strong> has the lowest coverage at
            <strong>{low_pct}%</strong> — a significant growth opportunity
            worth targeting at the cluster level.<br><br>
            🛒 Each store carries on average <strong>{avg_np} out of {len(prod_cols)}
            products</strong>, with <strong>{tot_u:,} total units</strong> sold.
            Increasing product variety per store is the fastest path to revenue growth.
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 2 — STORE MAP
# ════════════════════════════════════════════════════════
with tab_map:
    if "StoreLatitude" not in df.columns or "StoreLongitude" not in df.columns:
        st.warning("No latitude/longitude columns found.")
    else:
        st.markdown('<p class="sec">🔧 MAP FILTERS</p>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        with f1:
            all_cl  = sorted(df["GeoCluster"].unique()) if "GeoCluster" in df.columns else []
            # Default to previously selected clusters if available, else all
            default_cl = st.session_state["sel_clusters"] if st.session_state["sel_clusters"] is not None else all_cl
            # Make sure default_cl only contains valid values
            default_cl = [c for c in default_cl if c in all_cl] or all_cl
            sel_cl  = st.multiselect("Filter by Geo Cluster", all_cl, default=default_cl,
                                     key="map_cl")
            # Save to session state so other tabs can use it
            st.session_state["sel_clusters"] = sel_cl
        with f2:
            vmin = int(df["Total_Visits"].min()); vmax = int(df["Total_Visits"].max())
            vis_r = st.slider("Total Visits Range", vmin, vmax, (vmin, vmax), key="map_vis")
        with f3:
            colour_by = st.selectbox(
                "Colour stores by",
                ["Geo Cluster","Total Visits","Call Duration",
                 "Products Stocked","Total Units Sold"],
                key="map_col",
            )

        dff = df.copy()
        if sel_cl and "GeoCluster" in dff.columns:
            dff = dff[dff["GeoCluster"].isin(sel_cl)]
        if "Total_Visits" in dff.columns:
            dff = dff[dff["Total_Visits"].between(vis_r[0], vis_r[1])]
        dff = dff.dropna(subset=["StoreLatitude","StoreLongitude"])

        if len(dff) == 0:
            st.warning("No stores match the current filters.")
        else:
            st.markdown('<p class="sec">STORE LOCATIONS</p>', unsafe_allow_html=True)
            mc, sc = st.columns([2.2, 1])

            with mc:
                def hover(row):
                    sid = str(row.get("StoreID","Store"))
                    cl  = f"Cluster {int(row['GeoCluster'])}" if "GeoCluster" in row.index else ""
                    vis = f"Visits: {int(row['Total_Visits'])}" if "Total_Visits" in row.index else ""
                    dur = f"Call: {row['Avg_Call_Duration']:.1f}m" if "Avg_Call_Duration" in row.index else ""
                    np_ = f"Products: {int(row['_n_products'])}" if "_n_products" in row.index else ""
                    qty = f"Units: {int(row['_total_qty']):,}" if "_total_qty" in row.index else ""
                    return "<br>".join(x for x in [f"<b>{sid}</b>",cl,vis,dur,np_,qty] if x)

                dff = dff.copy()
                dff["_hover"] = dff.apply(hover, axis=1)
                clat = float(dff["StoreLatitude"].mean())
                clon = float(dff["StoreLongitude"].mean())

                if colour_by == "Geo Cluster" and "GeoCluster" in dff.columns:
                    fig_map = go.Figure()
                    for i, cl in enumerate(sorted(dff["GeoCluster"].unique())):
                        sub = dff[dff["GeoCluster"] == cl]
                        fig_map.add_trace(go.Scattermapbox(
                            lat=sub["StoreLatitude"].tolist(),
                            lon=sub["StoreLongitude"].tolist(),
                            mode="markers",
                            marker=dict(size=7, color=PAL[i % len(PAL)], opacity=0.85),
                            text=sub["_hover"].tolist(),
                            hoverinfo="text",
                            name=f"Cluster {int(cl)}",
                        ))
                else:
                    col_map   = {"Total Visits":"Total_Visits",
                                 "Call Duration":"Avg_Call_Duration",
                                 "Products Stocked":"_n_products",
                                 "Total Units Sold":"_total_qty"}
                    scale_map = {"Total Visits":"Blues","Call Duration":"Oranges",
                                 "Products Stocked":"Greens","Total Units Sold":"YlOrRd"}
                    c_col   = col_map.get(colour_by, "_n_products")
                    c_scale = scale_map.get(colour_by, "Blues")
                    c_vals  = dff[c_col].tolist() if c_col in dff.columns else None

                    fig_map = go.Figure(go.Scattermapbox(
                        lat=dff["StoreLatitude"].tolist(),
                        lon=dff["StoreLongitude"].tolist(),
                        mode="markers",
                        marker=dict(size=7, opacity=0.85,
                                    color=c_vals, colorscale=c_scale,
                                    showscale=True, colorbar=cb(colour_by)),
                        text=dff["_hover"].tolist(),
                        hoverinfo="text",
                    ))

                fig_map.update_layout(
                    mapbox=dict(style="carto-positron",
                                center=dict(lat=clat, lon=clon), zoom=5),
                    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                    margin=dict(l=0, r=0, t=8, b=0), height=460,
                    font=dict(family="Inter", color="#374151"),
                    legend=dict(font=dict(color="#111827", size=11, family="Inter"),
                                bgcolor="rgba(255,255,255,0.95)",
                                bordercolor="#E5E7EB", borderwidth=1),
                )
                st.plotly_chart(fig_map, use_container_width=True,
                                config={"displayModeBar": False})

            with sc:
                st.markdown('<div class="card"><div class="ct">CLUSTER BREAKDOWN</div>',
                            unsafe_allow_html=True)
                if "GeoCluster" in df.columns:
                    for i, cl in enumerate(sorted(df["GeoCluster"].unique())):
                        sub   = df[df["GeoCluster"] == cl]
                        count = len(sub); pct = count/len(df)*100
                        avg_v = f"{round(sub['Total_Visits'].mean())} visits" \
                                if "Total_Visits" in sub.columns else "—"
                        avg_d = f"{round(sub['Avg_Call_Duration'].mean())} min" \
                                if "Avg_Call_Duration" in sub.columns else "—"
                        cc = PAL[i % len(PAL)]
                        st.markdown(f"""
                        <div class="clc" style="--cc:{cc}">
                          <div class="clcn">Cluster {int(cl)}
                            <span style="float:right;color:{cc};font-weight:700;
                                         font-size:0.77rem">{count:,} ({pct:.0f}%)</span>
                          </div>
                          <div class="clcm">Avg visits: <b>{avg_v}</b> · Call: <b>{avg_d}</b></div>
                        </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Product demand heatmap by cluster
            if qty_cols and "GeoCluster" in df.columns:
                st.markdown('<p class="sec">PRODUCT DEMAND BY CLUSTER</p>',
                            unsafe_allow_html=True)
                heat = df.groupby("GeoCluster")[qty_cols].sum()
                heat.columns = [c.replace("_Qty","") for c in heat.columns]
                heat.index   = [f"Cluster {int(x)}" for x in heat.index]
                z_vals   = heat.values.tolist()
                text_co  = [[f"{int(v):,}" for v in row] for row in heat.values]

                fig_h = go.Figure(go.Heatmap(
                    z=z_vals,
                    x=heat.columns.tolist(),
                    y=heat.index.tolist(),
                    colorscale=[[0,"#EFF6FF"],[0.5,"#3B82F6"],[1,"#1E3A5F"]],
                    hovertemplate="<b>%{y} × %{x}</b><br>Units sold: %{z:,} units<extra></extra>",
                    showscale=True, colorbar=cb("Units"),
                    text=text_co, texttemplate="%{text}",
                    textfont=dict(color="#111827", size=9, family="Inter"),
                ))
                fig_layout(fig_h, "Product Demand by Cluster (Total Units Sold)",
                           h=260, ml=90, mr=80, mt=45, mb=60,
                           xgrid=False, ygrid=False)
                fig_h.update_xaxes(tickangle=-30,
                                   tickfont=dict(color="#374151", size=10, family="Inter"))
                fig_h.update_yaxes(autorange="reversed",
                                   tickfont=dict(color="#374151", size=10, family="Inter"))
                st.plotly_chart(fig_h, use_container_width=True,
                                config={"displayModeBar": False})
                explain("Each cell shows the total units sold for a product in a specific cluster (region). "
                        "Darker blue = more units sold. This helps you instantly see which products are "
                        "popular in which regions, so you can focus your distribution and promotions "
                        "where they will have the most impact.")

# ════════════════════════════════════════════════════════
# TAB 3 — CROSS-SELL
# ════════════════════════════════════════════════════════
with tab_cs:
    st.markdown('<p class="sec">🎯 CROSS-SELL SETTINGS</p>', unsafe_allow_html=True)
    cs1, cs2, cs3 = st.columns([1.5, 1, 1])
    with cs1:
        sel_product = st.selectbox("Select product to analyse", prod_cols, key="cs_prod")
    with cs2:
        top_n = st.slider("Top N recommendations", 3, 7, 5, key="cs_topn")
    with cs3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶  Run Cross-Sell Analysis", use_container_width=True, key="cs_run"):
            with st.spinner("Analysing buying patterns…"):
                # Run MBA on the cluster-filtered dataset
                r = run_mba(dff)
                st.session_state["rules"]     = r
                st.session_state["mba_ready"] = True
            if r is not None and not r.empty:
                st.success(f"✅ {len(r)} association rules generated!")
            else:
                st.warning("No rules found.")

    if not st.session_state["mba_ready"]:
        st.info("👆 Click **▶ Run Cross-Sell Analysis** to generate recommendations.")
    elif st.session_state["rules"] is None or st.session_state["rules"].empty:
        st.warning("No cross-sell rules available.")
    else:
        rules_df = st.session_state["rules"]
        recs     = get_recs(sel_product, rules_df, top_n)
        st.markdown(f'<p class="sec">RECOMMENDATIONS FOR — {sel_product.upper()}</p>',
                    unsafe_allow_html=True)
        left, right = st.columns([1, 1.6])

        with left:
            if not recs:
                st.info(f"No rules found for **{sel_product}**.")
            else:
                for i, r in enumerate(recs, 1):
                    c     = r["confidence"]
                    color = "#EF4444" if c>=50 else "#F59E0B" if c>=30 else "#2563EB"
                    badge = (f'<span class="bh">High</span>'   if c>=50 else
                             f'<span class="bm">Medium</span>' if c>=30 else
                             f'<span class="bl">Low</span>')
                    st.markdown(f"""
                    <div class="rec" style="--rc:{color}">
                      <div class="rk">#{i}</div>
                      <div style="flex:1">
                        <div class="rn">{r['product']} {badge}</div>
                        <div class="rm">Confidence: <b>{c}%</b> &nbsp;·&nbsp;
                          Lift: <b>{r['lift']}×</b> &nbsp;·&nbsp;
                          In <b>{r['support']}%</b> of stores</div>
                        <div class="rb">
                          <div class="rbf" style="width:{min(c,100)}%;background:{color}">
                          </div>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                t  = recs[0]
                s2 = (f" <strong>{recs[1]['product']}</strong> is also a strong pairing "
                      f"at <strong>{recs[1]['confidence']}%</strong>."
                      if len(recs) > 1 else "")
                st.markdown(f"""
                <div class="insight">
                  📌 <strong>What this means:</strong><br><br>
                  Stores stocking <strong>{sel_product}</strong> are
                  <strong>{t['confidence']}%</strong> likely to also carry
                  <strong>{t['product']}</strong> — <strong>{t['lift']}×</strong>
                  stronger than random chance. This appears in
                  <strong>{t['support']}%</strong> of stores. Bundle these in
                  promotions and sales visit routes.{s2}
                </div>""", unsafe_allow_html=True)

        with right:
            if recs:
                prods_r  = [r["product"]    for r in recs]
                confs_r  = [r["confidence"] for r in recs]
                lifts_r  = [r["lift"]       for r in recs]
                colors_r = ["#EF4444" if c>=50 else "#F59E0B" if c>=30 else "#2563EB"
                            for c in confs_r]
                fig_bar = go.Figure(go.Bar(
                    x=confs_r, y=prods_r, orientation="h",
                    marker=dict(color=colors_r, line=dict(width=0)),
                    text=[f" {c}%" for c in confs_r], textposition="outside",
                    textfont=dict(color="#374151", size=11, family="Inter"),
                    customdata=lifts_r,
                    hovertemplate=("<b>%{y}</b><br>Confidence: %{x}%<br>"
                                   "Lift: %{customdata}×<extra></extra>"),
                ))
                fig_layout(fig_bar, f"Cross-Sell Confidence — {sel_product}",
                           h=max(230, len(recs)*52+60), ml=100,
                           xtitle="Confidence (% of stores that stock both products)",
                           xrange=[0, max(confs_r)+22],
                           ygrid=False, yreversed=True)
                fig_bar.update_xaxes(ticksuffix="%")
                fig_bar.update_yaxes(tickfont=dict(color="#374151", size=10, family="Inter"))
                fig_bar.update_layout(bargap=0.28)
                st.plotly_chart(fig_bar, use_container_width=True,
                                config={"displayModeBar": False})
                explain(f"The longer the bar, the more likely a store that stocks {sel_product} "
                        "also stocks that product. Red = very strong signal (bundle these together in promotions). "
                        "Orange = moderate signal. Blue = worth monitoring.")
            net_r = rules_df[
                rules_df["antecedents"].apply(lambda x: sel_product in x) |
                rules_df["consequents"].apply(lambda x: sel_product in x)
            ].sort_values("lift", ascending=False).head(12)
            if not net_r.empty:
                G = nx.DiGraph()
                for _, row in net_r.iterrows():
                    for a in row["antecedents"]:
                        for b in row["consequents"]:
                            G.add_edge(a, b, lift=row["lift"])
                pos = nx.spring_layout(G, seed=42, k=2.5)
                ex, ey = [], []
                for u, v in G.edges():
                    x0,y0=pos[u]; x1,y1=pos[v]
                    ex += [x0,x1,None]; ey += [y0,y1,None]
                nx_x,nx_y,nlab,ncol,nsz = [],[],[],[],[]
                for node in G.nodes():
                    x,y=pos[node]
                    nx_x.append(x); nx_y.append(y); nlab.append(node)
                    ncol.append("#EF4444" if node==sel_product else "#2563EB")
                    nsz.append(36 if node==sel_product else 22)
                fig_net = go.Figure()
                fig_net.add_trace(go.Scatter(x=ex,y=ey,mode="lines",
                    line=dict(width=1.5,color="#D1D5DB"),hoverinfo="none",showlegend=False))
                fig_net.add_trace(go.Scatter(x=nx_x,y=nx_y,mode="markers+text",
                    text=nlab, textposition="top center",
                    textfont=dict(size=10,color="#111827",family="Inter"),
                    marker=dict(size=nsz,color=ncol,line=dict(width=2,color="#FFFFFF")),
                    hoverinfo="text",showlegend=False))
                fig_layout(fig_net, f"Product Network — {sel_product}", h=290)
                fig_net.update_xaxes(showgrid=False,zeroline=False,showticklabels=False,showline=False)
                fig_net.update_yaxes(showgrid=False,zeroline=False,showticklabels=False,showline=False)
                st.plotly_chart(fig_net, use_container_width=True,
                                config={"displayModeBar": False})
                explain(f"This is a map of product relationships. "
                        f"The <span style='color:#EF4444;font-weight:700'>{sel_product}</span> node (red) "
                        "is your selected product. Every other node it is connected to is a product that "
                        "frequently appears in the same store. Products with more connections are "
                        "better bundling partners — promote or shelve them together to increase basket size.")

        st.markdown('<p class="sec">ALL RULES — RANKED BY LIFT</p>', unsafe_allow_html=True)
        tbl = rules_df[["from_str","to_str","support","confidence","lift"]].copy()
        tbl.columns = ["If a store stocks →","It likely also stocks →",
                       "Support","Confidence","Lift"]
        tbl["Support"]    = (tbl["Support"]    *100).round(1).astype(str)+"%"
        tbl["Confidence"] = (tbl["Confidence"] *100).round(1).astype(str)+"%"
        tbl["Lift"]       = tbl["Lift"].round(2)
        tbl = tbl.sort_values("Lift", ascending=False).reset_index(drop=True)
        st.dataframe(tbl, use_container_width=True, height=260, hide_index=True)
        explain("This table lists every cross-sell rule found in the data. "
                "<strong>Support</strong> = how common the pairing is across all stores. "
                "<strong>Confidence</strong> = if a store has the first product, how likely it also has the second. "
                "<strong>Lift</strong> = how much more likely this pairing is compared to random chance "
                "(anything above 1.0 is a meaningful signal; above 2.0 is a strong one). "
                "Sort by Lift to find your most powerful cross-sell opportunities.")

        # ── BOSS'S GRAPH: Cross-Sell Strength Scatter ──────────
        st.markdown('<p class="sec">📈 CROSS-SELL STRENGTH — SELECTED PRODUCT VS ALL OTHER SKUs</p>',
                    unsafe_allow_html=True)

        if prod_cols and sel_product in dff.columns and "_total_qty" in dff.columns:
            other_prods = [p for p in prod_cols if p != sel_product]
            qty_sel_col = f"{sel_product}_Qty"
            qty_other_cols = [f"{p}_Qty" for p in other_prods if f"{p}_Qty" in dff.columns]

            if qty_sel_col in dff.columns and qty_other_cols:
                # For each store compute: avg other-SKU qty sold and selected-product qty
                scatter_df = dff[[qty_sel_col] + qty_other_cols].copy()
                scatter_df["avg_other"] = scatter_df[qty_other_cols].mean(axis=1)
                scatter_df["sel_qty"]   = scatter_df[qty_sel_col]

                # Compute median thresholds from the filtered data
                med_sel   = scatter_df["sel_qty"].median()
                med_other = scatter_df["avg_other"].median()

                # Classify each store
                def classify(row):
                    if row["sel_qty"] > med_sel and row["avg_other"] > med_other:
                        return "SKU Dominates in Store"
                    elif row["sel_qty"] <= med_sel and row["avg_other"] > med_other:
                        return "High Cross-Sell Opportunity"
                    else:
                        return "Average / In-Line"

                scatter_df["Performance_Flag"] = scatter_df.apply(classify, axis=1)

                # Sample for performance (max 3000 points)
                if len(scatter_df) > 3000:
                    scatter_df = scatter_df.sample(3000, random_state=42)

                color_map = {
                    "SKU Dominates in Store":    "#3B82F6",  # blue
                    "High Cross-Sell Opportunity":"#F97316",  # orange
                    "Average / In-Line":          "#10B981",  # green
                }

                fig_cs = go.Figure()
                for flag, color in color_map.items():
                    sub = scatter_df[scatter_df["Performance_Flag"] == flag]
                    fig_cs.add_trace(go.Scatter(
                        x=sub["avg_other"].tolist(),
                        y=sub["sel_qty"].tolist(),
                        mode="markers",
                        name=flag,
                        marker=dict(color=color, size=7, opacity=0.75,
                                    line=dict(width=0.5, color="white")),
                        hovertemplate=(f"Avg other-product sales: %{{x:.0f}} units<br>"
                                       f"{sel_product} sales: %{{y}} units<br>"
                                       f"<b>%{{fullData.name}}</b><extra></extra>"),
                    ))
                # Add threshold lines
                fig_cs.add_hline(y=med_sel, line=dict(color="#6B7280", width=1, dash="dot"),
                                 annotation_text=f"Median {sel_product} qty",
                                 annotation_font=dict(color="#6B7280", size=10))
                fig_cs.add_vline(x=med_other, line=dict(color="#6B7280", width=1, dash="dot"),
                                 annotation_text="Median other-SKU qty",
                                 annotation_font=dict(color="#6B7280", size=10))

                fig_layout(fig_cs,
                           f"{sel_product} Sales vs Average Sales of Other Products",
                           h=400, ml=70, mr=30, mt=55, mb=60,
                           xtitle="Average Sales of Other Products (units per store)",
                           ytitle=f"{sel_product} Units Sold (per store)")
                fig_cs.update_layout(
                    legend=dict(
                        title=dict(text="Store Category",
                                   font=dict(color="#111827", size=11)),
                        font=dict(color="#111827", size=11),
                        bgcolor="#FFFFFF", bordercolor="#E5E7EB", borderwidth=1,
                    )
                )
                st.plotly_chart(fig_cs, use_container_width=True,
                                config={"displayModeBar": False})
                explain(f"This chart plots every store: the X-axis shows how well other products sell "
                        f"in that store on average, and the Y-axis shows how well <strong>{sel_product}</strong> sells. "
                        f"<br>🟢 <strong>Average / In-Line</strong> — store performs as expected. "
                        f"🔵 <strong>SKU Dominates in Store</strong> — {sel_product} sells strongly here AND other products "
                        f"sell well; these are your best stores. "
                        f"🟠 <strong>High Cross-Sell Opportunity</strong> — other products sell well but "
                        f"{sel_product} is underperforming. These stores are your biggest immediate opportunity: "
                        f"push {sel_product} harder here and you could significantly increase revenue.")

# ════════════════════════════════════════════════════════
# TAB 4 — PRODUCT PERFORMANCE
# ════════════════════════════════════════════════════════
with tab_prod:
    st.markdown('<p class="sec">PRODUCT PERFORMANCE DEEP-DIVE</p>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)

    with p1:
        if qty_cols:
            qtots = dff[qty_cols].sum().sort_values()
            qtots.index = [c.replace("_Qty","") for c in qtots.index]
            fig = go.Figure(go.Bar(
                x=qtots.values.tolist(), y=qtots.index.tolist(), orientation="h",
                marker=dict(color=[PROD_COL.get(p,"#2563EB") for p in qtots.index],
                            line=dict(width=0)),
                text=[f" {int(v):,}" for v in qtots.values],
                textposition="outside",
                textfont=dict(color="#374151", size=10, family="Inter"),
                hovertemplate="<b>%{y}</b>: %{x:,} units sold<extra></extra>",
            ))
            fig_layout(fig, "Total Units Sold — Which Products Move the Most?",
                       h=340, ml=80,
                       xtitle="Total Units Sold (all stores combined)",
                       ygrid=False)
            fig.update_layout(bargap=0.22)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            explain("This bar chart ranks every product by the total number of units sold across "
                    "all stores in the selected clusters. The longer the bar, the more in-demand that product is. "
                    "Use this to prioritise which products to push hardest in your sales visits — "
                    "the top products are already popular, while products at the bottom may need "
                    "targeted promotions or better shelf placement.")

    with p2:
        if prod_cols:
            # Co-purchase: int dot product → copy array → fill diagonal 0
            bm  = dff[prod_cols].astype(int)
            co  = bm.T.dot(bm)           # DataFrame with real counts
            arr = co.to_numpy().copy()   # writable copy
            np.fill_diagonal(arr, 0)
            z_co   = arr.tolist()
            x_co   = co.columns.tolist()
            y_co   = co.index.tolist()
            txt_co = [[f"{int(v):,}" for v in row] for row in arr]

            fig_co = go.Figure(go.Heatmap(
                z=z_co, x=x_co, y=y_co,
                colorscale=[[0,"#F9FAFB"],[0.4,"#3B82F6"],[1,"#1E3A5F"]],
                hovertemplate=("<b>%{x}</b> and <b>%{y}</b><br>"
                               "Both stocked together in %{z:,} stores<extra></extra>"),
                showscale=True, colorbar=cb("Stores"),
                text=txt_co, texttemplate="%{text}",
                textfont=dict(color="#111827", size=9, family="Inter"),
            ))
            fig_layout(fig_co, "Which Products Are Most Often Stocked Together?",
                       h=340, ml=80, mr=80, mb=80, xgrid=False, ygrid=False)
            fig_co.update_xaxes(tickangle=-35,
                                tickfont=dict(color="#374151", size=10, family="Inter"))
            fig_co.update_yaxes(autorange="reversed",
                                tickfont=dict(color="#374151", size=10, family="Inter"))
            st.plotly_chart(fig_co, use_container_width=True,
                            config={"displayModeBar": False})
            explain("Each cell in this grid shows how many stores stock both products at the same time. "
                    "A darker blue cell means those two products appear together in more stores — "
                    "they are natural companions. This tells you which products make the best "
                    "bundling pairs for promotions, combo deals, or joint shelf placement. "
                    "The darker the cell, the stronger the pairing.")

    if qty_cols and "GeoCluster" in dff.columns:
        st.markdown('<p class="sec">HOW DOES PRODUCT DEMAND VARY BY CLUSTER?</p>',
                    unsafe_allow_html=True)
        by_cl = dff.groupby("GeoCluster")[qty_cols].sum()
        by_cl.columns = [c.replace("_Qty","") for c in by_cl.columns]
        fig_gr = go.Figure()
        for j, cl in enumerate(sorted(by_cl.index)):
            fig_gr.add_trace(go.Bar(
                name=f"Cluster {int(cl)}",
                x=by_cl.columns.tolist(),
                y=by_cl.loc[cl].tolist(),
                marker=dict(color=PAL[j % len(PAL)]),
                hovertemplate=(f"Cluster {int(cl)}<br>"
                               "<b>%{x}</b>: %{y:,} units sold<extra></extra>"),
            ))
        fig_layout(fig_gr, "Units Sold per Product per Cluster", h=320,
                   xtitle="Product",
                   ytitle="Total Units Sold (all stores in cluster)",
                   xgrid=False,
                   barmode="group", mb=60)
        fig_gr.update_layout(bargap=0.18, bargroupgap=0.05)
        st.plotly_chart(fig_gr, use_container_width=True, config={"displayModeBar": False})
        explain("Each group of coloured bars represents one product. Each bar within the group is a "
                "different geographic cluster (region). This lets you instantly compare how demand for "
                "each product differs across regions. A tall bar in one cluster but a short bar in another "
                "means that product is popular in one region but underutilised in another — "
                "a clear signal to increase distribution or run a targeted promotion in the lagging region.")

# ════════════════════════════════════════════════════════
# TAB 5 — STORE PERFORMANCE
# ════════════════════════════════════════════════════════
with tab_perf:
    st.markdown('<p class="sec">STORE-LEVEL PERFORMANCE</p>', unsafe_allow_html=True)

    # ── What these graphs mean ──────────────────────────────────
    st.markdown("""
    <div style="background:#FFFFFF;border-radius:12px;padding:1rem 1.4rem;
         box-shadow:0 1px 6px rgba(0,0,0,0.07);margin-bottom:1.2rem;
         border-left:5px solid #10B981;">
      <p style="font-size:0.85rem;color:#111827;font-weight:700;margin:0 0 5px;">
        📖 How to read the Store Performance charts</p>
      <p style="font-size:0.82rem;color:#374151;line-height:1.68;margin:0;">
        <strong>Visits vs Call Duration</strong> — Each dot is one store. The further right
        it sits, the more times it was visited; the higher it sits, the longer each visit lasted.
        Dots top-right are your most engaged stores. Dots bottom-left are stores your team
        barely reaches — these need attention.<br>
        <strong>Salespeople per Store</strong> — Shows how your sales workforce is distributed.
        If most stores have only 1 salesperson, there may be an under-resourcing issue in
        high-potential regions.<br>
        <strong>Product Variety Box Plot</strong> — The box shows the typical range of products
        stocked in stores in each cluster. A short box high up means stores in that cluster
        stock many products consistently. A box sitting low means stores carry very few products —
        a cross-sell opportunity waiting to be unlocked.
      </p>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)

    with s1:
        if ("Total_Visits" in dff.columns and "Avg_Call_Duration" in dff.columns
                and "GeoCluster" in dff.columns):
            fig_sc = go.Figure()
            for i, cl in enumerate(sorted(dff["GeoCluster"].unique())):
                sub   = dff[dff["GeoCluster"] == cl]
                # Sample for performance (up to 600 per cluster)
                sub_s = sub.sample(min(600, len(sub)), random_state=i)
                tid   = (sub_s["StoreID"].astype(str).tolist()
                         if "StoreID" in sub_s.columns else [""]*len(sub_s))
                fig_sc.add_trace(go.Scatter(
                    x=sub_s["Total_Visits"].tolist(),
                    y=sub_s["Avg_Call_Duration"].tolist(),
                    mode="markers",
                    name=f"Cluster {int(cl)}",
                    marker=dict(color=PAL[i % len(PAL)], size=5, opacity=0.6),
                    text=tid,
                    hovertemplate=("<b>%{text}</b><br>Total Visits: %{x}<br>"
                                   "Avg Call Duration: %{y} min<extra></extra>"),
                ))
            fig_layout(fig_sc, "Visits vs Call Duration per Store", h=310,
                       xtitle="Total Visits (all time)",
                       ytitle="Avg Call Duration (minutes per visit)")
            st.plotly_chart(fig_sc, use_container_width=True,
                            config={"displayModeBar": False})
            explain("Each dot is one store, coloured by its region. Stores to the far right receive "
                    "the most visits. Stores high up have longer sales calls. Stores in the bottom-left "
                    "corner are barely visited and may be missing out on sales. These are worth "
                    "prioritising in your next sales route.")

    with s2:
        if "Salesperson_Count" in dff.columns:
            dist = dff["Salesperson_Count"].value_counts().sort_index()
            fig_sp = go.Figure(go.Bar(
                x=dist.index.astype(str).tolist(),
                y=dist.values.tolist(),
                marker=dict(color="#10B981", line=dict(color="#FFFFFF", width=1.5)),
                text=dist.values.tolist(), textposition="outside",
                textfont=dict(color="#374151", size=11, family="Inter"),
                hovertemplate="<b>%{x} salesperson(s)</b>: %{y:,} stores<extra></extra>",
            ))
            fig_layout(fig_sp, "How Many Salespeople Are Assigned Per Store?", h=310,
                       xtitle="Number of Salespeople (per store)",
                       ytitle="Number of Stores",
                       xgrid=False)
            fig_sp.update_layout(bargap=0.28)
            st.plotly_chart(fig_sp, use_container_width=True,
                            config={"displayModeBar": False})
            explain("This shows how many salespeople are assigned to each store. "
                    "The tallest bar tells you the most common resourcing level. "
                    "If most stores only have 1 salesperson, consider whether adding "
                    "more staff to high-volume stores could meaningfully boost sales.")

    with s3:
        if "_n_products" in dff.columns and "GeoCluster" in dff.columns:
            fig_bx = go.Figure()
            for i, cl in enumerate(sorted(dff["GeoCluster"].unique())):
                sub = dff[dff["GeoCluster"] == cl]["_n_products"]
                fig_bx.add_trace(go.Box(
                    y=sub.tolist(),
                    name=f"Cluster {int(cl)}",
                    marker_color=PAL[i % len(PAL)],
                    boxmean="sd",
                    hovertemplate=(f"Cluster {int(cl)}<br>"
                                   "Products stocked: %{y} products<extra></extra>"),
                ))
            fig_layout(fig_bx, "Product Variety per Store by Cluster", h=310,
                       xtitle="Geographic Cluster",
                       ytitle="Number of Products Stocked (per store)",
                       xgrid=False, showlegend=False)
            st.plotly_chart(fig_bx, use_container_width=True,
                            config={"displayModeBar": False})
            explain("Each box represents one cluster (region). The middle line inside the box "
                    "is the typical number of products a store in that region stocks. "
                    "Clusters with low boxes are under-stocked regions — "
                    "introducing more products there is a direct revenue growth lever.")

    # ── Store Profile Drill-Down ────────────────────────────────
    if "StoreID" in dff.columns:
        st.markdown('<p class="sec">🔍 INDIVIDUAL STORE PROFILE</p>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#FFFFFF;border-radius:12px;padding:0.9rem 1.3rem;
             box-shadow:0 1px 6px rgba(0,0,0,0.07);margin-bottom:1rem;
             border-left:5px solid #8B5CF6;font-size:0.82rem;color:#374151;">
            Select any store from the dropdown below to instantly see its full profile —
            visit history, products stocked, units sold, and how it compares to
            the average store in its cluster.
        </div>
        """, unsafe_allow_html=True)

        all_store_ids = sorted(dff["StoreID"].astype(str).unique().tolist())
        sel_store = st.selectbox(
            "Select a Store ID to view its full profile",
            all_store_ids,
            key="store_profile_select",
        )

        store_row = dff[dff["StoreID"].astype(str) == sel_store].iloc[0]

        # ── Cluster peers for comparison ──
        cl_val = int(store_row["GeoCluster"]) if "GeoCluster" in store_row.index else None
        peers  = dff[dff["GeoCluster"] == cl_val] if cl_val is not None else dff

        # ── Helper: compare vs cluster average ──
        def vs_avg(val, avg, unit="", higher_is_better=True):
            diff  = val - avg
            pct   = (diff / avg * 100) if avg != 0 else 0
            up    = diff >= 0
            good  = up if higher_is_better else not up
            arrow = "▲" if up else "▼"
            color = "#10B981" if good else "#EF4444"
            label = "above" if up else "below"
            return (f'<span style="color:{color};font-weight:700;font-size:0.78rem">'
                    f'{arrow} {abs(round(pct))}% {label} cluster avg</span>')

        # ── Profile header ──
        cl_name  = f"Cluster {cl_val}" if cl_val is not None else "—"
        cl_color = PAL[cl_val % len(PAL)] if cl_val is not None else "#6B7280"
        n_prods  = int(store_row["_n_products"]) if "_n_products" in store_row.index else 0
        tot_u    = int(store_row["_total_qty"])   if "_total_qty"  in store_row.index else 0
        visits   = int(store_row["Total_Visits"]) if "Total_Visits" in store_row.index else 0
        dur      = round(store_row["Avg_Call_Duration"]) if "Avg_Call_Duration" in store_row.index else 0
        sales_p  = int(store_row["Salesperson_Count"]) if "Salesperson_Count" in store_row.index else 0
        lat      = round(store_row["StoreLatitude"],  4) if "StoreLatitude"  in store_row.index else "—"
        lon      = round(store_row["StoreLongitude"], 4) if "StoreLongitude" in store_row.index else "—"

        avg_visits = round(peers["Total_Visits"].mean())       if "Total_Visits" in peers.columns else 0
        avg_dur    = round(peers["Avg_Call_Duration"].mean())  if "Avg_Call_Duration" in peers.columns else 0
        avg_prods  = round(peers["_n_products"].mean())        if "_n_products" in peers.columns else 0
        avg_qty    = round(peers["_total_qty"].mean())         if "_total_qty"  in peers.columns else 0

        # Rank of this store within dff by total qty
        rank_val = int((dff["_total_qty"] > tot_u).sum()) + 1 if "_total_qty" in dff.columns else "—"
        total_stores_dff = len(dff)

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1E3A5F,#2563EB);
             border-radius:14px;padding:1.3rem 1.8rem;margin-bottom:1rem;
             display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;">
          <div style="font-size:2.5rem">🏪</div>
          <div>
            <div style="font-size:1.3rem;font-weight:800;color:#FFFFFF">{sel_store}</div>
            <div style="margin-top:4px">
              <span style="background:{cl_color};color:#FFFFFF;padding:2px 12px;
                border-radius:20px;font-size:0.72rem;font-weight:700">
                {cl_name}
              </span>
              <span style="color:#BFDBFE;font-size:0.8rem;margin-left:10px">
                📍 {lat}, {lon}
              </span>
              <span style="color:#BFDBFE;font-size:0.8rem;margin-left:10px">
                🏆 Rank #{rank_val} of {total_stores_dff:,} stores
              </span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 4 mini KPIs ──
        k1, k2, k3, k4 = st.columns(4)
        def store_kpi(col, icon, label, val, unit, avg, higher_is_better=True):
            cmp = vs_avg(val, avg, unit, higher_is_better)
            with col:
                st.markdown(f"""
                <div style="background:#FFFFFF;border-radius:12px;padding:1rem 1.2rem;
                     box-shadow:0 1px 6px rgba(0,0,0,0.07);
                     border-top:4px solid #8B5CF6;">
                  <div style="font-size:0.66rem;font-weight:700;text-transform:uppercase;
                       letter-spacing:0.9px;color:#6B7280;margin-bottom:4px">{icon} {label}</div>
                  <div style="font-size:1.7rem;font-weight:800;color:#111827;line-height:1.1">
                    {val} <span style="font-size:0.85rem;font-weight:500;color:#6B7280">{unit}</span>
                  </div>
                  <div style="margin-top:4px">{cmp}</div>
                  <div style="font-size:0.7rem;color:#9CA3AF;margin-top:2px">
                    Cluster avg: {avg} {unit}
                  </div>
                </div>""", unsafe_allow_html=True)

        store_kpi(k1, "📞", "Total Visits",    visits,  "visits", avg_visits)
        store_kpi(k2, "⏱️", "Avg Call Duration", dur,   "min",    avg_dur)
        store_kpi(k3, "📦", "Products Stocked", n_prods, "",      avg_prods)
        store_kpi(k4, "🛒", "Total Units Sold", tot_u,  "units",  avg_qty)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Products this store stocks ──
        left_p, right_p = st.columns([1, 1.4])

        with left_p:
            st.markdown("""
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                 letter-spacing:1px;color:#6B7280;margin-bottom:0.7rem">
                 PRODUCTS STOCKED & UNITS SOLD
            </div>""", unsafe_allow_html=True)

            stocked, not_stocked = [], []
            for p in prod_cols:
                flag = int(store_row[p]) if p in store_row.index else 0
                qty  = int(store_row[f"{p}_Qty"]) if f"{p}_Qty" in store_row.index else 0
                if flag:
                    stocked.append((p, qty))
                else:
                    not_stocked.append(p)

            if stocked:
                for p, qty in sorted(stocked, key=lambda x: -x[1]):
                    bar_w = int(qty / max(q for _, q in stocked) * 100) if stocked else 0
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:10px;
                         margin-bottom:6px;">
                      <div style="min-width:90px;font-size:0.82rem;font-weight:600;
                           color:#111827">{p}</div>
                      <div style="flex:1;background:#F3F4F6;border-radius:4px;height:10px;">
                        <div style="width:{bar_w}%;height:100%;border-radius:4px;
                             background:{PROD_COL.get(p,'#2563EB')}"></div>
                      </div>
                      <div style="min-width:60px;font-size:0.8rem;font-weight:700;
                           color:#374151;text-align:right">{qty:,} units</div>
                    </div>""", unsafe_allow_html=True)

            if not_stocked:
                st.markdown(f"""
                <div style="margin-top:0.8rem;padding:0.65rem 0.9rem;
                     background:#FEF2F2;border-radius:8px;border-left:4px solid #EF4444;">
                  <div style="font-size:0.72rem;font-weight:700;color:#DC2626;
                       margin-bottom:4px">❌ NOT STOCKED — Cross-sell Opportunities</div>
                  <div style="font-size:0.8rem;color:#374151">
                    {' &nbsp;·&nbsp; '.join(not_stocked)}
                  </div>
                </div>""", unsafe_allow_html=True)

        with right_p:
            # Radar chart — store vs cluster average
            st.markdown("""
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                 letter-spacing:1px;color:#6B7280;margin-bottom:0.7rem">
                 STORE vs CLUSTER AVERAGE (UNITS SOLD)
            </div>""", unsafe_allow_html=True)

            radar_prods = [p for p in prod_cols if f"{p}_Qty" in dff.columns]
            if radar_prods:
                store_vals   = [int(store_row.get(f"{p}_Qty", 0)) for p in radar_prods]
                cluster_vals = [round(peers[f"{p}_Qty"].mean()) for p in radar_prods]
                # Close the radar loop
                cats = radar_prods + [radar_prods[0]]
                sv   = store_vals  + [store_vals[0]]
                cv   = cluster_vals + [cluster_vals[0]]

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=cv, theta=cats, fill="toself",
                    name="Cluster Average",
                    line=dict(color="#93C5FD", width=2),
                    fillcolor="rgba(147,197,253,0.15)",
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=sv, theta=cats, fill="toself",
                    name=sel_store,
                    line=dict(color="#8B5CF6", width=2.5),
                    fillcolor="rgba(139,92,246,0.18)",
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="#FFFFFF",
                        radialaxis=dict(
                            visible=True, showticklabels=True,
                            tickfont=dict(size=9, color="#6B7280"),
                            gridcolor="#F3F4F6",
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=10, color="#374151"),
                            gridcolor="#E5E7EB",
                        ),
                    ),
                    showlegend=True,
                    legend=dict(font=dict(size=11, color="#111827"),
                                bgcolor="#FFFFFF", bordercolor="#E5E7EB", borderwidth=1),
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    height=320,
                    margin=dict(l=40, r=40, t=20, b=20),
                    font=dict(family="Inter", color="#374151"),
                )
                st.plotly_chart(fig_radar, use_container_width=True,
                                config={"displayModeBar": False})
                explain("The purple shape is this store. The blue shape is the average for its cluster. "
                        "Wherever the purple is <strong>inside</strong> the blue, that product is "
                        "underperforming vs the region — a direct action target for your sales team.")

        # ── Opportunity summary ──
        missing_revenue_prods = not_stocked[:3] if not_stocked else []
        under_prods = [(p, round(peers[f"{p}_Qty"].mean()), int(store_row.get(f"{p}_Qty",0)))
                       for p in prod_cols if f"{p}_Qty" in dff.columns
                       and int(store_row.get(f"{p}_Qty",0)) < round(peers[f"{p}_Qty"].mean()) * 0.5
                       and int(store_row.get(p, 0)) == 1]
        under_prods = sorted(under_prods, key=lambda x: x[1]-x[2], reverse=True)[:3]

        st.markdown('<p class="sec">💡 ACTION PLAN FOR THIS STORE</p>', unsafe_allow_html=True)

        action_html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">'

        if missing_revenue_prods:
            prods_str = ", ".join(missing_revenue_prods)
            action_html += f"""
            <div style="background:#FEF2F2;border-radius:10px;padding:0.9rem 1.1rem;
                 border-left:5px solid #EF4444;">
              <div style="font-weight:700;font-size:0.85rem;color:#DC2626;margin-bottom:4px">
                🚀 Introduce New Products</div>
              <div style="font-size:0.8rem;color:#374151;line-height:1.6">
                This store does not stock <strong>{prods_str}</strong>.
                Other stores in {cl_name} sell these successfully.
                Introducing even one could meaningfully boost revenue.
              </div>
            </div>"""

        if under_prods:
            up_str = ", ".join([f"{p} (selling {s} vs avg {a})" for p,a,s in under_prods])
            action_html += f"""
            <div style="background:#FFFBEB;border-radius:10px;padding:0.9rem 1.1rem;
                 border-left:5px solid #F59E0B;">
              <div style="font-weight:700;font-size:0.85rem;color:#D97706;margin-bottom:4px">
                📈 Boost Under-selling Products</div>
              <div style="font-size:0.8rem;color:#374151;line-height:1.6">
                <strong>{up_str}</strong> — selling well below the cluster average.
                Check shelf placement, expiry, and ask if the store needs extra stock.
              </div>
            </div>"""

        if visits < avg_visits:
            action_html += f"""
            <div style="background:#EFF6FF;border-radius:10px;padding:0.9rem 1.1rem;
                 border-left:5px solid #2563EB;">
              <div style="font-weight:700;font-size:0.85rem;color:#1D4ED8;margin-bottom:4px">
                📞 Increase Visit Frequency</div>
              <div style="font-size:0.8rem;color:#374151;line-height:1.6">
                This store receives <strong>{visits} visits</strong> vs the cluster average of
                <strong>{avg_visits} visits</strong>. More visits = more shelf space negotiated
                = more sales. Prioritise this store on the next route.
              </div>
            </div>"""

        if n_prods >= len(prod_cols) * 0.8 and tot_u > avg_qty:
            action_html += f"""
            <div style="background:#ECFDF5;border-radius:10px;padding:0.9rem 1.1rem;
                 border-left:5px solid #10B981;">
              <div style="font-weight:700;font-size:0.85rem;color:#059669;margin-bottom:4px">
                ⭐ High-Value Store — Protect & Grow</div>
              <div style="font-size:0.8rem;color:#374151;line-height:1.6">
                This store stocks <strong>{n_prods} products</strong> and sells
                <strong>{tot_u:,} units</strong> — well above average.
                Prioritise relationship management and ensure no stockouts.
              </div>
            </div>"""

        action_html += '</div>'
        st.markdown(action_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Top / Bottom stores
    if "_total_qty" in dff.columns and "StoreID" in dff.columns:
        st.markdown('<p class="sec">TOP & BOTTOM PERFORMING STORES</p>',
                    unsafe_allow_html=True)
        show   = [c for c in ["StoreID","GeoCluster","Total_Visits",
                               "Salesperson_Count","_n_products","_total_qty"]
                  if c in dff.columns]
        rename = {"StoreID":"Store ID","GeoCluster":"Cluster",
                  "Total_Visits":"Visits","Salesperson_Count":"Salespeople",
                  "_n_products":"Products Stocked","_total_qty":"Total Units"}
        top_df = dff.nlargest(10,  "_total_qty")[show].rename(columns=rename).reset_index(drop=True)
        bot_df = dff.nsmallest(10, "_total_qty")[show].rename(columns=rename).reset_index(drop=True)
        tc, bc = st.columns(2)
        with tc:
            st.markdown("**🏆 Top 10 Stores — Highest Units Sold**")
            st.dataframe(top_df, use_container_width=True, hide_index=True, height=300)
        with bc:
            st.markdown("**⚠️ Bottom 10 Stores — Needs Attention**")
            st.dataframe(bot_df, use_container_width=True, hide_index=True, height=300)
        explain("The top table shows your 10 best-performing stores by total units sold — "
                "study what they do right (visits, product range, location) and replicate it elsewhere. "
                "The bottom table shows your 10 weakest stores. These are your highest-priority "
                "intervention targets: increase visit frequency, introduce more products, "
                "or investigate whether they need additional sales support.")

# ════════════════════════════════════════════════════════
# TAB 6 — AI ADVISOR
# ════════════════════════════════════════════════════════
with tab_ai:
    st.markdown('<p class="sec">🤖 AI CROSS-SELL ADVISOR</p>', unsafe_allow_html=True)

    api_key = GEMINI_API_KEY.strip()

    # ── Intro card ──────────────────────────────────────
    st.markdown("""
    <div style="background:#FFFFFF;border-radius:12px;padding:1.1rem 1.5rem;
         box-shadow:0 1px 6px rgba(0,0,0,0.07);margin-bottom:1.2rem;
         border-left:5px solid #8B5CF6;">
      <p style="font-size:0.88rem;font-weight:700;color:#111827;margin:0 0 5px">
        🧠 Your AI-Powered Sales Strategy Analyst</p>
      <p style="font-size:0.82rem;color:#374151;line-height:1.68;margin:0">
        This advisor reads your actual store data — products, clusters, visit patterns,
        and cross-sell rules — and gives you <strong>specific, actionable
        recommendations</strong> in plain English. Ask anything about your data,
        request a strategy for a specific product or cluster, or get a full
        cross-sell action plan. No technical knowledge needed.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Build data context from live dataset ────────────
    def build_data_context():
        """Summarise the loaded dataset into a compact prompt context."""
        lines = []

        # Overview
        lines.append(f"DATASET OVERVIEW")
        lines.append(f"- Total stores: {len(dff):,}")
        lines.append(f"- Geographic clusters: {sorted(dff['GeoCluster'].unique().tolist()) if 'GeoCluster' in dff.columns else 'N/A'}")
        lines.append(f"- Products tracked: {', '.join(prod_cols)}")

        # Product presence
        if prod_cols:
            lines.append(f"\nPRODUCT PRESENCE (% of stores stocking each product):")
            pcts = (dff[prod_cols].sum() / len(dff) * 100).round(0).astype(int).sort_values(ascending=False)
            for p, v in pcts.items():
                lines.append(f"  {p}: {v}%")

        # Units sold
        if qty_cols:
            lines.append(f"\nTOTAL UNITS SOLD:")
            qtots = dff[qty_cols].sum().sort_values(ascending=False)
            qtots.index = [c.replace("_Qty","") for c in qtots.index]
            for p, v in qtots.items():
                lines.append(f"  {p}: {int(v):,} units")

        # Cluster breakdown
        if "GeoCluster" in dff.columns:
            lines.append(f"\nCLUSTER BREAKDOWN:")
            for cl in sorted(dff["GeoCluster"].unique()):
                sub = dff[dff["GeoCluster"] == cl]
                avg_v = round(sub["Total_Visits"].mean()) if "Total_Visits" in sub.columns else "N/A"
                avg_d = round(sub["Avg_Call_Duration"].mean()) if "Avg_Call_Duration" in sub.columns else "N/A"
                lines.append(f"  Cluster {int(cl)}: {len(sub):,} stores, "
                             f"avg {avg_v} visits, avg {avg_d} min call duration")

        # Top cross-sell rules
        rules_df = st.session_state.get("rules")
        if rules_df is not None and not rules_df.empty:
            lines.append(f"\nTOP 15 CROSS-SELL RULES (by Lift):")
            top_rules = rules_df.nlargest(15, "lift")[["from_str","to_str","confidence","lift","support"]]
            for _, row in top_rules.iterrows():
                conf = round(row["confidence"] * 100)
                lift = round(row["lift"], 1)
                sup  = round(row["support"]  * 100)
                lines.append(f"  {row['from_str']} → {row['to_str']}: "
                             f"{conf}% confidence, {lift}x lift, {sup}% stores")
        else:
            lines.append("\nCROSS-SELL RULES: Not yet generated. "
                        "User should run analysis on Cross-Sell tab first.")

        # Visit stats
        if "Total_Visits" in dff.columns:
            lines.append(f"\nVISIT STATS:")
            lines.append(f"  Average visits per store: {round(dff['Total_Visits'].mean())}")
            lines.append(f"  Max visits: {int(dff['Total_Visits'].max())}")
            lines.append(f"  Stores with only 1 visit: {int((dff['Total_Visits']==1).sum()):,}")

        return "\n".join(lines)

    SYSTEM_PROMPT = """You are an expert retail sales strategy analyst specialising in 
FMCG (fast-moving consumer goods) cross-sell and distribution optimisation.

You have been given a summary of a real retail store dataset. Your job is to provide 
clear, specific, actionable advice that a non-technical sales manager or business 
executive can immediately act on.

Rules:
- Always refer to specific products, cluster numbers, and percentages from the data
- Give concrete action steps, not generic advice
- Keep language simple — no jargon, no technical terms
- Format responses with clear headings and bullet points
- When recommending promotions, be specific about which products, which clusters, 
  and what the expected uplift could be
- If the user asks something not covered by the data, say so clearly"""

    # ── Preset quick-action buttons ─────────────────────
    st.markdown('<p class="sec">⚡ QUICK ANALYSIS — CLICK ANY BUTTON TO GET INSTANT INSIGHTS</p>',
                unsafe_allow_html=True)

    q_cols = st.columns(4)
    quick_prompts = {
        "🏆 Top Cross-Sell Opportunities":
            "Based on the cross-sell rules and product data, what are the top 5 most actionable "
            "cross-sell opportunities I should focus on right now? For each one, tell me exactly "
            "which products to pair, which clusters to target, and what action to take.",
        "📍 Worst-Performing Clusters":
            "Which clusters have the lowest product variety and fewest visits? "
            "What specific steps should the sales team take to improve performance in those clusters?",
        "📦 Underutilised Products":
            "Which products are stocked in fewer stores than they should be, "
            "and which clusters should we introduce them to first? Give me a prioritised rollout plan.",
        "🗓️ Sales Visit Strategy":
            "Based on the visit data, which stores are being under-visited? "
            "Suggest a prioritised visit schedule for the sales team to maximise cross-sell potential.",
    }

    selected_quick = None
    for i, (label, prompt) in enumerate(quick_prompts.items()):
        with q_cols[i]:
            if st.button(label, use_container_width=True, key=f"qbtn_{i}"):
                selected_quick = prompt

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chat interface ───────────────────────────────────
    st.markdown('<p class="sec">💬 ASK THE AI ADVISOR ANYTHING ABOUT YOUR DATA</p>',
                unsafe_allow_html=True)

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["ai_messages"]:
            role  = msg["role"]
            text  = msg["content"]
            if role == "user":
                st.markdown(
                    f'<div style="background:#EFF6FF;border-radius:10px 10px 2px 10px;'
                    f'padding:0.75rem 1rem;margin:0.4rem 0 0.4rem 20%;'
                    f'font-size:0.85rem;color:#1E3A5F;border:1px solid #BFDBFE">'
                    f'<strong>You:</strong> {text}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="background:#FFFFFF;border-radius:10px 10px 10px 2px;'
                    f'padding:0.75rem 1rem;margin:0.4rem 20% 0.4rem 0;'
                    f'font-size:0.85rem;color:#111827;border:1px solid #E5E7EB;'
                    f'box-shadow:0 1px 4px rgba(0,0,0,0.06)">'
                    f'<strong>🤖 AI Advisor:</strong><br>{text}</div>',
                    unsafe_allow_html=True
                )

    # Input area
    inp_col, btn_col = st.columns([5, 1])
    with inp_col:
        user_input = st.text_input(
            "Ask a question about your data",
            placeholder="e.g. Which cluster should I focus on for Milk cross-sell? "
                        "Which stores should I visit first?",
            label_visibility="collapsed",
            key="ai_input",
        )
    with btn_col:
        send_btn = st.button("Send ▶", use_container_width=True, key="ai_send")

    # Clear chat
    if st.button("🗑️ Clear conversation", key="ai_clear"):
        st.session_state["ai_messages"] = []
        st.rerun()

    # ── Process query (quick button or typed) ────────────
    query = selected_quick or (user_input if send_btn and user_input.strip() else None)

    if query:
        data_ctx = build_data_context()
        full_user_msg = (
            f"Here is a summary of the retail store dataset:\n\n"
            f"{data_ctx}\n\n"
            f"---\n\n"
            f"My question: {query}"
        )

        # Add user message to history (show only the question, not the data context)
        st.session_state["ai_messages"].append({"role": "user", "content": query})

        with st.spinner("AI Advisor is analysing your data…"):
            try:
                import requests, re
                # Gemini free-tier endpoint — v1beta with gemini-2.5-flash
                url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

                # Build conversation in Gemini format (user / model roles only)
                gemini_messages = []

                # Inject system instructions as opening user/model exchange
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": SYSTEM_PROMPT}]
                })
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": "Understood. I am ready to analyse the retail store data and provide specific, actionable recommendations in plain English."}]
                })

                # Add prior conversation turns (last 4 pairs for context)
                history = st.session_state["ai_messages"][:-1]
                for m in history[-4:]:
                    g_role = "user" if m["role"] == "user" else "model"
                    gemini_messages.append({
                        "role":  g_role,
                        "parts": [{"text": m["content"]}]
                    })

                # Add the current message (includes full data context)
                gemini_messages.append({
                    "role":  "user",
                    "parts": [{"text": full_user_msg}]
                })

                resp = requests.post(
                    url,
                    headers={
                        "Content-Type":  "application/json",
                        "x-goog-api-key": api_key,
                    },
                    json={
                        "contents": gemini_messages,
                        "generationConfig": {
                            "maxOutputTokens": 8192,
                            "temperature":     0.4,
                        },
                    },
                    timeout=90,
                )
                data = resp.json()

                # Parse Gemini response
                try:
                    reply = data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    err = data.get("error", {}).get("message", str(data))
                    raise ValueError(f"API error: {err}")

                # Convert markdown bold/bullets → HTML for chat bubble
                reply_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', reply)
                reply_html = re.sub(r'\*(.*?)\*',     r'<em>\1</em>',         reply_html)
                reply_html = reply_html.replace("\n\n", "<br><br>")
                reply_html = reply_html.replace("\n- ", "<br>• ")
                reply_html = reply_html.replace("\n• ", "<br>• ")
                reply_html = reply_html.replace("\n# ",  "<br><strong>")
                reply_html = reply_html.replace("\n## ", "<br><strong>")
                st.session_state["ai_messages"].append(
                    {"role": "assistant", "content": reply_html}
                )

            except Exception as e:
                st.session_state["ai_messages"].append(
                    {"role": "assistant",
                     "content": (
                         f"⚠️ Could not get a response: {str(e)}<br><br>"
                         "Please check that your Gemini API key is correct in the "
                         "<code>GEMINI_API_KEY</code> variable at the top of the file."
                     )}
                )
        st.rerun()

    # ── Tips ─────────────────────────────────────────────
    if not st.session_state["ai_messages"]:
        st.markdown("""
        <div style="background:#F9FAFB;border-radius:10px;padding:1rem 1.4rem;
             border:1px dashed #D1D5DB;margin-top:0.5rem">
          <p style="font-size:0.82rem;font-weight:700;color:#374151;margin:0 0 6px">
            💡 Example questions you can ask:</p>
          <ul style="font-size:0.8rem;color:#6B7280;margin:0;padding-left:1.2rem;line-height:1.9">
            <li>Which product has the best cross-sell potential in Cluster 2?</li>
            <li>Create a 4-week sales action plan for my team to boost Ghee sales</li>
            <li>Which 3 stores should I visit first next week and why?</li>
            <li>Why is IceCream underperforming and what should I do about it?</li>
            <li>Compare Cluster 0 and Cluster 4 — where should I invest more?</li>
            <li>What combo deals should I offer to increase average basket size?</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    explain("The AI Advisor reads your actual loaded data — product distribution, "
            "cross-sell rules, cluster performance, and visit patterns — and gives you "
            "specific, plain-English recommendations. Use the quick buttons for instant "
            "analysis, or type any question in your own words. The more the cross-sell "
            "analysis has been run (on the Cross-Sell tab), the richer the advice will be.")
st.markdown("""
<hr style="border:none;border-top:1px solid #E5E7EB;margin:2rem 0 1rem">
<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px">
  <span style="color:#9CA3AF;font-size:0.73rem">
    📊 Retail Cross-Sell Intelligence Platform · Market Basket Analysis
  </span>
  <span style="color:#9CA3AF;font-size:0.73rem">
    Streamlit · Plotly · mlxtend · NetworkX
  </span>
</div>
""", unsafe_allow_html=True)
