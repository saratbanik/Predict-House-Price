"""Responsive Streamlit interface for the trained house-price regression model.

Run from the project directory with: streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "house_price_model.joblib"

st.set_page_config(
    page_title="House Price Studio | Prediction",
    page_icon="⌂",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# The UI is intentionally self-contained so no other project files or assets are needed.
st.markdown(
    """
    <style>
    :root {
      --ink: #eef4ff;
      --muted: #9aa9c4;
      --line: rgba(164, 184, 220, .16);
      --panel: rgba(17, 27, 49, .78);
      --panel-strong: rgba(20, 33, 59, .94);
      --cyan: #64e3d1;
      --blue: #8ba7ff;
      --violet: #b2a0ff;
    }

    html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    [data-testid="stAppViewContainer"] {
      color: var(--ink);
      background:
        radial-gradient(ellipse at 8% 2%, rgba(71, 91, 190, .24), transparent 31%),
        radial-gradient(ellipse at 92% 12%, rgba(35, 177, 170, .13), transparent 28%),
        linear-gradient(145deg, #080d18 0%, #0b1222 48%, #0a1020 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1.5rem; }
    .main .block-container { max-width: 1240px; padding: 2.2rem 2rem 3.5rem; }
    p, label, span { letter-spacing: -.01em; }

    .hero-shell {
      position: relative; overflow: hidden; padding: 1.8rem 2rem 2rem;
      border: 1px solid rgba(151, 176, 255, .22); border-radius: 26px;
      background: linear-gradient(120deg, rgba(24, 38, 72, .96), rgba(18, 29, 54, .88) 58%, rgba(18, 53, 67, .82));
      box-shadow: 0 24px 70px rgba(0, 0, 0, .22);
      margin-bottom: 1.4rem;
    }
    .hero-shell:before {
      content: ""; position: absolute; width: 330px; height: 330px; border-radius: 50%;
      right: -90px; top: -175px; background: radial-gradient(circle, rgba(100, 227, 209, .18), transparent 68%);
      pointer-events: none;
    }
    .hero-topline { display:flex; align-items:center; gap:.8rem; position:relative; z-index:1; }
    .brand-mark { display:flex; align-items:center; justify-content:center; width:44px; height:44px; border-radius:15px;
      color:#061722; font-size:1.8rem; font-weight:900; background:linear-gradient(135deg, var(--cyan), #9eb5ff); box-shadow:0 8px 26px rgba(100,227,209,.17); }
    .brand-copy { display:flex; flex-direction:column; gap:.18rem; }
    .brand-copy strong { color:#f3f6ff; font-size:.94rem; letter-spacing:.07em; }
    .brand-copy small { color:var(--muted); font-size:.77rem; }
    .status-pill { margin-left:auto; display:inline-flex; align-items:center; gap:.45rem; padding:.42rem .7rem; border-radius:999px;
      border:1px solid rgba(100,227,209,.27); background:rgba(100,227,209,.08); color:#aaf5e9; font-size:.72rem; font-weight:700; letter-spacing:.07em; white-space:nowrap; }
    .status-dot { width:7px; height:7px; border-radius:50%; background:var(--cyan); box-shadow:0 0 0 4px rgba(100,227,209,.10); }
    .hero-grid { display:grid; grid-template-columns:minmax(0,1.5fr) minmax(230px,.8fr); gap:2rem; align-items:end; margin-top:2.15rem; position:relative; z-index:1; }
    .hero-eyebrow { color:var(--cyan); text-transform:uppercase; font-weight:800; letter-spacing:.14em; font-size:.72rem; margin-bottom:.65rem; }
    .hero-title { color:#f4f7ff; font-size:clamp(2rem, 4.4vw, 3.6rem); font-weight:800; line-height:1.05; letter-spacing:-.055em; margin:0; }
    .hero-title span { color:var(--cyan); }
    .hero-copy { color:#b3bfd7; max-width:590px; line-height:1.65; font-size:.98rem; margin:.95rem 0 0; }
    .hero-info { padding:1.15rem 1.2rem; border-radius:18px; border:1px solid rgba(175,191,230,.18); background:rgba(4,11,25,.28); }
    .hero-info-label { color:#8fa0c2; font-size:.68rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
    .hero-info-title { color:#f3f7ff; font-size:1.1rem; font-weight:750; margin:.45rem 0 .3rem; }
    .hero-info-copy { color:#9eacc7; font-size:.8rem; line-height:1.55; }
    .top-stats { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.8rem; margin:0 0 1.6rem; }
    .top-stat { border:1px solid var(--line); background:rgba(19,29,51,.55); padding:.9rem 1rem; border-radius:15px; display:flex; gap:.8rem; align-items:center; }
    .top-stat-icon { width:35px; height:35px; display:flex; flex-shrink:0; align-items:center; justify-content:center; border-radius:11px; background:rgba(139,167,255,.12); color:#aebfff; font-size:1rem; }
    .top-stat strong { display:block; color:#eff4ff; font-size:.86rem; }
    .top-stat small { display:block; color:#93a2bd; font-size:.73rem; margin-top:.16rem; }

    [data-testid="stVerticalBlockBorderWrapper"] { background:linear-gradient(155deg, rgba(20,31,54,.88), rgba(13,22,41,.82)); border:1px solid var(--line); border-radius:22px; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { border-radius:22px; }
    .panel-kicker { color:var(--cyan); font-size:.68rem; letter-spacing:.14em; font-weight:800; text-transform:uppercase; margin:0 0 .4rem; }
    .panel-title { color:#f0f4ff; font-size:1.34rem; font-weight:750; letter-spacing:-.035em; margin:0; }
    .panel-copy { color:#9aa9c4; font-size:.84rem; line-height:1.55; margin:.45rem 0 1.1rem; }
    .preset-label { color:#aab7d0; font-weight:700; font-size:.76rem; margin:.2rem 0 .45rem; }
    div[data-testid="stButton"] button { min-height:2.45rem; border:1px solid rgba(145,166,211,.22); border-radius:11px; color:#dce6fb; background:rgba(115,139,194,.07); font-size:.78rem; font-weight:650; transition:all .18s ease; }
    div[data-testid="stButton"] button:hover { border-color:rgba(100,227,209,.65); color:#baf7ee; background:rgba(100,227,209,.08); transform:translateY(-1px); }
    div[data-testid="stFormSubmitButton"] button, button[kind="primary"] { border:0 !important; border-radius:12px !important; color:#071520 !important; font-weight:800 !important; background:linear-gradient(105deg, #64e3d1, #9eb5ff) !important; min-height:3rem !important; box-shadow:0 8px 25px rgba(100,227,209,.14); }
    div[data-testid="stFormSubmitButton"] button:hover, button[kind="primary"]:hover { filter:brightness(1.07); transform:translateY(-1px); }
    [data-testid="stNumberInput"] label { color:#d5deef !important; font-weight:650 !important; font-size:.82rem !important; }
    [data-testid="stNumberInput"] input { color:#f1f5ff !important; }
    [data-testid="stNumberInput"] div[data-baseweb="input"] { border:1px solid rgba(143,163,207,.20); border-radius:11px; background:rgba(5,12,25,.43); }
    [data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within { border-color:rgba(100,227,209,.75); box-shadow:0 0 0 2px rgba(100,227,209,.08); }
    [data-testid="stNumberInput"] button { background:transparent !important; color:#a7b8d6 !important; border:0 !important; }
    [data-testid="stCaptionContainer"] p { color:#8f9fbb; font-size:.75rem; line-height:1.5; }

    .result-card { min-height:320px; display:flex; flex-direction:column; justify-content:space-between; overflow:hidden; position:relative; padding:1.55rem; border-radius:20px;
      border:1px solid rgba(100,227,209,.22); background:radial-gradient(circle at 100% 0%, rgba(100,227,209,.14), transparent 42%), linear-gradient(145deg, rgba(18,48,59,.83), rgba(17,29,53,.95)); }
    .result-card:after { content:"$"; position:absolute; right:1.2rem; top:1.3rem; color:rgba(181,245,235,.055); font-weight:900; font-size:9rem; line-height:1; pointer-events:none; }
    .result-label { color:#9baec9; font-size:.73rem; text-transform:uppercase; letter-spacing:.12em; font-weight:800; }
    .result-amount { position:relative; z-index:1; color:#f7fbff; font-size:clamp(2.1rem,3.4vw,3.1rem); line-height:1.15; font-weight:850; letter-spacing:-.055em; margin:.7rem 0 .2rem; overflow-wrap:anywhere; }
    .currency-tag { display:inline-flex; padding:.3rem .54rem; border-radius:7px; color:#aaf5e9; background:rgba(100,227,209,.09); border:1px solid rgba(100,227,209,.2); font-size:.7rem; font-weight:800; letter-spacing:.08em; vertical-align:middle; margin-left:.4rem; }
    .result-caption { color:#99a9c4; font-size:.79rem; line-height:1.55; margin-top:.6rem; }
    .result-divider { height:1px; background:rgba(157,180,220,.14); margin:1.15rem 0; }
    .result-detail-label { color:#91a1bd; font-size:.68rem; text-transform:uppercase; letter-spacing:.1em; font-weight:800; margin-bottom:.5rem; }
    .result-chips { display:flex; gap:.45rem; flex-wrap:wrap; }
    .result-chip { color:#dce7fc; background:rgba(141,165,220,.09); border:1px solid rgba(141,165,220,.14); border-radius:8px; padding:.42rem .55rem; font-size:.72rem; }
    .empty-result { min-height:320px; border:1px dashed rgba(143,167,215,.28); border-radius:20px; padding:1.55rem; display:flex; flex-direction:column; align-items:flex-start; justify-content:center; background:linear-gradient(145deg, rgba(24,36,61,.54), rgba(14,24,44,.54)); }
    .empty-icon { display:flex; align-items:center; justify-content:center; width:54px; height:54px; border-radius:16px; font-size:1.45rem; color:#9fb4ff; background:rgba(139,167,255,.12); margin-bottom:1.1rem; }
    .empty-title { color:#edf3ff; font-size:1.28rem; font-weight:750; }
    .empty-copy { color:#9baac5; font-size:.84rem; line-height:1.65; margin-top:.5rem; max-width:360px; }
    .metric-card { padding:.9rem; min-height:115px; border:1px solid var(--line); background:rgba(17,27,49,.68); border-radius:15px; }
    .metric-label { color:#95a6c2; font-size:.7rem; font-weight:750; letter-spacing:.08em; text-transform:uppercase; }
    .metric-value { color:#eef4ff; font-size:1.26rem; letter-spacing:-.035em; font-weight:800; margin-top:.45rem; }
    .metric-note { color:#8292af; font-size:.68rem; line-height:1.45; margin-top:.3rem; }
    .section-title { color:#eaf1ff; font-size:1.12rem; font-weight:750; letter-spacing:-.025em; margin:.35rem 0 .3rem; }
    .section-subtitle { color:#96a6c2; font-size:.83rem; margin-bottom:1rem; }
    [data-testid="stTab"] { color:#abbad5; }
    [data-testid="stTab"][aria-selected="true"] { color:#8df0df; }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color:var(--cyan); }
    [data-testid="stExpander"] { border-color:var(--line); background:rgba(14,23,42,.5); border-radius:14px; }
    a { color:var(--cyan) !important; }

    @media (max-width: 900px) {
      .main .block-container { padding:1.25rem 1rem 2.3rem; }
      .hero-shell { padding:1.25rem; border-radius:20px; }
      .hero-grid { grid-template-columns:1fr; gap:1.2rem; margin-top:1.6rem; }
      .hero-info { max-width:none; }
      .top-stats { gap:.55rem; }
      .top-stat { padding:.75rem .7rem; gap:.55rem; }
      .top-stat-icon { width:30px; height:30px; }
      .top-stat strong { font-size:.78rem; }
      .top-stat small { font-size:.67rem; }
      .result-card, .empty-result { min-height:260px; }
    }
    @media (max-width: 560px) {
      .hero-shell { padding:1rem; }
      .brand-mark { width:38px; height:38px; border-radius:12px; }
      .brand-copy strong { font-size:.78rem; }
      .brand-copy small { font-size:.68rem; }
      .status-pill { font-size:.59rem; padding:.35rem .45rem; }
      .hero-title { font-size:2.25rem; }
      .hero-copy { font-size:.88rem; }
      .top-stats { grid-template-columns:1fr; }
      .top-stat { min-height:58px; }
      .result-amount { font-size:2.25rem; }
      .metric-card { padding:.72rem; }
      .metric-value { font-size:1.05rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)


# Header / brand area.
st.markdown(
    """
    <div class="hero-shell">
      <div class="hero-topline">
        <div class="brand-mark">⌂</div>
        <div class="brand-copy">
          <strong>HOUSE PRICE STUDIO</strong>
          <small>Machine learning · Property insights</small>
        </div>
        <div class="status-pill"><span class="status-dot"></span> MODEL READY</div>
      </div>
      <div class="hero-grid">
        <div>
          <div class="hero-eyebrow">Regression-powered estimate</div>
          <h1 class="hero-title">A clearer view of<br/><span>home value.</span></h1>
          <p class="hero-copy">Explore a home's estimated sale price using its living area, bedrooms, and bathrooms. Adjust the details and run a prediction in seconds.</p>
        </div>
        <div class="hero-info">
          <div class="hero-info-label">Your model</div>
          <div class="hero-info-title">Linear Regression</div>
          <div class="hero-info-copy">Trained on 1,460 historical house sales from Ames, Iowa, using the Kaggle House Prices dataset.</div>
        </div>
      </div>
    </div>
    <div class="top-stats">
      <div class="top-stat"><div class="top-stat-icon">⌂</div><div><strong>3 property features</strong><small>Simple, interpretable inputs</small></div></div>
      <div class="top-stat"><div class="top-stat-icon">↗</div><div><strong>Held-out evaluation</strong><small>Performance on unseen rows</small></div></div>
      <div class="top-stat"><div class="top-stat-icon">$</div><div><strong>USD estimates</strong><small>Matches the dataset currency</small></div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.is_file():
    st.error("The trained model was not found in the artifacts folder.")
    st.markdown("Run `python train_model.py` from the project directory to create it.")
    st.stop()

try:
    bundle = load_bundle(str(MODEL_PATH))
except Exception as exc:  # Give a useful message if the saved artifact is damaged/incompatible.
    st.error(f"Could not load the trained model: {exc}")
    st.markdown("Try reinstalling the dependencies and running `python train_model.py` again.")
    st.stop()

# Keep the input values and latest submitted result between Streamlit reruns.
st.session_state.setdefault("area_input", 1500)
st.session_state.setdefault("bedrooms_input", 3)
st.session_state.setdefault("bathrooms_input", 2.0)
st.session_state.setdefault("last_prediction", None)
st.session_state.setdefault("last_inputs", None)

left, right = st.columns([1.03, 0.97], gap="large")

with left:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-kicker">01 / Property details</div><div class="panel-title">Tell us about the home</div><div class="panel-copy">Start with a quick profile or customize the measurements below.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="preset-label">QUICK START PROFILES</div>', unsafe_allow_html=True)
        preset_cols = st.columns(3, gap="small")
        presets = [
            ("Compact", 900, 2, 1.0),
            ("Family", 1500, 3, 2.0),
            ("Spacious", 2500, 4, 3.0),
        ]
        for col, (label, preset_area, preset_beds, preset_baths) in zip(preset_cols, presets):
            with col:
                if st.button(label, key=f"preset_{label.lower()}", use_container_width=True):
                    st.session_state["area_input"] = preset_area
                    st.session_state["bedrooms_input"] = preset_beds
                    st.session_state["bathrooms_input"] = preset_baths

        with st.form("prediction_form", clear_on_submit=False):
            area = st.number_input(
                "Above-ground living area (sq. ft.)",
                min_value=100,
                max_value=10000,
                step=50,
                key="area_input",
                help="Enter the above-ground living area in square feet.",
            )
            bedrooms = st.number_input(
                "Bedrooms above ground",
                min_value=0,
                max_value=10,
                step=1,
                key="bedrooms_input",
                help="Count bedrooms above ground level.",
            )
            bathrooms = st.number_input(
                "Total bathroom equivalent",
                min_value=0.0,
                max_value=12.0,
                step=0.5,
                key="bathrooms_input",
                help="Count a full bathroom as 1 and a half-bath as 0.5. Basement bathrooms are included.",
            )
            submitted = st.form_submit_button("Estimate house price  →", use_container_width=True)

        st.caption("Tip: use the quick profiles as examples, then adjust each value to explore how your inputs affect the estimate.")

        if submitted:
            values = pd.DataFrame(
                [[float(area), float(bedrooms), float(bathrooms)]],
                columns=bundle["feature_columns"],
            )
            try:
                st.session_state["last_prediction"] = float(bundle["model"].predict(values)[0])
                st.session_state["last_inputs"] = {
                    "area": int(area),
                    "bedrooms": int(bedrooms),
                    "bathrooms": float(bathrooms),
                }
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with right:
    st.markdown(
        '<div class="panel-kicker">02 / Your result</div><div class="panel-title">Estimated sale price</div><div class="panel-copy">A model-based estimate, not a formal property valuation.</div>',
        unsafe_allow_html=True,
    )
    prediction = st.session_state.get("last_prediction")
    inputs = st.session_state.get("last_inputs")
    if prediction is None or inputs is None:
        st.markdown(
            """
            <div class="empty-result">
              <div class="empty-icon">↗</div>
              <div class="empty-title">Your estimate starts here</div>
              <div class="empty-copy">Choose a quick profile or enter the home's details, then select <b>Estimate house price</b> to see the result here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if prediction <= 0:
            st.warning("The model returned a non-positive estimate for these inputs. Try realistic property details; linear regression has limitations at the edges of the data.")
        st.markdown(
            f"""
            <div class="result-card">
              <div>
                <div class="result-label">Predicted sale price</div>
                <div class="result-amount">${prediction:,.0f}<span class="currency-tag">USD</span></div>
                <div class="result-caption">Estimated from the three property features you submitted. This is an educational prediction based on historical sales.</div>
              </div>
              <div>
                <div class="result-divider"></div>
                <div class="result-detail-label">Inputs used for this estimate</div>
                <div class="result-chips">
                  <span class="result-chip">⌂ &nbsp; {inputs['area']:,} sq. ft.</span>
                  <span class="result-chip">▦ &nbsp; {inputs['bedrooms']} bedrooms</span>
                  <span class="result-chip">◉ &nbsp; {inputs['bathrooms']:g} bath equiv.</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown(
    '<div class="panel-kicker">03 / Model health</div><div class="section-title">How the model performs</div><div class="section-subtitle">Evaluation uses a held-out test split that was not used to fit the model.</div>',
    unsafe_allow_html=True,
)
metrics = bundle.get("metrics", {})
metric_cols = st.columns(3, gap="medium")
metric_cards = [
    ("Mean absolute error", f"${metrics['MAE_USD']:,.0f}" if "MAE_USD" in metrics else "N/A", "Typical absolute prediction error"),
    ("Root mean squared error", f"${metrics['RMSE_USD']:,.0f}" if "RMSE_USD" in metrics else "N/A", "Penalizes larger errors more"),
    ("R² score", f"{metrics['R2']:.3f}" if "R2" in metrics else "N/A", "Variation explained on test data"),
]
for col, (label, value, note) in zip(metric_cols, metric_cards):
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
with st.expander("Understand the estimate and its limitations"):
    st.markdown(
        """
        - **Living area:** above-ground living area in square feet.
        - **Bedrooms:** number of bedrooms above ground.
        - **Bathrooms:** full bathrooms count as 1 and half-baths count as 0.5; basement bathrooms are included.
        - **Currency and location:** the training target is historical sale price from Ames, Iowa, USA, and values are displayed in USD.
        - **Important limitation:** the model uses only three features. It does not account for neighborhood, property condition, age, lot size, or current market conditions. Do not use it as a formal appraisal or an estimate for Indian properties.
        """
    )

st.markdown(
    '<div style="text-align:center;color:#71819e;font-size:.72rem;padding:1.25rem .25rem .1rem;">HOUSE PRICE STUDIO · BUILT FOR MACHINE LEARNING EDUCATION</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div style="text-align:center;color:#d1d5db;font-size:.78rem;'
    'padding:1.25rem .25rem .1rem;">'
    'Developed by '
    '<a href="https://www.linkedin.com/in/saratbanik" '
    'target="_blank" rel="noopener noreferrer" '
    'style="color:inherit;font-weight:700;text-decoration:none;">'
    'SARAT BANIK</a>'
    '</div>',
    unsafe_allow_html=True,
)