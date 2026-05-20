
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import plotly.express as px

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="PH Earthquake Significance Predictor",
    page_icon="🌏",
    layout="wide"
)

# ── Load model & metadata ────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, scaler, meta

model, scaler, meta = load_artifacts()

# ── Significance interpreter ─────────────────────────────
def interpret_sig(score):
    if score < 100:
        return "🟢 Low",      "Minor event. Unlikely to cause damage.", "#2ecc71"
    elif score < 500:
        return "🟡 Moderate", "Felt by many. Possible minor damage.",  "#f39c12"
    elif score < 1000:
        return "🟠 High",     "Significant event. Mobilize response.", "#e67e22"
    else:
        return "🔴 Critical", "Major earthquake! Full disaster response needed.", "#e74c3c"

# ── Header ───────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#353F8E;'>
    🌏 Philippine Earthquake Significance Predictor
</h1>
<p style='text-align:center; color:gray; font-size:16px;'>
    Predictive Regression Model for Rapid Disaster Response &nbsp;|&nbsp;
    CPELX130 — Data Science
</p>
<hr>
""", unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    with st.form("earthquake_form"):
        st.subheader("📥 Earthquake Input Parameters")
        st.caption("Type the exact values or use the arrows to adjust.")

        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            mag = st.number_input("Magnitude (Richter)",
                                  min_value=0.0, max_value=9.5, value=7.2, step=0.1, format="%.1f")
        with row1_col2:
            depth = st.number_input("Depth (km)",
                                    min_value=0.0, max_value=700.0, value=12.0, step=1.0, format="%.1f")

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            latitude = st.number_input("Latitude (°N)",
                                       min_value=4.5, max_value=21.5, value=9.80, step=0.01, format="%.4f")
        with row2_col2:
            longitude = st.number_input("Longitude (°E)",
                                        min_value=116.0, max_value=127.5, value=124.00, step=0.01, format="%.4f")

        extra_inputs = {}
        optional_cols = [f for f in meta["features"] if f not in ["mag","depth","latitude","longitude"]]
        if optional_cols:
            with st.expander("⚙️ Advanced Parameters (optional)"):
                for col in optional_cols:
                    extra_inputs[col] = st.number_input(col, value=0.0, format="%.4f")

        predict_btn = st.form_submit_button("🔮 Predict Significance Score",
                                            use_container_width=True, type="primary")

with col_result:
    st.subheader("📊 Prediction Result")

    if predict_btn:
        # Build input array
        input_dict = {
            "mag": mag, "depth": depth,
            "latitude": latitude, "longitude": longitude
        }
        input_dict.update(extra_inputs)

        input_df = pd.DataFrame([[input_dict[f] for f in meta["features"]]],
                                 columns=meta["features"])

        if meta["scaled_input"]:
            input_scaled = scaler.transform(input_df)
            prediction   = model.predict(input_scaled)[0]
        else:
            prediction   = model.predict(input_df)[0]

        prediction = max(0, prediction)  # no negative scores
        label, desc, color = interpret_sig(prediction)

        # Big score display
        st.markdown(f"""
        <div style='background:{color}22; border:2px solid {color};
                    border-radius:12px; padding:20px; text-align:center;'>
            <h2 style='color:{color}; margin:0;'>Predicted Significance Score</h2>
            <h1 style='color:{color}; font-size:64px; margin:8px 0;'>
                {prediction:.1f}
            </h1>
            <h3 style='color:{color}; margin:0;'>{label}</h3>
            <p style='color:#555; margin-top:8px;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # Gauge bar
        st.markdown("#### Significance Level Gauge")
        fig_g, ax_g = plt.subplots(figsize=(7, 1.2))
        ax_g.set_xlim(0, 1500)
        ax_g.set_ylim(0, 1)
        ax_g.set_yticks([])

        zones = [(0,100,'#2ecc71','Low'),(100,500,'#f39c12','Moderate'),
                 (500,1000,'#e67e22','High'),(1000,1500,'#e74c3c','Critical')]
        for x0,x1,c,lbl in zones:
            ax_g.barh(0, x1-x0, left=x0, height=0.6, color=c, alpha=0.5)
            ax_g.text((x0+x1)/2, 0.7, lbl, ha='center', va='bottom', fontsize=8)

        ax_g.axvline(min(prediction, 1490), color='black', linewidth=3, label=f'{prediction:.0f}')
        ax_g.set_xlabel("Significance Score", fontsize=9)
        ax_g.legend(loc='upper right', fontsize=8)
        st.pyplot(fig_g, use_container_width=True)

        # NEW: Bulletproof Plotly Map Integration
        st.markdown("#### 📍 Epicenter Location")
        map_df = pd.DataFrame({"latitude": [latitude], "longitude": [longitude]})

        # Create map using OpenStreetMap tiles (no API key needed)
        fig_map = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            zoom=5,
            hover_name=["Epicenter"]
        )
        # Style the marker
        fig_map.update_traces(marker=dict(size=16, color='red'))
        # Set map layout
        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0, "t":0, "l":0, "b":0},
            height=350
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Summary table
        st.markdown("#### Input Summary")
        summary_data = {
            "Parameter": ["Magnitude","Depth (km)","Latitude","Longitude","Predicted Sig.","Level"],
            "Value"    : [f"{mag:.1f}",f"{depth:.1f} km",f"{latitude:.4f}°N",
                          f"{longitude:.4f}°E",f"{prediction:.2f}", label]
        }
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

    else:
        st.info("👈 Enter the exact earthquake parameters on the left and click **Predict**.")

# ── Model Info ───────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("📈 Model Performance")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Best Model",  meta["model_name"].split()[0])
m2.metric("R² Score",    f"{meta['metrics']['R2']:.4f}")
m3.metric("RMSE",        f"{meta['metrics']['RMSE']:.2f}")
m4.metric("MAE",         f"{meta['metrics']['MAE']:.2f}")
'''

with open("app.py", "w") as f:
    f.write(app_code)

print("✅ app.py written successfully with robust Plotly Map!")
