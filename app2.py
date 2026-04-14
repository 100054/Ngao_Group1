import numpy as np
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import base64
import os
import traceback


# -----------------------------
# App config (must come early)
# -----------------------------
st.set_page_config(
    page_title="Vaccine Prediction Dashboard",
    page_icon="💉",
    layout="wide"
)


# -----------------------------
# Load models and encoders
# -----------------------------
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    files = {
        "seasonal_model": "Gradient_boosting_Seasonal_grid.joblib",
        "h1n1_model": "model_xgb_H1N1.joblib",
        "geo_region_encoder": "geo_region_encoder.pkl",
        "age_group_encoder": "age_group_encoder.pkl",
        "edu_label_encoder": "edu_label_encoder.pkl",
        "census_msa_encoder": "census_msa_encoder.pkl",
        "behavioral_features_scaler": "behavioral_features_scaler.pkl",
        "doctor_recc_seasonal_scaler": "doctor_recc_seasonal_scaler.pkl",
        "doctor_recc_h1n1_scaler": "doctor_recc_h1n1_scaler.pkl",
        "health_insurance_scaler": "health_insurance_scaler.pkl"
    }

    loaded_objects = {}

    for name, filename in files.items():
        file_path = os.path.join(BASE_DIR, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{filename} not found at {file_path}")

        loaded_objects[name] = joblib.load(file_path)

    return loaded_objects


try:
    models = load_models()

    seasonal_model = models["seasonal_model"]
    h1n1_model = models["h1n1_model"]

    geo_region_encoder = models["geo_region_encoder"]
    age_group_encoder = models["age_group_encoder"]
    edu_label_encoder = models["edu_label_encoder"]
    census_msa_encoder = models["census_msa_encoder"]

    behavioral_scaler = models["behavioral_features_scaler"]
    seasonal_rec_scaler = models["doctor_recc_seasonal_scaler"]
    h1n1_rec_scaler = models["doctor_recc_h1n1_scaler"]
    health_insurance_scaler = models["health_insurance_scaler"]

    models_loaded = True

except Exception as e:
    models_loaded = False
    load_error = repr(e)
    load_traceback = traceback.format_exc()


# -----------------------------
# Helper functions
# -----------------------------
def get_encoder_options(encoder_obj):
    """
    Return options for either:
    - dict encoder
    - sklearn LabelEncoder-like object with .classes_
    """
    if isinstance(encoder_obj, dict):
        return list(encoder_obj.keys())
    elif hasattr(encoder_obj, "classes_"):
        return list(encoder_obj.classes_)
    else:
        raise TypeError(f"Unsupported encoder type: {type(encoder_obj)}")


def encode_value(encoder_obj, value):
    """
    Encode one value using either:
    - dict encoder
    - sklearn LabelEncoder-like object
    """
    if isinstance(encoder_obj, dict):
        return encoder_obj[value]
    elif hasattr(encoder_obj, "transform"):
        return encoder_obj.transform([value])[0]
    else:
        raise TypeError(f"Unsupported encoder type: {type(encoder_obj)}")



def get_expected_feature_names(model):
    """
    Return the exact feature names and order expected by a fitted model or pipeline.
    Supports plain estimators, sklearn pipelines, and GridSearchCV wrappers.
    """
    candidate = model

    if hasattr(candidate, "best_estimator_"):
        candidate = candidate.best_estimator_

    if hasattr(candidate, "feature_names_in_"):
        return list(candidate.feature_names_in_)

    if hasattr(candidate, "named_steps"):
        for _, step in reversed(list(candidate.named_steps.items())):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    raise AttributeError("Could not determine expected feature names from the loaded model.")


def align_features_to_model(model, input_df, label="model"):
    """
    Reindex input_df so its columns exactly match the order used during model training.
    Raises a helpful error if required columns are missing.
    """
    expected_cols = get_expected_feature_names(model)
    actual_cols = list(input_df.columns)

    missing = [col for col in expected_cols if col not in actual_cols]
    extra = [col for col in actual_cols if col not in expected_cols]

    if missing:
        raise ValueError(
            f"{label} is missing required features: {missing}. "
            f"Available features: {actual_cols}"
        )

    aligned_df = input_df.reindex(columns=expected_cols)

    return aligned_df, {
        "expected": expected_cols,
        "actual": actual_cols,
        "missing": missing,
        "extra": extra
    }


def create_donut_chart(prob, title, color):
    remaining = 1 - prob
    df_chart = pd.DataFrame({
        "Category": ["Vaccinate", "Not Vaccinate"],
        "Value": [prob, remaining]
    })

    fig = px.pie(
        df_chart,
        values="Value",
        names="Category",
        hole=0.6,
        title=title,
        color_discrete_sequence=[color, "rgba(240, 240, 240, 0.7)"]
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate=None,
        marker=dict(line=dict(color="white", width=2)),
        insidetextfont=dict(color="white")
    )

    fig.update_layout(
        showlegend=False,
        annotations=[dict(
            text=f"{prob:.0%}",
            x=0.5,
            y=0.5,
            font_size=28,
            showarrow=False
        )],
        title_font_color="white",
        font=dict(color="white"),
        margin=dict(t=40, b=0, l=0, r=0),
        paper_bgcolor="rgba(44, 69, 80, 0.85)",
        plot_bgcolor="rgba(200, 200, 200, 0.85)"
    )

    return fig


def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="vaccine_predictions.csv">Download CSV with predictions</a>'
    return href


def get_decision(prob):
    if prob >= 0.6:
        return "LIKELY TO VACCINATE", "likely"
    elif prob >= 0.4:
        return "MAY VACCINATE", "maybe"
    else:
        return "UNLIKELY TO VACCINATE", "unlikely"


def get_decision_category(prob):
    if prob >= 0.6:
        return "Likely"
    elif prob >= 0.4:
        return "Maybe"
    else:
        return "Unlikely"


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/free-vector/coronavirus-globe-concept_52683-35722.jpg?t=st=1744156153~exp=1744159753~hmac=2f129d537b8f4320b204be88ce714a8c8cd3942065f751d1ae02f391cd988702&w=996");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(44, 69, 80, 0.85) !important;
        color: white !important;
        backdrop-filter: blur(10px);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6,
    section[data-testid="stSidebar"] p {
        color: white !important;
    }

    .main-content {
        background-color: rgba(44, 69, 80, 0.85);
        backdrop-filter: blur(8px);
        border-radius: 15px;
        padding: 2.5rem;
        margin: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        border: 1px solid rgba(44, 69, 80, 0.85);
    }

    h1, h2, h3 {
        color: #FFFFFF;
    }

    .stSelectbox, .stRadio, .stCheckbox {
        background-color: #808080;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 15px;
    }

    .decision-box {
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .likely {
        background: linear-gradient(135deg, rgba(76, 35, 80, 0.9), rgba(129, 199, 132, 0.9));
        color: white;
    }

    .unlikely {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.9), rgba(229, 115, 115, 0.9));
        color: white;
    }

    .maybe {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.9), rgba(255, 213, 79, 0.9));
        color: #2c3e50;
    }

    .stButton > button {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 24px;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #059669, #047857);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Main app
# -----------------------------
st.title("Vaccine Uptake Prediction")
st.markdown("Predict the likelihood of receiving H1N1 and seasonal flu vaccines")

if not models_loaded:
    st.error(f"Error loading models: {load_error}")
    st.text(load_traceback)
    st.stop()

st.success("All models loaded successfully")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])


# -----------------------------
# Tab 1: Single Prediction
# -----------------------------
with tab1:
    with st.form("vaccine_form"):
        st.subheader("Personal Information")

        col1, col2 = st.columns(2)

        with col1:
            hhs_geo_region = st.selectbox(
                "Geographic Region",
                options=get_encoder_options(geo_region_encoder)
            )

            age_group = st.selectbox(
                "Age Group",
                options=get_encoder_options(age_group_encoder)
            )

            census_msa = st.selectbox(
                "Metropolitan Status",
                options=get_encoder_options(census_msa_encoder)
            )

        with col2:
            education = st.selectbox(
                "Education Level",
                options=get_encoder_options(edu_label_encoder)
            )

            health_insurance = st.radio(
                "Has Health Insurance",
                options=[1, 0],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True
            )

        st.subheader("Doctor Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            doctor_recc_seasonal = st.radio(
                "Recommended Seasonal Vaccine",
                options=[1, 0],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True
            )

        with col2:
            doctor_recc_h1n1 = st.radio(
                "Recommended H1N1 Vaccine",
                options=[1, 0],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True
            )

        st.subheader("Protective Behaviors")
        st.markdown("Select all behaviors practiced in the last month:")

        cols = st.columns(2)

        with cols[0]:
            behavioral_antiviral_meds = st.checkbox("Took antiviral medications")
            behavioral_avoidance = st.checkbox("Avoided crowded places")
            behavioral_face_mask = st.checkbox("Wore face mask")

        with cols[1]:
            behavioral_wash_hands = st.checkbox("Frequently washed hands")
            behavioral_large_gatherings = st.checkbox("Avoided large gatherings")
            behavioral_outside_home = st.checkbox("Reduced time outside home")

        behavioral_touch_face = st.checkbox("Avoided touching face")

        submitted = st.form_submit_button("Predict Vaccine Uptake")

    if submitted:
        try:
            behavioral_features = sum([
                int(behavioral_antiviral_meds),
                int(behavioral_avoidance),
                int(behavioral_face_mask),
                int(behavioral_wash_hands),
                int(behavioral_large_gatherings),
                int(behavioral_outside_home),
                int(behavioral_touch_face)
            ])

            behavioral_features_scaled = behavioral_scaler.transform([[behavioral_features]])[0][0]
            doctor_recc_seasonal_scaled = seasonal_rec_scaler.transform([[doctor_recc_seasonal]])[0][0]
            doctor_recc_h1n1_scaled = h1n1_rec_scaler.transform([[doctor_recc_h1n1]])[0][0]
            health_insurance_scaled = health_insurance_scaler.transform([[health_insurance]])[0][0]

            base_input_data = pd.DataFrame({
                "encoded_geo_region": [encode_value(geo_region_encoder, hhs_geo_region)],
                "behavioral_features": [behavioral_features_scaled],
                "encoded_age_group": [encode_value(age_group_encoder, age_group)],
                "education": [encode_value(edu_label_encoder, education)],
                "doctor_recc_seasonal": [doctor_recc_seasonal_scaled],
                "doctor_recc_h1n1": [doctor_recc_h1n1_scaled],
                "encoded_census_msa": [encode_value(census_msa_encoder, census_msa)],
                "health_insurance": [health_insurance_scaled]
            })

            seasonal_input_data, seasonal_debug = align_features_to_model(
                seasonal_model, base_input_data, label="seasonal_model"
            )
            h1n1_input_data, h1n1_debug = align_features_to_model(
                h1n1_model, base_input_data, label="h1n1_model"
            )

            seasonal_prob = seasonal_model.predict_proba(seasonal_input_data)[0][1]
            h1n1_prob = h1n1_model.predict_proba(h1n1_input_data)[0][1]

            seasonal_decision, seasonal_class = get_decision(seasonal_prob)
            h1n1_decision, h1n1_class = get_decision(h1n1_prob)

            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(
                    create_donut_chart(seasonal_prob, "Seasonal Vaccine", "#3498db"),
                    use_container_width=True
                )
                st.markdown(
                    f'<div class="decision-box {seasonal_class}">{seasonal_decision}</div>',
                    unsafe_allow_html=True
                )

            with col2:
                st.plotly_chart(
                    create_donut_chart(h1n1_prob, "H1N1 Vaccine", "#e74c3c"),
                    use_container_width=True
                )
                st.markdown(
                    f'<div class="decision-box {h1n1_class}">{h1n1_decision}</div>',
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            if "seasonal_debug" in locals():
                st.write("Seasonal model expected order:", seasonal_debug["expected"])
                st.write("Seasonal input columns provided:", seasonal_debug["actual"])
                st.write("Seasonal extra columns ignored:", seasonal_debug["extra"])
            if "h1n1_debug" in locals():
                st.write("H1N1 model expected order:", h1n1_debug["expected"])
                st.write("H1N1 input columns provided:", h1n1_debug["actual"])
                st.write("H1N1 extra columns ignored:", h1n1_debug["extra"])
            st.text(traceback.format_exc())


# -----------------------------
# Tab 2: Batch Prediction
# -----------------------------
with tab2:
    st.header("Batch Predictions from CSV")
    st.markdown("Upload a CSV file with multiple records to get predictions for all entries")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            original_columns = df.columns.tolist()

            required_columns = [
                "hhs_geo_region",
                "age_group",
                "census_msa",
                "education",
                "health_insurance",
                "doctor_recc_seasonal",
                "doctor_recc_h1n1"
            ]

            behavioral_columns = [
                "behavioral_antiviral_meds",
                "behavioral_avoidance",
                "behavioral_face_mask",
                "behavioral_wash_hands",
                "behavioral_large_gatherings",
                "behavioral_outside_home",
                "behavioral_touch_face"
            ]

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.stop()

            yes_no_map = {
                "yes": "1",
                "no": "0",
                "y": "1",
                "n": "0",
                "1": "1",
                "0": "0"
            }

            binary_columns = [
                "health_insurance",
                "doctor_recc_seasonal",
                "doctor_recc_h1n1",
                *behavioral_columns
            ]

            for col in binary_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip().str.lower().map(yes_no_map).fillna(df[col].astype(str))
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

            df["encoded_geo_region"] = df["hhs_geo_region"].apply(lambda x: encode_value(geo_region_encoder, x))
            df["encoded_age_group"] = df["age_group"].apply(lambda x: encode_value(age_group_encoder, x))
            df["encoded_census_msa"] = df["census_msa"].apply(lambda x: encode_value(census_msa_encoder, x))
            df["education"] = df["education"].apply(lambda x: encode_value(edu_label_encoder, x))

            behavioral_cols_present = [col for col in behavioral_columns if col in df.columns]
            if behavioral_cols_present:
                df["behavioral_features"] = df[behavioral_cols_present].sum(axis=1)
            else:
                df["behavioral_features"] = 0

            df["behavioral_features"] = behavioral_scaler.transform(df[["behavioral_features"]]).ravel()
            df["doctor_recc_seasonal"] = seasonal_rec_scaler.transform(df[["doctor_recc_seasonal"]]).ravel()
            df["doctor_recc_h1n1"] = h1n1_rec_scaler.transform(df[["doctor_recc_h1n1"]]).ravel()
            df["health_insurance"] = health_insurance_scaler.transform(df[["health_insurance"]]).ravel()

            base_input_data = df[[
                "encoded_geo_region",
                "behavioral_features",
                "encoded_age_group",
                "education",
                "doctor_recc_seasonal",
                "doctor_recc_h1n1",
                "encoded_census_msa",
                "health_insurance"
            ]].copy()

            seasonal_input_data, seasonal_debug = align_features_to_model(
                seasonal_model, base_input_data, label="seasonal_model"
            )
            h1n1_input_data, h1n1_debug = align_features_to_model(
                h1n1_model, base_input_data, label="h1n1_model"
            )

            seasonal_probs = seasonal_model.predict_proba(seasonal_input_data)[:, 1]
            h1n1_probs = h1n1_model.predict_proba(h1n1_input_data)[:, 1]

            df["seasonal_vaccine_prob"] = seasonal_probs
            df["h1n1_vaccine_prob"] = h1n1_probs

            df["seasonal_decision"] = df["seasonal_vaccine_prob"].apply(get_decision_category)
            df["h1n1_decision"] = df["h1n1_vaccine_prob"].apply(get_decision_category)

            st.subheader("Prediction Results Preview")
            st.dataframe(
                df[original_columns + [
                    "seasonal_vaccine_prob",
                    "h1n1_vaccine_prob",
                    "seasonal_decision",
                    "h1n1_decision"
                ]].head()
            )

            st.subheader("Prediction Summary")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Average Seasonal Vaccine Probability",
                    f"{df['seasonal_vaccine_prob'].mean():.1%}"
                )
                st.metric(
                    "Likely to Take Seasonal Vaccine",
                    f"{len(df[df['seasonal_decision'] == 'Likely'])} people"
                )

                fig1 = px.pie(
                    df,
                    names="seasonal_decision",
                    title="Seasonal Vaccine Decisions",
                    color_discrete_sequence=px.colors.sequential.Blues_r
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.metric(
                    "Average H1N1 Vaccine Probability",
                    f"{df['h1n1_vaccine_prob'].mean():.1%}"
                )
                st.metric(
                    "Likely to Take H1N1 Vaccine",
                    f"{len(df[df['h1n1_decision'] == 'Likely'])} people"
                )

                fig2 = px.pie(
                    df,
                    names="h1n1_decision",
                    title="H1N1 Vaccine Decisions",
                    color_discrete_sequence=px.colors.sequential.Reds_r
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Vaccine Probability by Age Group")
            age_group_analysis = df.groupby("age_group")[[
                "seasonal_vaccine_prob",
                "h1n1_vaccine_prob"
            ]].mean().reset_index()

            fig3 = px.bar(
                age_group_analysis,
                x="age_group",
                y=["seasonal_vaccine_prob", "h1n1_vaccine_prob"],
                barmode="group",
                title="Average Vaccine Probability by Age Group",
                labels={"value": "Probability", "variable": "Vaccine Type"}
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.markdown(get_table_download_link(df), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during batch processing: {str(e)}")
            if "seasonal_debug" in locals():
                st.write("Seasonal model expected order:", seasonal_debug["expected"])
                st.write("Seasonal input columns provided:", seasonal_debug["actual"])
                st.write("Seasonal extra columns ignored:", seasonal_debug["extra"])
            if "h1n1_debug" in locals():
                st.write("H1N1 model expected order:", h1n1_debug["expected"])
                st.write("H1N1 input columns provided:", h1n1_debug["actual"])
                st.write("H1N1 extra columns ignored:", h1n1_debug["extra"])
            st.text(traceback.format_exc())


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
    This dashboard predicts vaccine uptake probability based on:
    - Demographic factors
    - Doctor recommendations
    - Protective behaviors

    **Interpretation Guide:**
    - **≥60%**: Very likely to vaccinate
    - **40-59%**: May vaccinate
    - **<40%**: Unlikely to vaccinate
    """)

    st.markdown("---")

    st.markdown("**Model Information**")
    st.markdown("""
    - Uses Gradient Boosting models
    - Trained on CDC data
    - 8 key predictive factors
    """)


# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div style="text-align: center; color: white; margin-top: 20px;">
        © 2025 Vaccine Prediction Dashboard | For public health use
    </div>
    """,
    unsafe_allow_html=True
)
