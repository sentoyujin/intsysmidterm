import os
import streamlit as st
import joblib
import pandas as pd

# Page setup
st.set_page_config(page_title="Facility Accessibility Classification")

# Load the exported model/bundle
@st.cache_resource
def load_bundle():
    bundle_path = os.path.join(os.path.dirname(__file__), "model_bundle.joblib")
    return joblib.load(bundle_path)

bundle = load_bundle()

models = bundle["models"]
best_model_name = bundle["best_model"]
feature_columns = bundle["features"]
yes_no_map = bundle["yes_no_map"]
metrics = bundle["metrics"]
feature_importances = bundle["feature_importances"]

# UI header
st.title("Facility Accessibility Classifier by Group 28")
st.write(
    f"This interface uses machine leaerning to classify accessibility standards.  "
    f"The default evaluated model best model is **{best_model_name}**."
)

#Sidebar
with st.sidebar:
    st.sidebar.header("Model configuration")
    selected_model_name = st.selectbox(
        "Choose Model for Prediction: ", options=list(models.keys()), index=list(models.keys()).index(best_model_name)
    )

    st.divider()
    st.subheader(f"{selected_model_name} Metrics")
    model_stats = metrics[selected_model_name]
    st.metric("Test accuracy", f"{model_stats['accuracy']* 100:.1f}%")
    st.metric("Macro F1-Score", f"{model_stats['f1']:.3f}")
    st.metric("5-Fold CV F1", f"{model_stats['cv_f1']:.3f}")

#Main content tabs
tab_predict, tab_insights = st.tabs(["Predict Accessility", "Feature Insights"])

with tab_predict:
    st.subheader("Enter facility features")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            doorway_width = st.number_input(
                "Doorway Width (cm)", 
                min_value=1.0,
                max_value=300.0,
                value= 90.0,
                step = 1.0,
                help= "Measured passage width of primary access doors."
            )
            ramp = st.selectbox("Ramp Available?", ("Yes", "No"))
            elevator = st.selectbox("Elevator Available?", ("Yes", "No"))

        with col2:
            accessible_toilet= st.selectbox("Accessible Toilet Available?", ("Yes", "No"))
            handrails = st.selectbox("Handrails Installed?", ("Yes", "No"))
            accessible_parking = st.selectbox("Accessible Parking Available?", ("Yes", "No"))

        submit_btn = st.form_submit_button("Classify Facility")

    if submit_btn:
        raw_inputs = {
            "ramp": yes_no_map[ramp],
            "elevator": yes_no_map[elevator],
            "accessible_toilet": yes_no_map[accessible_toilet],
            "handrails": yes_no_map[handrails],
            "accessible_parking": yes_no_map[accessible_parking],
            "doorway_width_cm": float(doorway_width)
        }

        input_df = pd.DataFrame([raw_inputs])[feature_columns]
        active_model = models[selected_model_name]
        prediction = active_model.predict(input_df)[0]

        st.subheader("Classification Outcome")
        if prediction == "Accessible":
            st.success(f"### Result: **{prediction}**")
        elif prediction == "Partially Accessible":
            st.warning(f"### Result: **{prediction}**")
        else:
            st.error(f"### Result: **{prediction}##")

    with tab_insights:
        st.subheader(f"Feature Importance for {selected_model_name}")
        importances= pd.Series(feature_importances[selected_model_name]).sort_values(ascending=True)
        st.bar_chart(importances)