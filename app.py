# command to run it:
# python -m streamlit run app.py

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

model_names = list(models.keys())

# UI header
st.title("Facility Accessibility Classifier")
st.write(
    "Find out how accessible a facility is. Just enter its features below, choose an algorithm, and the app will classify a facility as Accessible, Partially Accessible, or Not Accessible, along with a short explanation. \n\n" 
    "The best performing model is **" + best_model_name + "**."
)

st.divider()

st.subheader("Enter Facility Features")

# The user input section
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        doorway_width = st.number_input(
            "Doorway Width (cm)",
            min_value=1.0,
            max_value=300.0,
            value=90.0,
            step=1.0,
            help="Measured passage width of primary access doors."
        )
        ramp = st.selectbox("Ramp Available?", ("Yes", "No"))
        elevator = st.selectbox("Elevator Available?", ("Yes", "No"))

    with col2:
        accessible_toilet = st.selectbox(
            "Accessible Toilet Available?", ("Yes", "No")
        )
        handrails = st.selectbox("Handrails Installed?", ("Yes", "No"))
        accessible_parking = st.selectbox(
            "Accessible Parking Available?", ("Yes", "No")
        )

    st.markdown("#### Choose a Classification Algorithm")
    st.caption(
        "Select which trained model you want to use for the prediction."
    )

    # Two algorithm choices, displayed horizontally
    model_options = [
        f"{name} (Best)" if name == best_model_name else name
        for name in model_names
    ]

    selected_model_option = st.radio(
        "Classification Algorithm",
        options=model_options,
        horizontal=True,
        label_visibility="collapsed"
    )

    # Recover the actual model name without the display label
    selected_model = model_names[model_options.index(selected_model_option)]

    classify_button = st.form_submit_button(
        "Classify Facility",
        type="primary",
        use_container_width=True
    )

# Input validation - just for the doorway width, since Yes/No fields will always have values
input_error = ""
 
if classify_button:
    if doorway_width is None:
        input_error = "Doorway width is missing. Please enter a value in cm."

    elif doorway_width < 1 or doorway_width > 300:
        input_error = "Doorway width must be between 1 and 300 cm."
 
# show the problem and skip the classification if the width is not valid
if input_error != "":
    st.error("**Input error:** " + input_error)

# Classification result
if classify_button:

    raw_inputs = {
        "ramp": yes_no_map[ramp],
        "elevator": yes_no_map[elevator],
        "accessible_toilet": yes_no_map[accessible_toilet],
        "handrails": yes_no_map[handrails],
        "accessible_parking": yes_no_map[accessible_parking],
        "doorway_width_cm": float(doorway_width)
    }

    # Arrange features in the order expected by the model
    input_df = pd.DataFrame([raw_inputs])[feature_columns]

    active_model = models[selected_model]
    prediction = active_model.predict(input_df)[0]

    st.subheader("Facility is:")
    st.caption("Model used: " + selected_model)

    if prediction == "Accessible":
        st.success("### " + prediction)
    elif prediction == "Partially Accessible":
        st.warning("### " + prediction)
    else:
        st.error("### " + prediction)

    # results output explanation
    # step-free access means the facility has a ramp or an elevator
    has_step_free_access = (ramp == "Yes") or (elevator == "Yes")

    # collect everything that is missing or too small
    problems = []

    if not has_step_free_access:
        problems.append("there is no ramp or elevator (no step-free entrance)")

    if doorway_width < 75:
        problems.append("the doorway is narrower than 75 cm")
    elif doorway_width < 90:
        problems.append("the doorway is narrower than 90 cm")

    if accessible_toilet == "No":
        problems.append("there is no accessible toilet")

    if handrails == "No":
        problems.append("there are no handrails")

    # Explanation text output
    if len(problems) == 0:
        explanation_text = (
            "The facility has step-free access (ramp and/or elevator), "
            "an accessible toilet, handrails, and a doorway of at least 90 cm."
        )
    else:
        problems_text = ""
        for problem in problems:
            if problems_text != "":
                problems_text = problems_text + ", "
            problems_text = problems_text + problem
        explanation_text = problems_text

    st.write("**Explanation:** " + explanation_text)

    # find what our labeling rules would give for these inputs
    if (not has_step_free_access) or doorway_width < 75:
        rule_label = "Not Accessible"
    elif accessible_toilet == "Yes" and handrails == "Yes" and doorway_width >= 90:
        rule_label = "Accessible"
    else:
        rule_label = "Partially Accessible"

    #  uncertain cases may happen, so we have this explanation
    # the model learns from a small dataset, so it can disagree with the rules
    if prediction != rule_label:
        st.info(
            "Note: the " + selected_model + " predicted **" + prediction + "**, "
            "but the accessibility rules above point to **" + rule_label + "**. "
            "The models learn from a small dataset, so they can disagree "
            "with the rules on unusual combinations. Treat this result with caution."
        )

# Model comparison table
st.divider()
st.subheader("Model Performance Comparison")
st.write("Evaluation scores of the two algorithms from training and testing.")

# The comparison table
comparison_table = pd.DataFrame(metrics).T

# Rename the metric columns to readable labels
comparison_table = comparison_table.rename(columns={
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "Macro F1",
    "cv_f1": "5-Fold CV F1"
})

# Marks the best model in the table
comparison_table = comparison_table.rename(
    index={best_model_name: best_model_name + " (Best)"}
)

st.dataframe(comparison_table.round(3))

# What each metric means
st.markdown("#### What these metrics mean")
st.markdown(
    "- **Accuracy:** how many did the model accurately determined the right label for a facility.\n"
    "- **Precision:** when a model predicted a label, is the output truly that accessible? "
    "A low precision means many false alarms.\n"
    "- **Recall:** out of all the facilities that really belong to a label, how many the model found. "
    "A low recall means many facilities were missed.\n"
    "- **Macro F1:** one score that balances precision and recall. "
    "It is averaged over the three labels so each label counts equally.\n"
    "- **5-Fold CV F1:** the F1 score averaged over 5 different train/test splits of the data. "
    "It is steadier than a single split, so it is used to pick the best model."
)

st.divider()

group_members = ["Crisostomo, Jerome M.", "Tividad, Eric Jr. V.", "Viado, Jose Miguel B."]

st.markdown("**Midterm Project | Made by Group 28:**")
for member_name in group_members:
    st.caption(member_name)