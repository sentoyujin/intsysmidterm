# Facility Accessibility Classification Application
A machine learning project that classifies a facility as Accessible, Partially Accessible, or Not Accessible based on its accessibility features. It compares two algorithms, a Decision Tree and a Random Forest, and includes a Streamlit interface for trying them out.

## About the Project
Given a few details about a facility (ramp, elevator, accessible toilet, handrails, accessible parking, and doorway width), the system predicts how accessible the facility is and explains the result in a short sentence. The system also uses two main algorithms (Decision tree and Random Forest) to compare their performance on predicting a facility accessibility. 

## Contents
- **`models_training.py`** - where both models are trained, evaluated through its metrics, and saves the bundle and comparison results.
- **`app.py`** - the Streamlit interface.
- **`Facility_Accessiliblity_Classification.knwf`** - the KNIME workflow of the project.
- **`model_bundle.joblib`** - the saved trained models and scores to be used by **`app.py`**.
- **`facilities.csv`** - the main dataset used to train the models.
- **`comparison_result.csv`** - comparison evaluation scores on both models.

## How to Run
### Prerequisites
Before running the application or scripts,
1. Install the required libraries through this command:
```bash 
python -m pip install pandas scikit-learn joblib streamlit
```
2. Make sure `model_bundle.joblib` and `facilities.csv` is installed and is in the same folder as the scripts are.

### 1. Running the Streamlit App
- Open the terminal of the project and enter this command:
```bash
python -m streamlit run app.py
```
The app opens in your browser at http://localhost:8501. To stop it, press Ctrl+C in the terminal or simply close it.
`app.py` loads model_bundle.joblib, so keep both files in the same folder.

### 2. Retraining the models (optional)
- Simply press the run code button for the `models_training.py` to re-train the models. This reads `facilities.csv`, trains both models, prints the evaluation scores, and overwrites `model_bundle.joblib` and `comparison_results.csv`. 

### 3. Opening the KNIME workflow
- Import `Facility_Accessiliblity_Classification.knwf` to the KNIME app, open the CSV Reader node, point it at your `facilities.csv` copy, and press Execute all.

## Model Comparison Results
Based on the evaluation metric results, The *Random Forest* performed best, scoring 91.7% accuracy and a 0.915 macro F1 on the test data, compared with 83.3% and 0.815 for the *Decision Tree*. It also led in 5-fold cross-validation (0.867 vs 0.775). Because the test set has only 12 facilities, the gap between the two models is based on just one or two predictions, so it should be read with caution. Kindly refer to `comparison_results.csv` to view the full breakdown of evaluation scores on both models.

## Credits
**CS0065 - Intelligent Systems: Midterm Project**

Made by Group 28 | AN 42:
- *Crisostomo, Jerome M.*
- *Tividad, Eric Jr. V.*
- *Viado, Jose Miguel B.*