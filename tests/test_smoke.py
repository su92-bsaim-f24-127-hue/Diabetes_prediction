import os
import pandas as pd

def test_dataset_exists():
    assert os.path.exists("DiabetesDataSet.csv"), "Dataset file missing"
    df = pd.read_csv("DiabetesDataSet.csv", nrows=10)
    expected_cols = {'gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes'}
    assert expected_cols.issubset(set(df.columns)), f"Missing columns: {expected_cols - set(df.columns)}"

def test_model_artifact_exists():
    assert os.path.exists("model_svc.pkl"), "Trained model artifact model_svc.pkl not found"
