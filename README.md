# Diabetes Prediction System

An end-to-end Machine Learning project predicting diabetes risk from clinical indicators.

## Overview
This project trains a Support Vector Classifier (SVC) on a 100,000-row clinical dataset, achieving 94.9% test accuracy. It includes both Streamlit and Flask applications for interactive prediction.

## Features Used
- gender, age
- hypertension, heart_disease
- smoking_history
- bmi, HbA1c_level, blood_glucose_level

## Running the Applications

### Streamlit Dashboard
`ash
pip install -r requirements.txt
streamlit run app.py
`

### Flask Web Interface
`ash
python app2.py
`

## Structure
- pp.py: Streamlit user interface
- pp2.py: Flask web application
- model_svc.pkl: Serialized trained SVC model
- DiabetesDataSet.csv: Patient dataset
- project.ipynb: Exploratory data analysis and model training
- 
equirements.txt: Project dependencies
- 	ests/test_smoke.py: Verification tests
