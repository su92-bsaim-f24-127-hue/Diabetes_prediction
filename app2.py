from flask import Flask, request
import pickle
import pandas as pd
from sklearn.exceptions import NotFittedError

app = Flask(__name__)

# Load model
def load_model():
    try:
        with open("model_svc.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        return None

model = load_model()

# Feature order
feature_names = ['gender', 'age', 'hypertension', 'heart_disease',
                 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']


# HTML Page (Single File)
html_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Diabetes Prediction</title>
    <style>
        body { background:#f0f0f0; font-family: Arial; padding:20px; }
        .box {
            background:white; width:450px; margin:auto; padding:25px;
            border-radius:12px; box-shadow:0 0 10px #ccc;
        }
        input, select {
            width:100%; padding:8px; margin:6px 0; border-radius:5px;
        }
        button {
            width:100%; padding:10px; background:#008cba; color:white;
            border:none; border-radius:5px; margin-top:10px; cursor:pointer;
        }
        button:hover { background:#006c94; }
        .result { font-size:22px; margin-top:15px; text-align:center; }
    </style>
</head>
<body>

<div class="box">
    <h2>🧬 Diabetes Prediction</h2>

    <form method="POST">

        <label>Gender (0 = Female, 1 = Male)</label>
        <select name="gender">
            <option value="0">Female</option>
            <option value="1">Male</option>
        </select>

        <label>Age</label>
        <input type="number" step="0.1" name="age" required>

        <label>Hypertension (0 = No, 1 = Yes)</label>
        <select name="hypertension">
            <option value="0">No</option>
            <option value="1">Yes</option>
        </select>

        <label>Heart Disease (0 = No, 1 = Yes)</label>
        <select name="heart_disease">
            <option value="0">No</option>
            <option value="1">Yes</option>
        </select>

        <label>Smoking History (encoded integer)</label>
        <input type="number" name="smoking_history" min="0" max="10">

        <label>BMI</label>
        <input type="number" step="0.1" name="bmi">

        <label>HbA1c Level</label>
        <input type="number" step="0.1" name="HbA1c_level">

        <label>Blood Glucose Level</label>
        <input type="number" step="0.1" name="blood_glucose_level">

        <button type="submit">Predict</button>
    </form>

    {prediction}
</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = ""

    if request.method == "POST":
        try:
            values = []
            for f in feature_names:
                values.append(float(request.form.get(f)))

            input_df = pd.DataFrame([values], columns=feature_names)

            prediction = model.predict(input_df)[0]

            if prediction == 1:
                prediction_text = "<div class='result'>🟢 Positive (Diabetic)</div>"
            else:
                prediction_text = "<div class='result'>🔴 Negative (Not Diabetic)</div>"

        except NotFittedError:
            prediction_text = "<div class='result'>❌ Model is not trained.</div>"
        except Exception as e:
            prediction_text = f"<div class='result'>❌ Error: {e}</div>"

    return html_page.replace("{prediction}", prediction_text)


if __name__ == "__main__":
    app.run(debug=True)
