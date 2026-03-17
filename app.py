from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
rf_model = pickle.load(open("model/random_forest.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    values = [float(x) for x in request.form.values()]
    final_input = np.array(values).reshape(1, -1)

    # Apply same scaling as training
    final_input = scaler.transform(final_input)

    age = int(values[0])
    trestbps = values[3]
    chol = values[4]
    thalach = values[7]

    # YES / NO prediction
    pred_class = rf_model.predict(final_input)[0]
    disease_result = "YES" if pred_class == 1 else "NO"

    # Probability
    probability = rf_model.predict_proba(final_input)[0][1] * 100
    probability = round(probability, 2)

    # Risk level
    if probability < 40:
        risk_level = "Low Risk"
        risk_color = "green"
    elif probability < 70:
        risk_level = "Medium Risk"
        risk_color = "orange"
    else:
        risk_level = "High Risk"
        risk_color = "red"

    # Explainable reasons
    reasons = []
    if age > 50:
        reasons.append("Age above 50 increases heart disease risk")
    if trestbps > 140:
        reasons.append("High resting blood pressure detected")
    if chol > 240:
        reasons.append("High cholesterol level observed")
    if thalach < 120:
        reasons.append("Low maximum heart rate during exercise")

    if not reasons:
        reasons.append("Most health parameters are within normal range")

    advice = [
        "Maintain a balanced and healthy diet",
        "Exercise regularly",
        "Avoid smoking and alcohol",
        "Consult a doctor for medical advice"
    ]

    return render_template(
        "result.html",
        disease_result=disease_result,
        probability=probability,
        risk_level=risk_level,
        risk_color=risk_color,
        age=age,
        reasons=reasons,
        advice=advice
    )


if __name__ == "__main__":
    app.run(debug=True)
