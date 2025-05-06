from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)



# Load model, imputer, and scaler
model = joblib.load("classifier.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")

# Feature and categorical mapping
FEATURES = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
            'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
            'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

categorical_mappings = {
    'rbc': {'normal': 0, 'abnormal': 1},
    'pc': {'normal': 0, 'abnormal': 1},
    'pcc': {'notpresent': 0, 'present': 1},
    'ba': {'notpresent': 0, 'present': 1},
    'htn': {'no': 0, 'yes': 1},
    'dm': {'no': 0, 'yes': 1, ' yes': 1, '\tno': 0, '\tyes': 1},
    'cad': {'no': 0, 'yes': 1, '\tno': 0},
    'appet': {'good': 1, 'poor': 0},
    'pe': {'no': 0, 'yes': 1},
    'ane': {'no': 0, 'yes': 1}
}

def preprocess_input(data):
    for col in FEATURES:
        if col not in data.columns:
            data[col] = np.nan
    for col, mapping in categorical_mappings.items():
        if col in data.columns:
            data[col] = data[col].map(mapping)

    data = data.apply(pd.to_numeric, errors='coerce')
    data = data[FEATURES]

    imputed = imputer.transform(data)
    scaled = scaler.transform(imputed)
    return scaled

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST" and "predict_single" in request.form:
        input_data = {col: request.form.get(col) for col in FEATURES}
        input_df = pd.DataFrame([{
    k: (v if v.strip() != "" else np.nan) for k, v in input_data.items()
}])
        processed = preprocess_input(input_df)
        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]

    return render_template("index.html", features=FEATURES, prediction=prediction, probability=probability)

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    file = request.files["csv_file"]
    if file:
        df = pd.read_csv(file)
        processed = preprocess_input(df)
        preds = model.predict(processed)
        df["Prediction"] = ["CKD" if p == 1 else "No CKD" for p in preds]
        result_file = "bulk_predictions.csv"
        df.to_csv(result_file, index=False)
        return send_file(result_file, as_attachment=True)

    return redirect("/")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
