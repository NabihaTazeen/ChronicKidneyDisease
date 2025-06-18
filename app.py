from flask import Flask, render_template, request
import numpy as np
import pickle
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from form input (24 features as per your dataset)
        features = [float(request.form[field]) for field in [
            "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
            "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
            "htn", "dm", "cad", "appet", "pe", "ane"
        ]]

        # Convert features into NumPy array and reshape for prediction
        final_features = np.array(features).reshape(1, -1)

        # Apply scaling to the features (since train.py used StandardScaler)
        final_features_scaled = scaler.transform(final_features)

        # Make prediction using the scaled features
        prediction = model.predict(final_features_scaled)[0]  # Get predicted value

        # Map prediction result
        result = "Chronic Kidney Disease Detected" if prediction == 1 else "No CKD Detected"

        # Generate a bar chart for feature visualization
        feature_names = [
            "Age", "BP", "SG", "AL", "SU", "RBC", "PC", "PCC", "BA",
            "BGR", "BU", "SC", "SOD", "POT", "Hemo", "PCV", "WC", "RC",
            "HTN", "DM", "CAD", "Appet", "PE", "ANE"
        ]

        plt.figure(figsize=(12, 6))
        plt.bar(feature_names, features, color="skyblue")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Features")
        plt.ylabel("Values")
        plt.title("Input Features for CKD Prediction")

        # Save the plot to a string buffer
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()

        return render_template("results.html", result=result, graph_url=graph_url)
    
    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
    
  # ckd values  24.0, 76, 1.015, 2.0, 4.0, 1, 0, 0, 0, 410.0, 31.0, 1.1, 137., 4, 12.4, 31, 62, 29, 0, 1, 1, 0, 1, 0
  
  # non ckd values 45.0, 70.0, 1.020, 0.0, 0.0, 1, 1, 0, 0, 110.0, 25.0, 1.0, 141.0, 4.5, 15.0, 34, 60, 32, 0, 0, 0, 0, 0, 0