import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # Allow all origins for testing

# Load the trained model, scaler, and label encoder
try:
    model = joblib.load("crop_recommendation_model.pkl")
    scaler = joblib.load("scaler.pkl")
    y_encoder = joblib.load("y_encoder.pkl")
    print("✅ Model, scaler, and encoder loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model files: {e}")

# Define feature names (same as in training data)
feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        print("📥 Received Data:", data)  # Debugging

        # Convert input data to DataFrame with proper float conversion
        features_df = pd.DataFrame([[
            float(data["N"]), float(data["P"]), float(data["K"]), 
            float(data["temperature"]), float(data["humidity"]), 
            float(data["ph"]), float(data["rainfall"])
        ]], columns=feature_names)

        print("🔍 Converted DataFrame:", features_df)  # Debugging

        # Apply feature scaling
        scaled_features = scaler.transform(features_df)

        # Make prediction
        prediction = model.predict(scaled_features)

        # Convert prediction to crop label
        predicted_crop_name = str(y_encoder.inverse_transform([prediction[0]])[0])

        print(f"🌾 Recommended Crop: {predicted_crop_name}")  # Debugging

        return jsonify({"recommended_crop": predicted_crop_name})

    except Exception as e:
        print(f"❌ Error during prediction: {e}")  # Log errors in Render logs
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=6000)
