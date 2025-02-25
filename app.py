import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model, scaler, and label encoder
model = joblib.load("crop_recommendation_model.pkl")
scaler = joblib.load("scaler.pkl")
y_encoder = joblib.load("y_encoder.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract input features
        features = np.array([[data["N"], data["P"], data["K"], data["temperature"], 
                              data["humidity"], data["ph"], data["rainfall"]]])
        
        # Apply feature scaling
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)

        # Convert numerical prediction back to crop label
        predicted_crop_name = str(y_encoder.inverse_transform([prediction[0]])[0])

        return jsonify({"recommended_crop": predicted_crop_name})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=6000)
