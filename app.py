import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS  # Enable CORS for React Native

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load the trained model, scaler, and label encoder
model = joblib.load("crop_recommendation_model.pkl")
scaler = joblib.load("scaler.pkl")
y_encoder = joblib.load("y_encoder.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate input fields
        required_fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Extract input features and convert to numpy array
        features = np.array([[float(data["N"]), float(data["P"]), float(data["K"]), 
                              float(data["temperature"]), float(data["humidity"]), 
                              float(data["ph"]), float(data["rainfall"])]])
        
        # Apply feature scaling
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)

        # Convert numerical prediction back to crop label
        predicted_crop_name = str(y_encoder.inverse_transform([prediction[0]])[0])

        return jsonify({"recommended_crop": predicted_crop_name})
    
    except ValueError:
        return jsonify({"error": "Invalid input: Ensure all inputs are numeric."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=6000)
