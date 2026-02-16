from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Load model and encoder
model = joblib.load("model.pkl")
ohe = joblib.load("ohe.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "AI Network Failure Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame([data])
        print("\n🔍 Received Input Data:\n", df)  # Debug print

        # Handle missing Network Type
        if "Network Type" not in df.columns:
            return jsonify({"error": "Missing 'Network Type' field"}), 400
        
        df['Network Type'] = df['Network Type'].fillna('Unknown')

        # Apply one-hot encoding
        encoded_nt = ohe.transform(df[['Network Type']])
        encoded_nt_df = pd.DataFrame(encoded_nt, columns=ohe.get_feature_names_out(['Network Type']))
        print("\n✅ One-Hot Encoded Features:\n", encoded_nt_df)  # Debug print
        
        # Ensure all training features exist (fill missing columns with 0)
        expected_features = ohe.get_feature_names_out(['Network Type'])
        for feature in expected_features:
            if feature not in encoded_nt_df.columns:
                encoded_nt_df[feature] = 0  # Add missing feature with value 0

        # Merge encoded data
        df = pd.concat([df, encoded_nt_df], axis=1)
        df.drop(columns=['Network Type'], inplace=True)

        # Ensure correct feature order
        trained_features = model.feature_names_in_  # Features used during training
        print("\n🔹 Model Trained Features:\n", trained_features)  # Debug print
        print("\n📌 Features in Current Input Data:\n", df.columns)  # Debug print

        # Reorder to match training data and fill missing features with 0
        df = df.reindex(columns=trained_features, fill_value=0)

        # Final Check Before Prediction
        print("\n🚀 Final Processed Input Data:\n", df)  # Debug print

        # Predict
        prediction = model.predict(df)
        return jsonify({"prediction": int(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5001, debug=True)

