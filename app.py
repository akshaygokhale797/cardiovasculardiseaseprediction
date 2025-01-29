from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib  # Save and load your trained model
import pandas as pd

# Load the trained model
from cardio_model_FE import pipeline  # Ensure pipeline is accessible
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Serve your HTML page

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        # Create DataFrame from user input
        user_data = pd.DataFrame([{
            'age_years': data['age'],
            'gender': data['gender'],
            'height': data['height'],
            'weight': data['weight'],
            'ap_hi': data['ap_hi'],
            'ap_lo': data['ap_lo'],
            'cholesterol': data['cholesterol'],
            'gluc': data['glucose'],
            'smoke': data['smoke'],
            'alco': data['alcohol'],
            'active': data['active'],
        }])

        # Calculate BMI
        user_data['bmi'] = user_data['weight'] / (user_data['height'] / 100) ** 2

        # Derive `bp_category_encoded`
        def categorize_bp(ap_hi, ap_lo):
            if ap_hi < 120 and ap_lo < 80:
                return 'Normal'
            elif 120 <= ap_hi <= 129 and ap_lo < 80:
                return 'Elevated'
            elif (130 <= ap_hi <= 139 or 80 <= ap_lo <= 89):
                return 'Hypertension Stage 1'
            elif (140 <= ap_hi or 90 <= ap_lo):
                return 'Hypertension Stage 2'
            else:
                return 'Hypertensive Crisis'

        user_data['bp_category_encoded'] = user_data.apply(
            lambda row: categorize_bp(row['ap_hi'], row['ap_lo']), axis=1)

        # Derive `age_range_gender`
        age_bins = [0, 20, 40, 60, 80, 100]
        age_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
        user_data['age_range_gender'] = user_data.apply(
            lambda row: f"{'Male' if row['gender'] == 2 else 'Female'}-{pd.cut([row['age_years']], bins=age_bins, labels=age_labels)[0]}",
            axis=1
        )

        # Make prediction
        prediction = pipeline.predict(user_data)
        result = int(prediction[0])  # Convert numpy output to native int

        return jsonify({'risk': result})  # Return the prediction
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
