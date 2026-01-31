from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your trained model and feature names
model = joblib.load('titanic_rf_model.joblib')
feature_names = joblib.load('titanic_feature_names.joblib')

print("Expected features:", feature_names)  # Debug: see what model expects


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get raw data from form
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        embarked = request.form.get('embarked', 'S')  # Default to Southampton

        # Create engineered features (matching your training)
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0
        fare_per_person = fare / family_size if family_size > 0 else fare

        # Create full feature dictionary with all possible features
        data = {
            'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked_C': 1 if embarked == 'C' else 0,
            'Embarked_Q': 1 if embarked == 'Q' else 0,
            'Embarked_S': 1 if embarked == 'S' else 0,
            'FamilySize': family_size,
            'IsAlone': is_alone,
            'FarePerPerson': fare_per_person,
            'Age*Class': age * pclass
        }

        # Create DataFrame with exact feature order from training
        input_df = pd.DataFrame([data])

        # Reorder to match training features
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        result = "✅ SURVIVED" if prediction == 1 else "❌ DID NOT SURVIVE"
        confidence = max(probability) * 100

        return render_template('index.html',
                               prediction_text=f'{result}',
                               confidence_text=f'Confidence: {confidence:.1f}%')

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f'Error: {str(e)}')


if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)