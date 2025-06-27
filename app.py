from flask import Flask, request, render_template, redirect, url_for, flash
import pickle
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for flashing messages

# Load model and normalizer
model_path = os.path.join(os.path.dirname(__file__), 'rf_acc_68.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'normalizer.pkl')

# Ensure files exist
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Model or scaler file not found. Please train the model first.")

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        input_values = list(request.form.values())

        # Check for missing fields
        if any(v.strip() == '' for v in input_values):
            flash("⚠️ All fields are required.")
            return redirect(url_for('home'))

        # Convert inputs to float list
        features = [float(v) for v in input_values]

        # Preprocess inputs
        scaled = scaler.transform([features])

        # Make prediction
        prediction = model.predict(scaled)
        proba = model.predict_proba(scaled)[0]

        # Handle binary or single-class probability
        if len(proba) == 2:
            probability = proba[1] * 100
        else:
            probability = proba[0] * 100

        # Prepare result string
        result = '✅ Liver Cirrhosis Detected' if prediction[0] == 1 else '✅ No Liver Cirrhosis'
        confidence = f"Prediction Confidence: {probability:.2f}%"

        return render_template('inner-page.html', prediction=result, confidence=confidence)

    except ValueError:
        flash("❌ Please enter valid numerical values.")
        return redirect(url_for('home'))

    except Exception as e:
        return render_template('inner-page.html', prediction=f"❌ Error: {str(e)}", confidence="")

if __name__ == '__main__':
    app.run(debug=True)
