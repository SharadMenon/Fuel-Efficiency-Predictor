from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('fuel_efficiency_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = pd.DataFrame([{
            'Model': request.form['Model'],
            'Maker': request.form['Maker'],
            'Type': request.form['Type'],
            'Fuel': request.form['Fuel'],
            'Transmission': request.form['Transmission'],
            'Drive': request.form['Drive']
            }])
        prediction = model.predict(input_data)
        result = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f"Predicted Fuel Efficiency: {result} km/l")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
