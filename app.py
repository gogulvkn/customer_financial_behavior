import numpy as np
import pandas as pd
import pickle
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load saved model and scaler using joblib
model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

df = pd.read_pickle('form_val.pkl')

@app.route('/')
def home():
    return render_template('index.html',form_data=df)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form in exact specified order
        features = [
            float(request.form['Annual_Income']),
            float(request.form['Monthly_Spending']),
            float(request.form['Savings']),
            float(request.form['Checking_Balance']),
            float(request.form['Credit_Score']),
            float(request.form['Credit_Card_Spending']),
            float(request.form['Loan_Amount']),
            float(request.form['Investment_Amount']),
            float(request.form['Debt_to_Income_Ratio']),
            float(request.form['Transaction_Frequency'])
        ]

        form_data=request.form
        
        # Scale features and run KMeans prediction
        input_data = np.array([features])
        scaled_data = scaler.transform(input_data)
        cluster = model.predict(scaled_data)[0]
        if cluster == 0:
            cluster_name = "Normal"
        else:
            cluster_name = "Premium"

        
        return render_template('index.html', prediction_text=f'Assigned Segment: Cluster {cluster_name}',form_data=form_data)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error processing input: {str(e)}',form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)