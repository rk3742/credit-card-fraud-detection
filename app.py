from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import gdown

# Auto download model if not present
if not os.path.exists('model.pkl'):
    print("Downloading model from Google Drive...")
    gdown.download(
        'https://drive.google.com/uc?id=1k1Iwhqovki7QJLMSp0T4-ArjcnGuIZXw',
        'model.pkl',
        quiet=False
    )
    print("Model downloaded! ✅")

# Auto download scaler if not present
if not os.path.exists('scaler.pkl'):
    print("Downloading scaler from Google Drive...")
    gdown.download(
        'https://drive.google.com/uc?id=1rbeqZJhRlZNPGWt-urQ0kYqrIRLX9HyD',
        'scaler.pkl',
        quiet=False
    )
    print("Scaler downloaded! ✅")

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load a sample of dataset for simulation
print("Loading sample data for simulation...")
df_sample = pd.read_csv('sample_data.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        amount = float(data['amount'])
        time = float(data['time'])
        v_features = [float(data[f'v{i}']) for i in range(1, 29)]

        amount_scaled = scaler.transform([[amount]])[0][0]
        time_scaled = scaler.transform([[time]])[0][0]

        features = [time_scaled] + v_features + [amount_scaled]
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1] * 100

        return jsonify({
            'prediction': int(prediction),
            'probability': round(probability, 2),
            'status': 'FRAUD' if prediction == 1 else 'LEGITIMATE'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/simulate', methods=['GET'])
def simulate():
    sample = df_sample.sample(1).iloc[0]
    result = {
        'time': float(sample['Time']),
        'amount': float(sample['Amount']),
        'actual': int(sample['Class'])
    }
    for i in range(1, 29):
        result[f'v{i}'] = float(sample[f'V{i}'])
    return jsonify(result)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)