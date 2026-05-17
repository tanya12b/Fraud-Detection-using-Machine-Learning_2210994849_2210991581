from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('models/saved_model.pkl')

@app.route('/')
def home():
    return 'Fraud Detection API Running Successfully!'

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json

    values = list(data.values())

    prediction = model.predict([values])[0]

    probability = model.predict_proba([values])[0][1]

    return jsonify({
        'prediction': int(prediction),
        'fraud_probability': float(probability)
    })

if __name__ == '__main__':
    app.run(debug=True)