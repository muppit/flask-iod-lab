from flask import Flask, request, jsonify, render_template
import pandas as pd
import os

import joblib
from utils import preprocessor

app = Flask(__name__)
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    input_text = request.form['text']

    # Assuming your model expects a DataFrame for the input
    data = pd.DataFrame([input_text], columns=['text'])

    # Preprocess and predict
    predicted_sentiment = model.predict(data)[0]
    if predicted_sentiment == 1:
        output = 'positive'
    else:
        output = 'negative'

    return render_template('result.html', sentiment=f'Predicted sentiment of "{input_text}" is {output}.')


if __name__ == "__main__":
    app.run(debug=True)
