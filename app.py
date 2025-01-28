import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the model and scaler
with open('Model/regmodel.pkl', 'rb') as f:
    regmodel = pickle.load(f)

with open('Model/scaling.pkl', 'rb') as f:
    scalar = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    # Transform input data
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    # Predict using the loaded model
    output = regmodel.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    # Transform input data
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    # Predict using the loaded model
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text="The house price prediction is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)
