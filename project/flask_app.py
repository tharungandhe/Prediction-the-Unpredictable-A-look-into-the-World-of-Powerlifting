import pandas as pd
import numpy as np
import xgboost
import pickle
import os
from flask import Flask, render_template, url_for, request 

app = Flask(__name__)
model = pickle.load(open(r'model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        input_feature = [float(x) for x in request.form.values()]
        features_values = [np.array(input_feature)]
        names = ['playerId', 'Sex', 'Equipment', 'Age', 'BodyweightKg', 'BestSquatKg', 'BestBenchKg']
        data = pd.DataFrame(features_values, columns=names)
        prediction = model.predict(data)
        text = "Estimated Deadlift for the builder is: "
        return render_template("result.html", prediction_text=text + str(prediction))
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
