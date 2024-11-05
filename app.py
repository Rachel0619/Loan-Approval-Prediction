import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template

app=Flask(__name__)
xgbmodel = pickle.load(open('models/xgbmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])

def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    output=xgbmodel.predict(data)
    print(output[0])
    return jsonify(output[0])

if __name__=='__main__':
    app.run(debug=True)
