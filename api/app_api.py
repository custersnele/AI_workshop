from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import numpy as np
import keras
import tensorflow as tf

app = Flask(__name__)
CORS(app)

model = keras.models.load_model("model/time_series.keras")

@app.route('/hello')
def hello_world():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    values = data["values"]
    prev_X = np.array([values])  # Convert the values to a numpy array
    horizon = 5  # Number of future predictions
    result = predict_values(prev_X, horizon)
    return jsonify(result=result)


def predict_values(prev_X, horizon):
    print('input shape: ')
    print(prev_X.shape)
    # Process data if needed (reshape, scale, etc.)
    # Assuming your model works with input data X and produces predictions y_pred
    result = []

    for i in range(horizon):
        print(prev_X)
        y_pred = model.predict(prev_X)
        print(y_pred)
        print(y_pred.shape)
        # For example, adding 0.5 as the new value
        # Add new value and remove the first value
        prev_X = np.roll(prev_X, shift=-1, axis=1)
        print("roll:")
        print(prev_X)

        prev_X[0][-1] = y_pred
        print(prev_X)
        result.append(float(y_pred[0][0]))
    return result


if __name__ == '__main__':
    app.run(host='0.0.0.0')
