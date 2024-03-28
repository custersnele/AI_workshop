import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import keras
import tensorflow as tf

print(tf.__version__)
model = keras.models.load_model("model/time_series.keras")
seq_len = 20

def predict(data, horizon):
    for i in range(horizon):
        predict = predict_values(np.array([data[-seq_len:].values]), 1)
        print("INPUT TO PREDICT")

        # Display the predicted values on the graph
        print("PREDICTED")
        #combined_data = pd.concat((data, pd.DataFrame(predict)), axis=0)
        data = pd.concat([data, pd.DataFrame(predict)], axis=0, ignore_index=True)
    print(data.shape)
    st.write(data)  # Display uploaded data

    st.line_chart(data)  # Display original values and predictions

# Function to make predictions using the loaded model
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
        result.append(y_pred[0][0])
    return result


@st.cache_data
def get_data():
    file = "data/time_series_data.csv"
    return pd.read_csv(file, header=None)



st.title('Forecasting')
data = get_data()

st.write(data)  # Display uploaded data

# Display time series data in a graph


horizon = st.slider('Choose the horizon', min_value=5, max_value=250, value=5, step=5)

st.line_chart(data)


forecast_btn = st.button('Forecast')

if forecast_btn:
    predict(data, horizon)

