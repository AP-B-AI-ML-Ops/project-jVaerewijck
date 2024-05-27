import os
import mlflow
import pickle
import pandas as pd
import numpy as np


run_id = "41da613d33154f84b20793aa13157d35"
model_name = "LSTM-best-model"
model_version = None  # Use None to load the latest version of the model
data_path = "./models/"

model = mlflow.pyfunc.load_model(f"models:/LSTM-best-model/latest")
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
scaler = load_pickle(os.path.join(data_path, "scaler.pkl"))
encoders_and_columns = load_pickle(os.path.join(data_path, "encoders.pkl"))

columns = encoders_and_columns.pop('columns')
encoders = encoders_and_columns

def encode_categorical(df, columns, encoders):
    for col in columns:
        df[col] = encoders[col].transform(df[col])
    return df

# Function to inverse transform categorical columns
def inverse_transform_categorical(df, columns, encoders):
    for col in columns:
        df[col] = encoders[col].inverse_transform(df[col].astype(int))
    return df

# Function to predict the next situation given the last known data
def predict_next_situation(model, last_known_data, scaler, sequence_length, columns, encoders):
    last_known_data = encode_categorical(last_known_data, columns, encoders)
    last_known_data_scaled = scaler.transform(last_known_data)
    last_sequence = np.array([last_known_data_scaled[-sequence_length:]])
    next_prediction_scaled = model.predict(last_sequence)
    next_prediction = scaler.inverse_transform(next_prediction_scaled)
    next_prediction_df = pd.DataFrame(next_prediction, columns=last_known_data.columns)
    next_prediction_df = inverse_transform_categorical(next_prediction_df, columns, encoders)
    return next_prediction_df
sequence_length = 5
data = load_pickle(os.path.join(data_path, "data.pkl"))
last_known_data = data.iloc[-1]
# Predict the next situation
next_situation = predict_next_situation(model, last_known_data, scaler, sequence_length, columns, encoders)

print('Next predicted situation:')
print(next_situation)