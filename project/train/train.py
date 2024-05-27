import os
import pickle
import mlflow
import pandas as pd
import numpy as np
from prefect import task, flow
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

@task
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)
    
@task
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@task
def normalize(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data,scaler

@task
def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length])
    return np.array(sequences), np.array(targets)

@task
def create_model(X,sequence_length):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, X.shape[2])))
    model.add(Dense(X.shape[2]))
    model.compile(optimizer='adam', loss='mse')
    return model

@task
def start_ml_experiment(model,X_train, y_train,epochs):
    with mlflow.start_run():
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)

@flow
def train_flow(data_path: str,epochs:int):
    mlflow.set_experiment("keras-LSTM")
    mlflow.sklearn.autolog()

    data = load_pickle(os.path.join(data_path, "data.pkl"))
    scaled_data, scaler = normalize(data)
    dump_pickle(scaler,os.path.join(data_path, "scaler.pkl"))
    sequence_length = 5  # Number of previous rows to use for predicting the next row
    X, y = create_sequences(scaled_data, sequence_length)
    split_ratio = 0.8
    split = int(len(X) * split_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = create_model(X,sequence_length)
    start_ml_experiment(model,X_train, y_train,epochs)