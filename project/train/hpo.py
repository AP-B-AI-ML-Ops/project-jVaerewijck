import os
import pickle
import mlflow
import optuna
import numpy as np
from prefect import task, flow
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


@task
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@task
def normalize(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


@task
def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i : i + sequence_length])
        targets.append(data[i + sequence_length])
    return np.array(sequences), np.array(targets)


@task
def create_model(trial, input_shape):
    n_units = trial.suggest_int("n_units", 32, 128)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    model = Sequential()
    model.add(LSTM(n_units, activation="relu", input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(Dense(X.shape[2]))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    return model


@task
def optimize(X, sequence_length, X_train, y_train, X_val, y_val, num_trials, epochs):
    def objective(trial):
        input_shape = (sequence_length, X.shape[2])
        n_units = trial.suggest_int("n_units", 32, 128)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=2,
            verbose=1,
            mode="min",
            restore_best_weights=False,
        )
        with mlflow.start_run():
            model = Sequential()
            model.add(LSTM(n_units, activation="relu", input_shape=input_shape))
            model.add(tf.keras.layers.Dropout(dropout_rate))
            model.add(Dense(X.shape[2]))
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss="mse")

            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                validation_data=(X_val, y_val),
                verbose=0,
                callbacks=[early_stopping],
            )
            val_loss = history.history["val_loss"][-1]

            mlflow.log_params(trial.params)
            mlflow.log_metric("val_loss", val_loss)

        return val_loss

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


@flow
def hpo_flow(data_path: str, num_trials: int, experiment_name: str, epochs: int):
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog()

    data = load_pickle(os.path.join(data_path, "data.pkl"))
    scaled_data = normalize(data)
    sequence_length = 5  # Number of previous rows to use for predicting the next row
    X, y = create_sequences(scaled_data, sequence_length)
    split_ratio = 0.8
    split = int(len(X) * split_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    optimize(X, sequence_length, X_train, y_train, X_test, y_test, num_trials, epochs)
