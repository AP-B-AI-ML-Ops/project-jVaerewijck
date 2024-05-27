import os
import pickle
import mlflow
import optuna
import numpy as np
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect import task, flow
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

@task
def load_pickle(filename):
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
        sequences.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length])
    return np.array(sequences), np.array(targets)

@task
def train_and_log_model(X,sequence_length,X_train, y_train,X_val, y_val, params):
    RF_PARAMS = ['n_units', 'dropout_rate', 'learning_rate']
    input_shape=(sequence_length, X.shape[2])
    with mlflow.start_run():
        for param in RF_PARAMS:
            params[param] = int(params[param])

        model = Sequential()
        model.add(LSTM(params['n_units'], activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.Dropout(params['dropout_rate']))
        model.add(Dense(X.shape[2]))
        optimizer = Adam(learning_rate=params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse')
        
        history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), verbose=0)
        val_loss = history.history['val_loss'][-1]
        # Evaluate model on the validation and test sets
        mlflow.log_metric("test_loss", val_loss)

@task
def get_experiment_runs(top_n,hpo_experiment_name):
    # Retrieve the top_n model runs and log the models
    client = MlflowClient()
    experiment = client.get_experiment_by_name(hpo_experiment_name)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.val_loss ASC"]
    )
    return runs

@task
def select_best_model(top_n,experiment_name):
    client = MlflowClient()
    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(experiment_name)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.test_loss ASC"]
    )[0]
    return best_run

@flow
def register_flow(model_path: str, top_n: int, experiment_name:str,hpo_experiment_name):
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog()
    
    data = load_pickle(os.path.join(model_path, "data.pkl"))
    scaled_data = normalize(data)
    sequence_length = 5  # Number of previous rows to use for predicting the next row
    X, y = create_sequences(scaled_data, sequence_length)
    split_ratio = 0.8
    split = int(len(X) * split_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    runs = get_experiment_runs(top_n,hpo_experiment_name)
    for run in runs:
        train_and_log_model(X,sequence_length,X_train, y_train,X_test, y_test,params=run.data.params)
    best_run = select_best_model(top_n,experiment_name)
    # Register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.keras.register_model(model_uri, name="LSTM-best-model")
    print(run_id)