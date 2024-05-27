import os
import pickle
import pandas as pd
import zipfile
from prefect import flow,task
from sklearn.feature_extraction import DictVectorizer

@task
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

@flow
def unzip_file(data_path: str):
    zip_file_path = os.path.join(data_path, "database.csv.zip")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)


@task
def read_dataframe(filename: str):
    df = pd.read_csv(filename)

    return df


@flow
def prep_flow(data_path: str, dest_path: str):
    unzip_file(data_path)
    # Load parquet files
    df = read_dataframe("database.csv")
    

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    dump_pickle(df,os.path.join(dest_path, "data.pkl"))