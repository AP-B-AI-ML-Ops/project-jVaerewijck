from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os
from prefect import task, flow

@flow
def collect_flow(datapath:str):
    print('collect')
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file('murderaccountability/homicide-reports','database.csv',datapath)
    zip_file_path = os.path.join(datapath, "database.csv.zip")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(datapath)