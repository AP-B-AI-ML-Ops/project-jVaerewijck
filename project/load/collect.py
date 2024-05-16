from kaggle.api.kaggle_api_extended import KaggleApi
from prefect import task, flow

@flow
def collect_flow(datapath:str):
    print('collect')
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file('murderaccountability/homicide-reports','database.csv',datapath)

