from prefect import flow
from load.collect import collect_flow
from load.prep import prep_flow
from train.train import train_flow
from train.hpo import hpo_flow
from train.register import register_flow

HPO_EXPERIMENT_NAME = "keras-LSTM-hyperopt"
REG_EXPERIMENT_NAME = "keras-LSTM-best-models"


@flow
def main_flow():
    collect_flow("./data/")

    prep_flow("./data/","./models/")

    train_flow("./models/")
    hpo_flow("./models/",50,HPO_EXPERIMENT_NAME)
    register_flow("./models/",5,REG_EXPERIMENT_NAME,HPO_EXPERIMENT_NAME)

if __name__ == "__main__":
    main_flow()