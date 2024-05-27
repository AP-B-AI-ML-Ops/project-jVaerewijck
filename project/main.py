from prefect import flow
from load.collect import collect_flow
from load.prep import prep_flow
from train.train import train_flow
from train.hpo import hpo_flow
from train.register import register_flow

HPO_EXPERIMENT_NAME = "keras-LSTM-hyperopt"
REG_EXPERIMENT_NAME = "keras-LSTM-best-models"
NUM_TRIALS = 5
EPOCHS = 1

@flow
def main_flow():
    collect_flow("./data/")

    prep_flow("./data/","./models/")

    train_flow("./models/",EPOCHS)
    hpo_flow("./models/",NUM_TRIALS,HPO_EXPERIMENT_NAME,EPOCHS)
    register_flow("./models/",NUM_TRIALS,REG_EXPERIMENT_NAME,HPO_EXPERIMENT_NAME,EPOCHS)

if __name__ == "__main__":
    main_flow()