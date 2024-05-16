from prefect import flow
from load.collect import collect_flow

@flow
def main_flow():
    collect_flow('./data/')

if __name__ == "__main__":
    main_flow()