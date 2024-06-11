import yaml
from datetime import datetime as dt


def generate_config(file):
    with open(file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        config["datetime"] = dt.today().strftime("%Y_%m_%d-%H_%M")

    return config
