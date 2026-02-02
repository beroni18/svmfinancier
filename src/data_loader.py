import pandas as pd

def load_data(path="../data/dataset_financier.csv"):
    """
    Charge le dataset financier depuis le chemin indiqué.
    """
    df = pd.read_csv(path)
    return df
