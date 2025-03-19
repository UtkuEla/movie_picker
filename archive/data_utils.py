##Load and return the movie dataset##

import pandas as pd

def load_data(file_path):
    return pd.read_parquet(file_path)