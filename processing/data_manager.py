import pandas as pd
from pathlib import Path

def load_dataset(*, file_name: str):
    data = pd.read_csv(Path(f"DATASET_DIR/{file_name}"))
    return data


