import numpy as np
import pandas as pd
import typer

import sys


from src.data_preparation.data_augmentation import (augmented_data,
                                                    generate_augmented_data)
from src.data_preparation.data_preparation import to_dataframe
from src.utils.constants import DATA_DIR, AUG_DATA_DIR

app = typer.Typer()

@app.command()
def about(name: str):
    pass

@app.command()
def peekup_data_table(num_row: int):
    df = to_dataframe(data_dir=DATA_DIR)
    print(df.head(num_row))

@app.command()
def peekup_data_image(category: str):
    pass

@app.command()
def generate_augmented_data(dir_path: str):
    pass

@app.command()
def visualize_model():
    pass

def main(name: str):
    print(f"Hello {name}")


if __name__ == "__main__":
    app()
