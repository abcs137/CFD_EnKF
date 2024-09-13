import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor

def get_csv_files_from_folder(folder_path):
    """
    Retrieves a list of all CSV files in a given folder.

    Parameters:
    - folder_path: Path to the folder containing CSV files.

    Returns:
    - List of CSV file paths.
    """
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]



def read_csv_file(file):
    """Reads a single CSV file and returns a DataFrame with an added filename attribute."""
    df = pd.read_csv(
        file,
        delim_whitespace=False,
        on_bad_lines='skip',
        header=3,
        skipinitialspace=True,
        index_col=False,
        dtype={
            'X [ m ]': np.float64,
            'Y [ m ]': np.float64,
            'Z [ m ]': np.float64,
            'Total Pressure [ Pa ]': np.float64
        }
    )
    df.filename = file
    return df

def read_csv_files(file_list):
    """Reads a list of CSV files and returns a list of DataFrames with an added filename attribute."""
    if isinstance(file_list, str):
        # If a single file path is passed, convert it to a list
        file_list = [file_list]

    with ProcessPoolExecutor() as executor:
        dfs = list(executor.map(read_csv_file, file_list))

    return dfs

