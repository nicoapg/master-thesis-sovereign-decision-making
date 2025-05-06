import pandas as pd
from typing import Tuple
import logging
import os
from pathlib import Path

# Get the root directory of the project
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data" / "20250505-fiona-db-exports"

def read_sgmlp_csv(file_path: str) -> Tuple[pd.DataFrame, list]:
    """
    Read a SGMLP CSV file and return the dataframe along with its column titles.
    This function can handle both experiments and results CSV files.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        Tuple containing:
        - pandas DataFrame with the relevant data
        - List of column titles
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Get column titles
        column_titles = df.columns.tolist()
        
        # Get just the filename from the path
        filename = os.path.basename(file_path)
        
        # Log the column titles
        logging.debug(f"Column titles in {filename}:")
        for title in column_titles:
             logging.debug(f"- {title}")
            
        return df, column_titles
        
    except Exception as e:
        logging.error(f"Error reading CSV file {os.path.basename(filename)}: {str(e)}")
        raise
        