import pandas as pd

from .processing_utils import homogenize_columns



def process_df_principles(df_principles: pd.DataFrame, 
    principle_columns: list[str]|None=None, 
    ) -> pd.DataFrame:
    """Process the principles dataframe.

    Args:
        df_principles (pd.DataFrame): principles dataframe.
        principle_columns (list[str], optional): principle columns. Defaults to [].

    Returns:
        pd.DataFrame: processed principles dataframe.
    """
    
    if principle_columns is None:
        principle_columns = []

    df_principles = (
        df_principles
        
        .pipe(homogenize_columns, "principle", principle_columns)
    )
    return df_principles
