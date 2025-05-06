import pandas as pd



def homogenize_columns(df: pd.DataFrame, processed_column: str, raw_columns: list[str]) -> pd.DataFrame:
    """Homogenize columns names.
    
    Args:
        df (pd.DataFrame): dataframe.
        processed_column (str): processed column name.
        raw_columns (list[str]): raw column names.
    
    Returns:
        pd.DataFrame: dataframe with homogenized columns names.
    """

    df = df.rename(columns={raw_column: processed_column for raw_column in raw_columns})
    return df

def homogenize_values(df: pd.DataFrame, column: str, processed_value: str, raw_values: list[str]) -> pd.DataFrame:
    """Homogenize column values.

    Args:
        df (pd.DataFrame): dataframe.
        column (str): name of column to process.
        processed_value (str): processed value.
        raw_values (list[str]): raw values.

    Returns:
        pd.DataFrame: dataframe with homogenized values on column.
    """
    
    df[column] = df[column].replace({raw_value: processed_value for raw_value in raw_values})
    return df
