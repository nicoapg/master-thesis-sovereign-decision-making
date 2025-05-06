import pandas as pd



def append_last_rank(df: pd.DataFrame, rank_column: str="rank", 
    dataset_params_columns: list[str]|None=None) -> pd.DataFrame:
    """Append the last rank column.
    
    Args:
        df (pd.DataFrame): dataframe.
        rank_column (str, optional): rank column. Defaults to "rank".
        dataset_params_columns (list[str]|None, optional): dataset parameters columns. Defaults to None.
        
    Returns:
        pd.DataFrame: dataframe with the last rank column."""
    
    if dataset_params_columns is None:
        dataset_params_columns = []

    df["rank_last"] = (
        df
        
        .groupby(["dataset", *dataset_params_columns, "ranker", "principle", "run"])
        [rank_column]
        .transform("last")
    )
    return df

def append_best_rank(df: pd.DataFrame, last_rank_column: str="rank_last", 
    dataset_params_columns: list[str]|None=None) -> pd.DataFrame:
    """Append the best rank column.

    Args:
        df (pd.DataFrame): dataframe.
        last_rank_column (str, optional): last rank column. Defaults to "rank_last".
        dataset_params_columns (list[str]|None, optional): dataset parameters columns. Defaults to None.

    Returns:
        pd.DataFrame: dataframe with the best rank column.
    """

    if dataset_params_columns is None:
        dataset_params_columns = []

    df["rank_best"] = (
        df
        
        .groupby(["dataset", *dataset_params_columns, "ranker", "principle"])
        [last_rank_column]
        .transform(lambda s: s.value_counts().idxmax())
    )
    return df
