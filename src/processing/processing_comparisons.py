import pandas as pd

from .processing_utils import homogenize_columns, homogenize_values



def construct_users_infos(users_variables: list[str], users_values: list[tuple[str]]) -> list[dict[str, str]]:
    """Construct list of users informations
    
    Args:
        users_variables (list[str]): variables for users informations.
        users_values (list[tuple[str]]): values of users.
        
    Returns:
        list[dict[str, str]]: list of users informations.
    """
    
    users_infos = [dict(zip(users_variables, user_values)) for user_values in users_values]
    return users_infos

def construct_comparisons_filenames(path_files_comparisons: str, users_infos: list[dict[str, str]]) -> list[str]:
    """Construct list of comparisons filenames.

    Args:
        path_files_comparisons (str): template path to comparisons files.
        users_informations (list[dict[str, str]]): list of users informations.
    
    Returns:
        list[str]: list of comparisons filenames.
    """
    
    comparisons_filenames = [path_files_comparisons.format(**user_infos) for user_infos in users_infos]
    return comparisons_filenames



def add_user_infos(df_comparisons: pd.DataFrame, user_infos: dict[str, str]) -> pd.DataFrame:
    """Add user informations to the comparisons dataframe.
    
    Args:
        df_comparisons (pd.DataFrame): comparisons dataframe.
        user_infos (dict[str, str]): user informations.

    Returns:
        pd.DataFrame: comparisons dataframe with user informations added.
    """

    for user_variable, user_value in user_infos.items():
        df_comparisons[f"user_{user_variable}"] = user_value
    return df_comparisons

def derive_formatted_comparisons(df_comparisons: pd.DataFrame) -> pd.DataFrame:
    """Derive formatted comparisons from the comparisons dataframe.

    Args:
        df_comparisons (pd.DataFrame): comparisons dataframe.

    Returns:
        pd.DataFrame: formatted comparisons dataframe.
    """

    df_comparisons = (
        df_comparisons
            
        .assign(
            is_valid=lambda df: df["vote"].isin(["left", "right", "tie"]), 
            
            upvoted=lambda df: df["left"].where(df["vote"] == "left", df["right"]).where(df["is_valid"]), 
            downvoted=lambda df: df["right"].where(df["vote"] == "left", df["left"]).where(df["is_valid"]), 
            is_tie=lambda df: (df["vote"] == "tie").where(df["is_valid"]), 
        )
    )
    return df_comparisons


def process_df_comparisons(df_comparisons: pd.DataFrame, 
    left_columns: list[str]|None=None, right_columns: list[str]|None=None, 
    vote_columns: list[str]|None=None, 
    left_values: list[str]|None=None, right_values: list[str]|None=None, tie_values: list[str]|None=None, 
    nan_value: str|None=None, 
    user_infos: dict[str, str]|None=None, 
    ) -> pd.DataFrame:
    """Process the comparisons dataframe.

    Args:
        df_comparisons (pd.DataFrame): comparisons dataframe.
        left_columns (list[str], optional): left columns. Defaults to [].
        right_columns (list[str], optional): right columns. Defaults to [].
        comparison_columns (list[str], optional): comparison columns. Defaults to [].
        left_values (list[str], optional): left values. Defaults to [].
        right_values (list[str], optional): right values. Defaults to [].
        tie_values (list[str], optional): tie values. Defaults to [].
        nan_value (str, optional): nan value. Defaults to None.
        user_infos (dict[str, str], optional): user informations. Defaults to {}.
    
    Returns:
        pd.DataFrame: processed comparisons dataframe.
    """

    if left_columns is None:
        left_columns = []
    if right_columns is None:
        right_columns = []
    if vote_columns is None:
        vote_columns = []
    
    if left_values is None:
        left_values = []
    if right_values is None:
        right_values = []
    if tie_values is None:
        tie_values = []
    
    if user_infos is None:
        user_infos = {}

    df_comparisons = (
        df_comparisons

        .pipe(homogenize_columns, "left", left_columns)
        .pipe(homogenize_columns, "right", right_columns)
        .pipe(homogenize_columns, "vote", vote_columns)

        .pipe(homogenize_values, "vote", "left", left_values)
        .pipe(homogenize_values, "vote", "right", right_values)
        .pipe(homogenize_values, "vote", "tie", tie_values)
        .assign(vote=lambda df: df["vote"].fillna(nan_value))

        .pipe(derive_formatted_comparisons)

        .pipe(add_user_infos, user_infos)
    )
    return df_comparisons
