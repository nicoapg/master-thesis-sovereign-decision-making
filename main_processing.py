import logging
import argparse

import pandas as pd

from src import config
from src.processing import processing_principles, processing_comparisons
from src.utils import utils_paths



def setup_argparser() -> argparse.ArgumentParser:
    """Setup the argparser for the main processing script."""

    # Initialize argparser
    argparser = argparse.ArgumentParser(
        prog="Main processing", 
        description="Script to process a dataset.", 
    )

    # Add dataset argument(s)
    argparser.add_argument(
        "--dataset_name", 
        type=str, 
        required=True, 
        help="Name of the dataset", 
    )
    
    return argparser



if __name__ == "__main__":
    logging.info("Main processing...")

    # Parse arguments
    args = setup_argparser().parse_args()
    logging.info("dataset: %s", args.dataset_name)

    # Retrieve referential
    dataset_referential = config.DATASETS_REFERENTIAL[args.dataset_name]


    # Process principles
    logging.info("processing principles...")

    path_raw_principles = utils_paths.compose_path_raw_file(
        path_data_raw=config.PATH_DATA_RAW, 
        dataset_name=args.dataset_name, 
        file_name=dataset_referential["PATH_FILE_PRINCIPLES"], 
    )

    df_principles = (
        pd.read_excel(path_raw_principles)
        .pipe(
            processing_principles.process_df_principles, 
            principle_columns=config.PRINCIPLE_COLUMNS, 
        )
    )


    # Process comparisons
    logging.info("processing comparisons...")

    users_infos = processing_comparisons.construct_users_infos(
        users_variables=dataset_referential["USERS_VARIABLES"], 
        users_values=dataset_referential["USERS_VALUES"], 
    )
    
    comparisons_filenames = processing_comparisons.construct_comparisons_filenames(
        path_files_comparisons=dataset_referential["PATH_FILES_COMPARISONS"], 
        users_infos=users_infos, 
    )

    dfs_comparisons = []
    for user_infos, comparisons_filename in zip(users_infos, comparisons_filenames):

        path_raw_comparisons = utils_paths.compose_path_raw_file(
            path_data_raw=config.PATH_DATA_RAW, 
            dataset_name=args.dataset_name, 
            file_name=comparisons_filename, 
        )

        df_comparisons = (
            pd.read_excel(path_raw_comparisons)
            .pipe(
                processing_comparisons.process_df_comparisons, 
                left_columns=config.LEFT_COLUMNS, right_columns=config.RIGHT_COLUMNS, 
                vote_columns=config.VOTE_COLUMNS, 
                left_values=config.LEFT_VALUES, right_values=config.RIGHT_VALUES, tie_values=config.TIE_VALUES, 
                nan_value=config.NAN_VALUE, 
                user_infos=user_infos, 
            )
        )
        
        dfs_comparisons.append(df_comparisons)
    df_comparisons = pd.concat(dfs_comparisons, ignore_index=True)


    # Save processed dataset

    path_processed_principles = utils_paths.compose_path_processed_princples(
        path_data_processed=config.PATH_DATA_PROCESSED, 
        dataset_name=args.dataset_name, 
    )
    df_principles.to_csv(path_processed_principles, index=False)
    
    path_processed_comparisons = utils_paths.compose_path_processed_comparisons(
        path_data_processed=config.PATH_DATA_PROCESSED, 
        dataset_name=args.dataset_name, 
    )
    df_comparisons.to_csv(path_processed_comparisons, index=False)
