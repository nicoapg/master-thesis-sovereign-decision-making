import os



# utils

def ensure_path_directory(path_directory: str) -> None:
    """Ensure path to directory will exist.

    Args:
        path_directory (str): path to directory.
    """

    if not os.path.exists(path_directory):
        os.makedirs(path_directory)

def ensure_path_file(path_file: str) -> None:
    """Ensure path to file will exist.

    Args:
        path_file (str): path to file.
    """

    path_directory = os.path.dirname(path_file)
    ensure_path_directory(path_directory)


# Raw data

def compose_path_raw_file(path_data_raw: str, dataset_name: str, file_name: str) -> str:
    """Compose path to raw file.

    Args:
        path_data_raw (str): path to raw data folder.
        dataset_name (str): name of the dataset.
        file_name (str): name of the file.
    
    Returns:
        str: path to raw file.
    """

    path_raw_file = os.path.join(path_data_raw, dataset_name, file_name)
    return path_raw_file


# Processed data

def compose_path_processed_file(path_data_processed: str, dataset_name: str, file_name: str) -> str:
    """Compose path to processed file.

    Args:
        path_data_processed (str): path to processed data folder.
        dataset_name (str): name of the dataset.
        file_name (str): name of the file.
    
    Returns:
        str: path to raw file.
    """

    path_processed_file = os.path.join(path_data_processed, dataset_name, f"{dataset_name}_{file_name}")
    ensure_path_file(path_processed_file)
    return path_processed_file

def compose_path_processed_princples(path_data_processed: str, dataset_name: str) -> str:
    """Compose path to processed principles file.

    Args:
        path_data_processed (str): path to processed data folder.
        dataset_name (str): name of the dataset.
    
    Returns:
        str: path to processed principles file.
    """

    return compose_path_processed_file(path_data_processed, dataset_name, "principles.csv")

def compose_path_processed_comparisons(path_data_processed: str, dataset_name: str) -> str:
    """Compose path to processed comparisons file.

    Args:
        path_data_processed (str): path to processed data folder.
        dataset_name (str): name of the dataset.
    
    Returns:
        str: path to processed comparisons file.
    """

    return compose_path_processed_file(path_data_processed, dataset_name, "comparisons.csv")


# Results

def compose_path_result_file(path_results: str, dataset_name: str, benchmark_name: str, file_name: str) -> str:
    """Compose path to result file.

    Args:
        path_results (str): path to results folder.
        dataset_name (str): name of the dataset.
        benchmark_name (str): name of the benchmark.
        file_name (str): name of the file.

    Returns:
        str: path to result file.
    """

    path_result_file = os.path.join(path_results, dataset_name, benchmark_name, file_name)
    ensure_path_file(path_result_file)
    return path_result_file

def compose_path_result_fitting_durations(path_results: str, dataset_name: str, file_name :str) -> str:
    """Compose path to fitting durations result file.

    Args:
        path_results (str): path to results folder.
        dataset_name (str): name of the dataset.
        file_name (str): name of the file.

    Returns:
        str: path to fitting durations result file.
    """

    return compose_path_result_file(path_results, dataset_name, "fitting_duration", file_name)

def compose_path_result_ranking_relevances(path_results: str, dataset_name: str, file_name :str) -> str:
    """Compose path to ranking relevances result file.

    Args:
        path_results (str): path to results folder.
        dataset_name (str): name of the dataset.
        file_name (str): name of the file.

    Returns:
        str: path to ranking relevances result file.
    """

    return compose_path_result_file(path_results, dataset_name, "ranking_relevance", file_name)

def compose_path_result_converging_paces(path_results: str, dataset_name: str, file_name :str) -> str:
    """Compose path to converging paces result file.

    Args:
        path_results (str): path to results folder.
        dataset_name (str): name of the dataset.
        file_name (str): name of the file.

    Returns:
        str: path to converging paces result file.
    """

    return compose_path_result_file(path_results, dataset_name, "converging_pace", file_name)
