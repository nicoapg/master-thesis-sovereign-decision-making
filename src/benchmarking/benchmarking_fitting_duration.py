import time
import itertools

import tqdm
import pandas as pd

from src.data import MasterDataset
from src.ranking import MasterRanker



def assess_fitting_durations(dataset: MasterDataset, 
    ranker_space: list[str], 
    n_runs: int, n_runs_warmup: int=0, 
    offline:bool=True, online: bool=True, l_batches: int=1, 
    show_progress: bool=True, 
    ) -> list[dict]:
    """Assess fitting durations of rankers on a dataset.
    
    Args:
        dataset (MasterDataset): dataset to assess.
        ranker_space (list[str]): names of rankers to assess.
        n_runs (int): number of runs.
        n_runs_warmup (int, optional): number of warmup runs. Defaults to 0.
        offline (bool, optional): whether to assess offline learning. Defaults to True.
        online (bool, optional): whether to assess online learning. Defaults to True.
        l_batches (int, optional): length of batches for online learning. Defaults to 1.
        show_progress (bool, optional): whether to show progress. Defaults to True.
    
    Returns:
        list[dict]: assessed fitting durations.
    """

    # Initialize fitting durations
    fitting_durations = []

    # Initialize progress
    progress = tqdm.tqdm(total=n_runs+n_runs_warmup, desc="Assessing fitting durations", 
        leave=False, disable=not show_progress, 
    )

    # Iterate over runs
    for run in range(-n_runs_warmup, n_runs):
        dataset.shuffle()

        # Iterate over rankers
        for ranker_name in ranker_space:
            
            # offline learning
            if offline:

                # Fit ranker
                time_start = time.time()
                ranker = MasterRanker(ranker_name)
                ranker.add_principles(dataset.principles)
                ranker.add_comparisons(dataset.comparisons)
                time_end = time.time()

                # Store fitting duration
                if run >= 0:
                    fitting_durations.append({
                        "dataset": dataset.name, **dataset.params, 
                        "ranker": ranker.name, 
                        "learning": "offline", 
                        "run": run, 
                        "fitting_duration": time_end-time_start, 
                    })

            # online learning
            if online:

                # Fit ranker
                time_start = time.time()
                ranker = MasterRanker(ranker_name)
                ranker.add_principles(dataset.principles)
                for comparions_batch in itertools.batched(dataset.comparisons, l_batches):
                    ranker.add_comparisons(comparions_batch)
                time_end = time.time()

                # Store fitting duration
                if run >= 0:
                    fitting_durations.append({
                        "dataset": dataset.name, **dataset.params, 
                        "ranker": ranker.name, 
                        "learning": "online", 
                        "run": run, 
                        "fitting_duration": time_end-time_start, 
                    })
        
        # Update progress
        progress.update()

    return fitting_durations


def format_fitting_durations(fitting_durations: list[dict], dataset_params_columns: list[str]) -> pd.DataFrame:
    """Format fitting durations.

    Args:
        fitting_durations (list[dict]): fitting durations.
        dataset_params_columns (list[str]): columns of dataset parameters.
        
    Returns:
        pd.DataFrame: formatted fitting durations.
    """

    df_fitting_durations = (
        pd.DataFrame(fitting_durations)
        
        .set_index(["dataset", *dataset_params_columns, "ranker", "learning", "run"])
        .sort_index()
    )
    return df_fitting_durations
