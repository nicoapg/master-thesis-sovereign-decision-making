import itertools

import tqdm
import scipy
import pandas as pd

from src.data import MasterDataset
from src.ranking import MasterRanker
from .benchmarking_utils import append_last_rank, append_best_rank



def assess_converging_paces(
    dataset: MasterDataset, 
    ranker_space: list[str], 
    n_runs: int, 
    l_batches: int=1, 
    show_progress: bool=True, 
    ) -> list[dict]:
    """Assess converging paces of rankers on a dataset.
    
    Args:
        dataset (MasterDataset): dataset to assess.
        ranker_space (list[str]): names of rankers to assess.
        n_runs (int): number of runs.
        l_batches (int, optional): length of batches for online learning. Defaults to 1.
        show_progress (bool, optional): whether to show progress. Defaults to True.
        
    Returns:
        list[dict]: assessed converging paces.
    """

    # Initialize converging paces
    converging_paces = []

    # Initialize progress
    progress = tqdm.tqdm(total=n_runs, desc="Assessing converging paces", 
        leave=False, disable=not show_progress, 
    )

    # Iterate over runs
    for run in range(n_runs):
        dataset.shuffle()

        # Iterate over rankers
        for ranker_name in ranker_space:
        
            # Fit ranker
            ranker = MasterRanker(ranker_name)
            ranker.add_principles(dataset.principles)
            n_votes = 0
            for comparions_batch in itertools.batched(dataset.comparisons, l_batches):
                ranker.add_comparisons(comparions_batch)
                n_votes += len(comparions_batch)

                # Store converging pace
                converging_paces.extend([
                    {
                        "dataset": dataset.name, **dataset.params, 
                        "ranker": ranker.name, 
                        "principle": principle, 
                        "n_votes": n_votes, 
                        "run": run, 
                        "rank": principle_infos["rank"], 
                        **({"rank_real": dataset.real_ranking[principle]} if dataset.real_ranking is not None else {}), 
                    }
                    for principle, principle_infos in ranker.get_result().items()
                ])

        # Update progress
        progress.update()

    return converging_paces


def format_converging_paces(converging_paces: list[dict], 
    dataset_params_columns: list[str]) -> pd.DataFrame:
    """Format converging paces.

    Args:
        converging_paces (list[dict]): converging paces.
        dataset_params_columns (list[str]): dataset parameters columns.
    
    Returns:
        pd.DataFrame: formatted converging paces.
    """

    df_converging_paces = (
        pd.DataFrame(converging_paces)
        
        .set_index(["dataset", *dataset_params_columns, "ranker", "principle", "n_votes", "run"])
        .sort_index()
    )
    return df_converging_paces

def enrich_converging_paces(df_converging_paces: pd.DataFrame
    ) -> pd.DataFrame:
    """Enrich converging paces.

    Args:
        df_converging_paces (pd.DataFrame): converging paces.

    Returns:
        pd.DataFrame: enriched converging paces.
    """

    df_converging_paces = (
        df_converging_paces
        
        .pipe(append_last_rank, rank_column="rank")
        .pipe(append_best_rank, last_rank_column="rank_last")

        .melt(id_vars="rank", var_name="reference", value_name="rank_reference", ignore_index=False)
        .set_index("reference", append=True)
        .sort_index()
    )
    return df_converging_paces


def compute_converging_paces(df_converging_paces: pd.DataFrame, 
    dataset_params_columns: list[str]) -> pd.DataFrame:
    """Compute converging paces.

    Args:
        df_converging_paces (pd.DataFrame): converging paces.
        dataset_params_columns (list[str]): dataset parameters columns.

    Returns:
        pd.DataFrame: computed converging paces.
    """

    df_converging_paces = (
        df_converging_paces
        
        .groupby(["dataset", *dataset_params_columns, "ranker", "n_votes", "reference", "run"])
        .apply(lambda df: scipy.stats.kendalltau(df["rank"], df["rank_reference"]).correlation, include_groups=False)
        .to_frame("converging_pace")
    )
    return df_converging_paces
