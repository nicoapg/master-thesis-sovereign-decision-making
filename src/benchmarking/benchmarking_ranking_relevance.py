import tqdm
import scipy
import pandas as pd

from src.data import MasterDataset
from src.ranking import MasterRanker
from .benchmarking_utils import append_best_rank



def assess_ranking_relevances(dataset: MasterDataset, 
    ranker_space: list[str], 
    n_runs: int, 
    show_progress: bool=True, 
    ) -> list[dict]:
    """Assess ranking relevances of rankers on a dataset.

    Args:
        dataset (MasterDataset): dataset.
        ranker_space (list[str]): names of rankers to assess.
        n_runs (int): number of runs.
        show_progress (bool, optional): show progress. Defaults to True.
    """

    # Initialize ranking_relevances
    ranking_relevances = []

    # Initialize progress
    progress = tqdm.tqdm(total=n_runs, desc="Assessing ranking relevances", 
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
            ranker.add_comparisons(dataset.comparisons)

            # Store ranking relevances
            ranking_relevances.extend([
                {
                    "dataset": dataset.name, **dataset.params, 
                    "ranker": ranker.name, 
                    "principle": principle, 
                    "run": run, 
                    "rank": principle_infos["rank"], 
                    **({"rank_real": dataset.real_ranking[principle]} if dataset.real_ranking is not None else {}), 
                }
                for principle, principle_infos in ranker.get_result().items()
            ])

        # Update progress
        progress.update()

    return ranking_relevances


def format_ranking_relevances(ranking_relevances: list[dict], 
    dataset_params_columns: list[str]) -> pd.DataFrame:
    """Format ranking relevances.
    
    Args:
        ranking_relevances (list[dict]): ranking relevances.
        dataset_params_columns (list[str]): dataset parameters columns.
    
    Returns:
        pd.DataFrame: formatted ranking relevances.
    """

    df_ranking_relevances = (
        pd.DataFrame(ranking_relevances)
        
        .set_index(["dataset", *dataset_params_columns, "ranker", "principle", "run"])
        .sort_index()
    )
    return df_ranking_relevances

def enrich_ranking_relevances(df_ranking_relevances: pd.DataFrame, 
    dataset_params_columns: list[str], 
    ) -> pd.DataFrame:
    """Enrich ranking relevances.

    Args:
        df_ranking_relevances (pd.DataFrame): ranking relevances.
        dataset_params_columns (list[str]): dataset parameters columns.

    Returns:
        pd.DataFrame: enriched ranking relevances.
    """

    df_ranking_relevances = (
        df_ranking_relevances

        .pipe(append_best_rank, 
            last_rank_column="rank", 
            dataset_params_columns=dataset_params_columns, 
        )
        
        .melt(id_vars="rank", var_name="reference", value_name="rank_reference", ignore_index=False)
        .set_index("reference", append=True)
        .sort_index()
    )
    return df_ranking_relevances


def compute_ranking_relevances(df_ranking_relevances: pd.DataFrame, 
    dataset_params_columns: list[str]) -> pd.DataFrame:
    """Compute ranking relevances.

    Args:
        df_ranking_relevances (pd.DataFrame): ranking relevances.
        dataset_params_columns (list[str]): dataset parameters columns.

    Returns:
        pd.DataFrame: ranking relevances.
    """

    df_ranking_relevances = (
        df_ranking_relevances
        
        .groupby(["dataset", *dataset_params_columns, "ranker", "run", "reference"])
        .apply(lambda df: scipy.stats.kendalltau(df["rank"], df["rank_reference"]).correlation, include_groups=False)
        .to_frame("ranking_relevance")
    )
    return df_ranking_relevances

def compute_ranking_relevance_matrixes(df_ranking_relevances: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Compute ranking relevance matrixes.

    Args:
        df_ranking_relevances (pd.DataFrame): ranking relevances.
        reference (str): reference.

    Returns:
        pd.DataFrame: ranking relevance matrixes.
    """

    df_ranking_relevances_matrixes = (
        df_ranking_relevances

        .xs(reference, level="reference")

        .groupby(["ranker", "principle", "rank_reference"])
        ["rank"]
        .value_counts(normalize=True)

        .unstack("rank")

        .sort_values(["ranker", "rank_reference"])
        .droplevel("rank_reference")
    )
    return df_ranking_relevances_matrixes
