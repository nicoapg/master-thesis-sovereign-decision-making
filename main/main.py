import logging
import scipy
import numpy as np
from rankers import BaseRanker, WinRateRanker, UserChoice, RankingAlgorithms, EloRanker, TrueSkillRanker, EigenvectorCentralityRanker, BradleyTerryRanker
import numpy as np
from utils import read_sgmlp_csv
import os
from pathlib import Path

# Get the root directory of the project
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data" / "20250505-fiona-db-exports"

# Configure logging
logging.basicConfig(level=logging.INFO)

def create_ranking_from_results(results_df, experiment_id, ranker_type: RankingAlgorithms) -> list:
    """
    Creates a ranking using the specified ranker based on the results data.
    
    Args:
        results_df: DataFrame containing the results data (this is the table with individual votes)
        experiment_id: ID of the experiment to analyze
        ranker_type: Type of ranker to use (from RankingAlgorithms enum)
        
    Returns:
        List of principle IDs sorted by rank (highest to lowest)
    """
    # Filter results for the specific experiment
    experiment_votes = results_df[results_df['experiment_id'] == experiment_id]
    print(f"Experiment total votes: {experiment_votes.shape}")
    
    # Convert results to the format expected by rankers
    votes = []
    for _, row in experiment_votes.iterrows():
        # Skip votes with no choice
        if row['choice'] == UserChoice.NO_CHOICE:
            continue
            
        vote = {
            "lhs_id": row['lhs_id'],
            "rhs_id": row['rhs_id'],
            "choice": row['choice']  # Using the enum values directly
        }
        votes.append(vote)
    
    print(f"Valid votes for ranking: {len(votes)}")
    
    # Create and use the appropriate ranker
    if ranker_type == RankingAlgorithms.WIN_RATE:
        ranker = WinRateRanker()
    elif ranker_type == RankingAlgorithms.ELO:
        ranker = EloRanker()
    elif ranker_type == RankingAlgorithms.TRUE_SKILL:
        ranker = TrueSkillRanker()
    elif ranker_type == RankingAlgorithms.EIGEN:
        ranker = EigenvectorCentralityRanker()
    elif ranker_type == RankingAlgorithms.BRADLEY_TERRY:
        ranker = BradleyTerryRanker()
    else:
        raise ValueError(f"Unknown ranker type: {ranker_type}")
    
    ranked_ids = ranker.compute(votes)
    return ranked_ids

# Example usage:
if __name__ == "__main__":
    print("----------------------------------------\n          NEW SOFTWARE RUN\n----------------------------------------")

    # Read experiments file
    experiments_file = DATA_DIR / "20250505_experiments.csv"
    results_file = DATA_DIR / "20250505_results.csv"
    principles_file = DATA_DIR / "20250505_principles.csv"

    # Dictionary of experiments to analyze
    experiments_to_analyze = {
        14: "DC-01-AMLD Question 1",
        #15: "DC-01-AMLD Question 2",
        #16: "DC-01-AMLD Question 3"
    }

    # Read experiments file, results file, and principles file.
    experiments_df, experiments_columns = read_sgmlp_csv(experiments_file)
    results_df, results_columns = read_sgmlp_csv(results_file)
    principles_df, principles_columns = read_sgmlp_csv(principles_file)

    print(f"\nThis is how the EXPERIMENTS DATAFRAME looks like (subsample of columns):\n\n {experiments_df[['id', 'name', 'description', 'principle_ids']].head(1).T}")
    print(f"\nThis is how the RESULTS DATAFRAME looks like (subsample of columns): \n{results_df[['id', 'experiment_id', 'lhs_id', 'rhs_id', 'choice']].head(1).T}")
    print(f"\nThis is how the PRINCIPLES DATAFRAME looks like (subsample of columns): {principles_df[['id', 'description']].head(1).T}")
        
    # Create rankings for each experiment using each ranker
    for experiment_id, experiment_name in experiments_to_analyze.items():
        print(f"\n{'='*80}")
        print(f"Creating rankings for experiment {experiment_id}: {experiment_name}")
        print(f"{'='*80}")
        
        for ranker_type in RankingAlgorithms:
            print(f"\n{'-'*40}")
            print(f"Using {ranker_type.value} ranker:")
            print(f"{'-'*40}")
            
            ranked_ids = create_ranking_from_results(results_df, experiment_id, ranker_type)
            
            print("\nRanked principle IDs (highest to lowest):")
            print(ranked_ids)

            # Print the actual principles in ranked order
            print("\nRANKED PRINCIPLES:")
            for rank, principle_id in enumerate(ranked_ids, 1):
                # This is the row from the dataframe
                principle = principles_df[principles_df['id'] == principle_id].iloc[0]
                print(f"{rank}. {principle['description']}")
            print(f"\n{'-'*40}\n")
        print(f"{'='*80}\n")

    
