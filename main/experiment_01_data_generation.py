import pandas as pd
from datetime import datetime
import os
from pathlib import Path
from utils import DATA_DIR, ROOT_DIR
import numpy as np
from rankers import UserChoice, RankingAlgorithms, WinRateRanker, EloRanker, TrueSkillRanker, EigenvectorCentralityRanker, BradleyTerryRanker
from scipy import stats
import logging
import seaborn as sns
import matplotlib.pyplot as plt

# Configure logging to suppress ranker warnings
logging.basicConfig(level=logging.ERROR)  # Only show ERROR level and above

# Set the mode: "data_generation" or "data_analysis"
MODE = "data_analysis"

def make_choice(lhs_id: int, rhs_id: int, correctness: float = 0.8) -> int:
    """
    Generate a choice between two principles based on their IDs and a correctness probability.
    
    Args:
        lhs_id: ID of the left-hand side principle
        rhs_id: ID of the right-hand side principle
        correctness: Probability of choosing the correct answer (default: 0.8)
        
    Returns:
        A value from UserChoice enum (0, 1, 2, or 3)
    """
    # Determine which ID is smaller (the "right" answer)
    right_answer = UserChoice.RHS if rhs_id < lhs_id else UserChoice.LHS
    
    # First, decide if we're going to pick the right or wrong answer
    pick_right = np.random.random() < correctness
    
    # If we're picking the right answer
    if pick_right:
        # 95% chance of picking the right answer, 5% chance of tie
        if np.random.random() < 0.95:
            return right_answer
        else:
            return UserChoice.TIE
    # If we're picking the wrong answer
    else:
        # 95% chance of picking the wrong answer, 5% chance of tie
        if np.random.random() < 0.95:
            return UserChoice.LHS if right_answer == UserChoice.RHS else UserChoice.RHS
        else:
            return UserChoice.TIE

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
    # Filter results for the specific experiment and remove NO_CHOICE votes
    experiment_votes = results_df[
        (results_df['experiment_id'] == experiment_id) & 
        (results_df['choice'] != UserChoice.NO_CHOICE)
    ]
    
    # Convert results to the format expected by rankers
    votes = []
    for _, row in experiment_votes.iterrows():
        vote = {
            "lhs_id": row['lhs_id'],
            "rhs_id": row['rhs_id'],
            "choice": row['choice']  # Using the enum values directly
        }
        votes.append(vote)
    
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

def generate_data():
    """Generate and save the experiment, principles, and results data."""
    # Create the experiment data
    experiment_data = {
        'id': [1],
        'author': ['abc'],
        'status': ['active'],
        'name': ['experiment_01'],
        'description': ['this is an experiment for convergence of the algorithms'],
        'is_probe': [True],
        'insertion_date': [datetime.now().strftime('%Y-%m-%d')],
        'update_date': [datetime.now().strftime('%Y-%m-%d')],
        'principle_ids': [f"{{{','.join(str(i) for i in range(1, 31))}}}"],
        'number_of_principles': [30],
        'participant_ids': ['{}'],
        'secret': ['abcde'],
        'number_of_pairs': [5],
        'is_public': [False]
    }

    # Create the principles data
    principles_data = {
        'id': list(range(1, 31)),
        'source': ['test-principles'] * 30,
        'description': [f'test-description-{i}' for i in range(1, 31)],
        'tags': ['{}'] * 30,
        'frequency': ['normal'] * 30,
        'insertion_date': [datetime.now().strftime('%Y-%m-%d')] * 30,
        'title': [f'Principle {i}' for i in range(1, 31)]
    }

    # Create the results data
    n_results = 10000
    today = datetime.now().strftime('%Y-%m-%d')

    # Generate random lhs_id and rhs_id pairs
    lhs_ids = np.random.randint(1, 31, size=n_results)
    rhs_ids = np.random.randint(1, 31, size=n_results)

    # Ensure lhs_id and rhs_id are different
    for i in range(n_results):
        while rhs_ids[i] == lhs_ids[i]:
            rhs_ids[i] = np.random.randint(1, 31)

    # Generate choices with 99% correctness
    choices = [make_choice(lhs_ids[i], rhs_ids[i], correctness=0.99) for i in range(n_results)]

    results_data = {
        'id': list(range(1, n_results + 1)),
        'experiment_id': [1] * n_results,
        'lhs_id': lhs_ids,
        'rhs_id': rhs_ids,
        'user_id': ['abc'] * n_results,
        'user_groups': ['{default}'] * n_results,
        'choice': choices,
        'insertion_date': [today] * n_results,
        'update_date': [today] * n_results,
        'processed': ['t'] * n_results
    }

    # Create DataFrames
    experiment_df = pd.DataFrame(experiment_data)
    principles_df = pd.DataFrame(principles_data)
    results_df = pd.DataFrame(results_data)

    # Save to CSV files
    experiment_output_file = DATA_DIR / "experiment_01.csv"
    principles_output_file = DATA_DIR / "principles_01.csv"
    results_output_file = DATA_DIR / "results_01.csv"

    experiment_df.to_csv(experiment_output_file, index=False)
    principles_df.to_csv(principles_output_file, index=False)
    results_df.to_csv(results_output_file, index=False)

    print(f"Experiment data saved to: {experiment_output_file}")
    print("\nExperiment DataFrame contents:")
    print(experiment_df.T)

    print(f"\nPrinciples data saved to: {principles_output_file}")
    print("\nPrinciples DataFrame contents (first 5 rows):")
    print(principles_df.head().T)

    print(f"\nResults data saved to: {results_output_file}")
    print("\nResults DataFrame contents (first 5 rows):")
    print(results_df.head().T)

    # Print some statistics about the results
    print("\nResults statistics:")
    print(f"Total number of results: {len(results_df)}")
    print("\nChoice distribution:")
    print(results_df['choice'].value_counts(normalize=True).sort_index())

def analyze_data():
    """Read the data files and generate rankings using different algorithms with increasing numbers of votes."""
    # Define file paths
    experiment_file = DATA_DIR / "experiment_01.csv"
    principles_file = DATA_DIR / "principles_01.csv"
    results_file = DATA_DIR / "results_01.csv"

    # Read the files
    experiment_df = pd.read_csv(experiment_file)
    principles_df = pd.read_csv(principles_file)
    results_df = pd.read_csv(results_file)

    print("\n=== Data Analysis Mode ===")

    # Create rankings for each experiment using each ranker with increasing numbers of votes
    experiment_id = 1  # We know this is the experiment ID from our data generation
    
    # Define the vote increments
    vote_increments = list(range(100, 10001, 1000))  # 100, 1100, 2100, ..., 9100
    
    # Define the ideal ranking (1 to 30)
    ideal_ranking = list(range(1, 31))
    
    # Store all results for plotting
    all_results = []
    
    # Analyze each ranking algorithm separately
    for ranker_type in RankingAlgorithms:
        print(f"\n{'='*80}")
        print(f"Analysis for {ranker_type.value} ranker:")
        print(f"{'='*80}")
        
        # Store results for this ranker
        tau_values = []
        vote_counts = []
        
        for n_votes in vote_increments:
            #print(f"\n{'-'*40}")
            #print(f"Using {n_votes} votes:")
            #print(f"{'-'*40}")
            
            # Take only the first n_votes from the results
            subset_results = results_df.head(n_votes)
            
            ranked_ids = create_ranking_from_results(subset_results, experiment_id, ranker_type)
            
            # Calculate Kendall's Tau
            tau, p_value = stats.kendalltau(ranked_ids, ideal_ranking)
            tau_values.append(tau)
            vote_counts.append(n_votes)
            
            #print("\nRanked principle IDs (winner at the top):")
            #print(ranked_ids)
            #print(f"\nKendall's Tau correlation with ideal ranking: {tau:.3f} (p-value: {p_value:.3e})")
            #print(f"\n{'-'*40}\n")
            # Store result for plotting
            all_results.append({
                'Votes': n_votes,
                'Kendall\'s Tau': tau,
                'Ranker': ranker_type.value
            })
        
        # Print summary for this ranker
        print(f"\nSummary for {ranker_type.value} ranker:")
        print("Votes | Kendall's Tau")
        print("-" * 20)
        for votes, tau in zip(vote_counts, tau_values):
            print(f"{votes:5d} | {tau:.3f}")
        print(f"{'='*80}\n")
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    df = pd.DataFrame(all_results)
    
    # Create swarmplot
    sns.swarmplot(data=df, x='Votes', y='Kendall\'s Tau', hue='Ranker', palette='Set2')
    
    # Customize the plot
    plt.title('Kendall\'s Tau Correlation vs Number of Votes by Ranker')
    plt.xlabel('Number of Votes')
    plt.ylabel('Kendall\'s Tau')
    plt.xticks(rotation=45)
    plt.legend(title='Ranking Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    plot_file = DATA_DIR / "kendall_tau_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    if MODE == "data_generation":
        generate_data()
    elif MODE == "data_analysis":
        analyze_data()
    else:
        raise ValueError(f"Invalid MODE: {MODE}. Must be either 'data_generation' or 'data_analysis'")
