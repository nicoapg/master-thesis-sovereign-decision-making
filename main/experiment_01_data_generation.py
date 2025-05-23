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
from experiment_utils import (
    generate_experiment_data,
    create_ranking_from_results,
    create_swarm_plot_kendalls_tau_vs_number_of_votes,
    create_ridge_plot,
    print_ranker_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DATA_DIR / 'experiment_analysis.log'),
        logging.StreamHandler()
    ]
)

def create_analysis_folder(experiment_file: str) -> Path:
    """
    Create a timestamped folder for the analysis results.
    
    Args:
        experiment_file: Name of the experiment file being analyzed
        
    Returns:
        Path to the created folder
    """
    # Get current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    # Create folder name
    folder_name = f"{timestamp}-{Path(experiment_file).stem}"
    analysis_folder = DATA_DIR / folder_name
    
    # Create the folder
    analysis_folder.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created analysis folder: {analysis_folder}")
    
    return analysis_folder

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

def generate_kendall_tau_versus_vote_number(shuffle_results: bool = False, N_repeats: int = 5):
    """
    Generate Kendall's Tau correlation values for different numbers of votes using various ranking algorithms.
    
    Args:
        shuffle_results: If True, randomly shuffle the results DataFrame before analysis
        N_repeats: Number of times to repeat the analysis for each vote count (default: 5)
    """
    # Define file paths
    experiment_file = DATA_DIR / "experiment_01.csv"
    principles_file = DATA_DIR / "principles_01.csv"
    results_file = DATA_DIR / "results_01.csv"

    # Read the files
    experiment_df = pd.read_csv(experiment_file)
    principles_df = pd.read_csv(principles_file)
    results_df = pd.read_csv(results_file)
    
    # Initial shuffle if requested
    if shuffle_results:
        results_df = results_df.sample(frac=1).reset_index(drop=True)
        logging.info("Results DataFrame has been shuffled")

    logging.info("=== Starting Kendall's Tau Analysis ===")
    logging.info(f"Number of repeats: {N_repeats}")

    # Create rankings for each experiment using each ranker with increasing numbers of votes
    experiment_id = 1  # We know this is the experiment ID from our data generation
    
    # Define the vote increments
    vote_increments = list(range(50, 1001, 50))  # 100, 1100, 2100, ..., 9100
    
    # Define the ideal ranking (1 to 30)
    ideal_ranking = list(range(1, 31))
    
    # Store all results for plotting
    all_results = []
    
    # Analyze each ranking algorithm separately
    for ranker_type in RankingAlgorithms:
        logging.info(f"\n{'='*80} \nAnalysis for {ranker_type.value} ranker:\n{'='*80}")
        
        # Store results for this ranker
        tau_values = []
        vote_counts = []
        
        for n_votes in vote_increments:
            # Take only the first n_votes from the results
            subset_results = results_df.head(n_votes)
            
            # Run the ranker N times
            for repeat in range(N_repeats):
                # Shuffle the subset for each repeat
                if shuffle_results:
                    subset_results = subset_results.sample(frac=1).reset_index(drop=True)
                
                ranked_ids = create_ranking_from_results(subset_results, experiment_id, ranker_type)
                
                # Calculate Kendall's Tau
                tau, p_value = stats.kendalltau(ranked_ids, ideal_ranking)
                
                # Store result for plotting
                all_results.append({
                    'Votes': n_votes,
                    'Kendall\'s Tau': tau,
                    'Ranker': ranker_type.value,
                    'Repeat': repeat + 1
                })
            
            # Calculate mean tau for this vote count
            mean_tau = np.mean([r['Kendall\'s Tau'] for r in all_results 
                              if r['Votes'] == n_votes and r['Ranker'] == ranker_type.value])
            tau_values.append(mean_tau)
            vote_counts.append(n_votes)
            
            # Log the mean tau value for this vote count
            logging.info(f"Votes: {n_votes:5d} | Mean Kendall's Tau: {mean_tau:.3f}")
        
        # Log summary for this ranker
        logging.info(f"\nSummary for {ranker_type.value} ranker:")
        logging.info("Votes | Mean Kendall's Tau")
        logging.info("-" * 25)
        for votes, tau in zip(vote_counts, tau_values):
            logging.info(f"{votes:5d} | {tau:.3f}")
        logging.info(f"{'='*80}\n")
    
    # Log the total number of results
    logging.info(f"Total number of results: {len(all_results)}")
    logging.info("=== Kendall's Tau Analysis Complete ===")
    
    return all_results, N_repeats

# Set the mode: "data_generation" or "data_analysis"
MODE = "data_analysis"
N_REPEATS = 10  # Number of repeats for each vote count - kendall tau analysis per ranker for a given number of votes
CORRECTNESS_STUDIED = [0, 0.50, 0.99, 1.00]  # Different correctness values to study

def get_experiment_files() -> list:
    """
    Get all experiment files with 'correctness' in their name.
    
    Returns:x
        List of experiment file names
    """
    experiment_files = []
    for file in DATA_DIR.glob("experiment_01_correctness*.csv"):
        experiment_files.append(file.name)
    return sorted(experiment_files)

def extract_correctness_from_filename(filename: str) -> float:
    """
    Extract the correctness value from the experiment filename.
    
    Args:
        filename: Name of the experiment file (e.g., 'experiment_01_correctness099.csv')
        
    Returns:
        Correctness value as a float (e.g., 0.99)
    """
    # Extract the correctness part (e.g., '099' from 'correctness099')
    correctness_str = filename.split('correctness')[1].split('.')[0]
    # Convert to float (e.g., '099' -> 0.99)
    return float(f"0.{correctness_str}")

if __name__ == "__main__":
    if MODE == "data_generation":
        # Generate datasets for each correctness value
        for correctness in CORRECTNESS_STUDIED:
            logging.info(f"\nGenerating dataset with correctness = {correctness}")
            generate_experiment_data(correctness=correctness)
            logging.info(f"Dataset generation complete for correctness = {correctness}")
    elif MODE == "data_analysis":
        # Get all experiment files to analyze
        experiment_files = get_experiment_files()
        logging.info(f"Found {len(experiment_files)} experiment files to analyze")
        
        # Analyze each experiment file
        for experiment_file in experiment_files:
            logging.info(f"\n{'='*80}")
            logging.info(f"Starting analysis for {experiment_file}")
            logging.info(f"{'='*80}")
            
            # Extract correctness from filename
            correctness = extract_correctness_from_filename(experiment_file)
            logging.info(f"Extracted correctness value: {correctness}")
            
            # Create analysis folder
            analysis_folder = create_analysis_folder(experiment_file)
            
            # Run the analysis
            all_results, N_repeats = generate_kendall_tau_versus_vote_number(
                shuffle_results=True, 
                N_repeats=N_REPEATS
            )
            
            # Create and save the analysis plots in the new folder
            create_swarm_plot_kendalls_tau_vs_number_of_votes(
                all_results, 
                N_repeats, 
                output_dir=analysis_folder,
                correctness=correctness
            )
            create_ridge_plot(
                all_results, 
                N_repeats, 
                output_dir=analysis_folder,
                correctness=correctness
            )
            
            logging.info(f"Analysis complete for {experiment_file}")
            logging.info(f"Results saved in: {analysis_folder}")
            logging.info(f"{'='*80}\n")
    else:
        raise ValueError(f"Invalid MODE: {MODE}. Must be either 'data_generation' or 'data_analysis'")
