import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from utils import DATA_DIR
from rankers import UserChoice, RankingAlgorithms, WinRateRanker, EloRanker, TrueSkillRanker, EigenvectorCentralityRanker, BradleyTerryRanker
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


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
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # VERY IMPORTANT: Determine which ID is smaller (the "right" answer)
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
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
    
    # If no votes, return a default ranking (1 to 30)
    if len(experiment_votes) == 0:
        return list(range(1, 31))
    
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
    
    # Ensure we have all 30 principles in the ranking
    all_principles = set(range(1, 31))
    ranked_set = set(ranked_ids)
    missing_principles = all_principles - ranked_set
    
    # Add missing principles at the end of the ranking
    if missing_principles:
        ranked_ids.extend(sorted(missing_principles))
    
    return ranked_ids

def generate_experiment_data(correctness: float = 0.99):
    """
    Generate and save the experiment, principles, and results data.
    
    Args:
        correctness: Probability of choosing the correct answer (default: 0.99)
    """
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

    # Generate choices with specified correctness
    print(f"Generating {n_results} choices with correctness={correctness} DEBUG!!")
    choices = [make_choice(lhs_ids[i], rhs_ids[i], correctness=correctness) for i in range(n_results)]

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

    # Format correctness value for filename (e.g., 0.99 -> 099)
    correctness_str = f"correctness{int(correctness * 100):03d}"

    # Save to CSV files with correctness in filename
    experiment_output_file = DATA_DIR / f"experiment_01_{correctness_str}.csv"
    principles_output_file = DATA_DIR / f"principles_01_{correctness_str}.csv"
    results_output_file = DATA_DIR / f"results_01_{correctness_str}.csv"

    print(results_df.head(10))


    experiment_df.to_csv(experiment_output_file, index=False)
    principles_df.to_csv(principles_output_file, index=False)
    results_df.to_csv(results_output_file, index=False)

    return experiment_df, principles_df, results_df

def create_swarm_plot_kendalls_tau_vs_number_of_votes(all_results, N_repeats, output_dir: Path = None, correctness: float = None):
    """Create and save a swarm plot of Kendall's Tau values vs number of votes."""
    plt.figure(figsize=(12, 6))
    df = pd.DataFrame(all_results)
    
    # Create swarmplot
    sns.swarmplot(data=df, x='Votes', y='Kendall\'s Tau', hue='Ranker', palette='Set2')
    
    # Customize the plot
    title = f'Kendall\'s Tau Correlation vs Number of Votes by Ranker (N={N_repeats} repeats)'
    if correctness is not None:
        title += f' (Correctness={correctness:.2f})'
    plt.title(title)
    plt.xlabel('Number of Votes')
    plt.ylabel('Kendall\'s Tau')
    plt.xticks(rotation=45)
    plt.legend(title='Ranking Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set axis limits
    plt.ylim(0.0, 1.1)
    
    # Add horizontal lines
    #plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    # Add tickers at y=0 and y=1
    plt.yticks(list(plt.yticks()[0]) + [0, 1])
    
    plt.tight_layout()
    
    # Save the plot
    if output_dir is None:
        output_dir = DATA_DIR
    filename = "kendall_tau_analysis"
    if correctness is not None:
        filename += f"_correctness{int(correctness * 100):03d}"
    plot_file = output_dir / f"{filename}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    
    # Show the plot
    #plt.show()

def create_ridge_plot(all_results, N_repeats, output_dir: Path = None, correctness: float = None):
    """Create and save a ridge plot of Kendall's Tau distributions."""
    # Set the theme
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    # Create DataFrame from results
    df = pd.DataFrame(all_results)
    print('I am here inside create_ridge_plot')
    print(df.head(2))
    
    # Create a separate plot for each ranker
    for ranker in df['Ranker'].unique():
        # Filter data for this ranker
        ranker_df = df[df['Ranker'] == ranker]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create the ridge plot
        pal = sns.cubehelix_palette(1, rot=-.25, light=.7)
        g = sns.FacetGrid(ranker_df, row="Votes", hue="Votes", aspect=15, height=.5, palette=pal)
        
        # Function to check if all values are the same
        def is_constant(x):
            return len(set(x)) == 1
        
        # For each vote count, check if we need to plot a line or a density
        for i, votes in enumerate(ranker_df['Votes'].unique()):
            vote_data = ranker_df[ranker_df['Votes'] == votes]['Kendall\'s Tau']
            
            if is_constant(vote_data):
                # If all values are the same, plot a vertical line in the corresponding subplot
                constant_value = vote_data.iloc[0]
                g.axes[i, 0].axvline(x=constant_value, color=pal[0], linewidth=2)
            else:
                # If values vary, plot the density
                g.map(sns.kdeplot, "Kendall's Tau",
                      bw_adjust=.5, clip_on=False,
                      fill=True, alpha=1, linewidth=1.5)
                g.map(sns.kdeplot, "Kendall's Tau", 
                      clip_on=False, color="w", lw=2, bw_adjust=.5)
        
        # Add reference line
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
        
        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, f"Votes: {label}", fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)
        
        g.map(label, "Kendall's Tau")
        
        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=-.25)
        
        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        
        # Add a title
        title = f'Distribution of Kendall\'s Tau for {ranker} (N={N_repeats} repeats)'
        if correctness is not None:
            title += f' (Correctness={correctness:.2f})'
        plt.suptitle(title, y=0.98)
        
        # Save the plot
        if output_dir is None:
            output_dir = DATA_DIR
        filename = f"ridge_plot_kendall_tau_analysis_{ranker.lower().replace(' ', '_')}"
        if correctness is not None:
            filename += f"_correctness{int(correctness * 100):03d}"
        plot_file = output_dir / f"{filename}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nRidge plot for {ranker} saved to: {plot_file}")
        
        # Show the plot
        #plt.show()

def print_ranker_summary(ranker_type, vote_counts, tau_values):
    """Print summary statistics for a ranker."""
    print(f"\nSummary for {ranker_type.value} ranker:")
    print("Votes | Mean Kendall's Tau")
    print("-" * 25)
    for votes, tau in zip(vote_counts, tau_values):
        print(f"{votes:5d} | {tau:.3f}")
    print(f"{'='*80}\n") 