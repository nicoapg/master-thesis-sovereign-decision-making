import logging
import argparse

from src import config
from src.data import MasterDataset
from src.benchmarking import benchmarking_fitting_duration, benchmarking_ranking_relevance, benchmarking_converging_paces
from src.visualizing import visualizing_benchmark
from src.utils import utils_paths, utils_figs



def setup_argparser() -> argparse.ArgumentParser:
    """Setup the argparser for the main processing script."""

    # Initialize argparser
    argparser = argparse.ArgumentParser(
        prog="Main benchmarking", 
        description="Script to benchmark rankers on a dataset.", 
    )

    # Add dataset argument(s)
    argparser.add_argument(
        "--dataset_name", 
        type=str, 
        required=True, 
        choices=config.DATASET_SPACE, 
        help="Name of the dataset", 
    )
    argparser.add_argument(
        "--n_principles", 
        type=int, 
        help="Number of principles to generate (only for synthetic datasets)", 
    )
    argparser.add_argument(
        "--n_comparisons", 
        type=int, 
        help="Number of comparisons to generate (only for synthetic datasets)", 
    )
    argparser.add_argument(
        "--sampling", 
        type=str, 
        choices=config.SAMPLING_SPACE, 
        help="Sampling method", 
    )
    argparser.add_argument(
        "--voting", 
        type=str, 
        choices=config.VOTING_SPACE, 
        help="Voting method", 
    )
    argparser.add_argument(
        "--noise", 
        type=float, 
        default=None, 
        help="Number of comparisons to generate (only for synthetic noisy datasets)", 
    )

    # Add benchmarking argument(s)
    argparser.add_argument(
        "--benchmarks", 
        type=str, nargs="+", 
        choices=["fitting_duration", "ranking_relevance", "converging_paces"], 
        default=["fitting_duration", "ranking_relevance", "converging_paces"], 
        help="List of benchmarks", 
    )

    argparser.add_argument(
        "--ranker_space", 
        type=str, nargs="+", 
        choices=config.RANKER_SPACE, 
        default=config.RANKER_SPACE, 
        help="List of names of rankers", 
    )

    argparser.add_argument(
        "--n_runs", 
        type=int, 
        default=config.N_RUNS, 
        help="Number of runs", 
    )
    argparser.add_argument(
        "--n_runs_warmup", 
        type=int, 
        default=config.N_RUNS_WARMUP, 
        help="Number of runs for warmup (for fitting durations)", 
    )

    argparser.add_argument(
        "--learnings", 
        type=str, nargs="+", 
        choices=["offline", "online"], 
        default=["offline", "online"], 
        help="List of learnings", 
    )
    argparser.add_argument(
        "--l_batches", 
        type=int, 
        default=config.L_BATCHES, 
        help="Length of batches", 
    )

    # Add other argument(s)
    argparser.add_argument(
        "--show_figs", 
        action="store_true", 
        help="Show figures ?", 
    )

    return argparser


def main_fitting_duration(dataset: MasterDataset, 
    ranker_space: list[str], 
    n_runs: int, n_runs_warmup: int=0, 
    offline:bool=True, online: bool=True, l_batches: int=1, 
    show_figs: bool=False, 
    ) -> None:
    """Main function for benchmarking fitting durations.
    
    Args:
        dataset (MasterDataset): dataset to benchmark.
        ranker_space (list[str]): list of rankers to benchmark.
        n_runs (int): number of runs.
        n_runs_warmup (int, optional): number of runs for warmup (for fitting durations). Defaults to 0.
        offline (bool, optional): toggle for offline learning. Defaults to True.
        online (bool, optional): toggle for online learning. Defaults to True.
        l_batches (int, optional): length of batches. Defaults to 1.
        show_figs (bool, optional): show figures ?. Defaults to False.
    """

    # Assess fitting durations
    fitting_durations = benchmarking_fitting_duration.assess_fitting_durations(
        dataset=dataset, 
        ranker_space=ranker_space, 
        n_runs=n_runs, n_runs_warmup=n_runs_warmup, 
        offline=offline, online=online, l_batches=l_batches, 
    )

    # Format fitting durations
    df_fitting_durations = benchmarking_fitting_duration.format_fitting_durations(
        fitting_durations, 
        dataset_params_columns=list(dataset.params.keys()), 
    )

    # Plot fitting durations
    dataset_infos = {"dataset": dataset.name, **dataset.params}
    benchmark_infos = {"n_runs": n_runs, "l_batches": l_batches if online else None}
    fig = visualizing_benchmark.plot_fitting_durations(
        df_fitting_durations, 
        dataset_infos=dataset_infos, benchmark_infos=benchmark_infos, 
    )
    fig_learning = visualizing_benchmark.plot_fitting_durations_learning(
        df_fitting_durations, 
        dataset_infos=dataset_infos, benchmark_infos=benchmark_infos, 
    )

    # Save fitting durations
    df_fitting_durations.to_csv(
        utils_paths.compose_path_result_fitting_durations(
            config.PATH_RESULTS, dataset.id, "tables/fitting_durations.csv", 
        ), 
    )
    utils_figs.save_figure(fig, 
        utils_paths.compose_path_result_fitting_durations(
            config.PATH_RESULTS, dataset.id, "figs/fitting_durations", 
        ), 
    )
    utils_figs.save_figure(fig_learning, 
        utils_paths.compose_path_result_fitting_durations(
            config.PATH_RESULTS, dataset.id, "figs/fitting_durations_learning", 
        ), 
    )

    # Show fitting durations
    if show_figs:
        fig.show()
        fig_learning.show()


def main_ranking_relevance(dataset: MasterDataset, 
    ranker_space: list[str], 
    n_runs: int, 
    show_figs: bool=False, 
    ) -> None:
    """Main function for benchmarking ranking relevances.
    
    Args:
        dataset (MasterDataset): dataset to benchmark.
        ranker_space (list[str]): list of rankers to benchmark.
        n_runs (int): number of runs.
        show_figs (bool, optional): show figures ?. Defaults to False.
    """

    # Assess ranking relevances
    ranking_relevances = benchmarking_ranking_relevance.assess_ranking_relevances(dataset=dataset, 
        ranker_space=ranker_space, 
        n_runs=n_runs, 
    )

    # Format & Enrich ranking relevances
    df_ranking_relevances = (
        benchmarking_ranking_relevance.format_ranking_relevances(
            ranking_relevances, 
            dataset_params_columns=list(dataset.params.keys()), 
        )
        .pipe(
            benchmarking_ranking_relevance.enrich_ranking_relevances, 
            dataset_params_columns=list(dataset.params.keys()), 
        )
    )

    # Compute ranking relevance matrixes
    reference = "rank_real" if dataset.real_ranking is not None else "rank_best"
    df_matrixes = df_ranking_relevances.pipe(
        benchmarking_ranking_relevance.compute_ranking_relevance_matrixes, 
        reference=reference, 
    )

    # Compute ranking relevances
    df_ranking_relevances = df_ranking_relevances.pipe(
        benchmarking_ranking_relevance.compute_ranking_relevances, 
        dataset_params_columns=list(dataset.params.keys()), 
    )

    # Plot fitting durations
    dataset_infos = {"dataset": dataset.name, **dataset.params}
    benchmark_infos = {"n_runs": n_runs}
    fig = visualizing_benchmark.plot_ranking_relevances(df_ranking_relevances, 
        dataset_infos=dataset_infos, benchmark_infos=benchmark_infos, 
    )
    fig_reference = visualizing_benchmark.plot_ranking_relevances_reference(df_ranking_relevances, 
        dataset_infos=dataset_infos, benchmark_infos=benchmark_infos, 
    )
    figs_matrix = {
        ranker_name: visualizing_benchmark.plot_ranking_relevance_matrix(df_matrixes.xs(ranker_name, level="ranker"), 
            ranker_name=ranker_name, reference=reference, 
            dataset_infos=dataset_infos, benchmark_infos=benchmark_infos, 
        )
        for ranker_name in ranker_space
    }


    # Save fitting durations
    df_ranking_relevances.to_csv(
        utils_paths.compose_path_result_ranking_relevances(
            config.PATH_RESULTS, dataset.id, "tables/ranking_relevances.csv", 
            ), 
    )
    utils_figs.save_figure(fig, 
        utils_paths.compose_path_result_ranking_relevances(
            config.PATH_RESULTS, dataset.id, "figs/ranking_relevances", 
        ), 
    )
    utils_figs.save_figure(fig_reference, 
        utils_paths.compose_path_result_ranking_relevances(
            config.PATH_RESULTS, dataset.id, "figs/ranking_relevances_reference", 
        ), 
    )
    for ranker_name, fig_matrix in figs_matrix.items():
        utils_figs.save_figure(fig_matrix, 
            utils_paths.compose_path_result_ranking_relevances(
                config.PATH_RESULTS, dataset.id, f"figs/ranking_relevances_matrix___{ranker_name}", 
            ), 
        )

    # Show fitting durations
    if show_figs:
        fig.show()
        fig_reference.show()
        for _, fig_matrix in figs_matrix.items():
            fig_matrix.show()


def main_converging_paces(dataset: MasterDataset,
    ranker_space: list[str], 
    n_runs: int, 
    l_batches: int, 
    show_figs: bool=False, 
    ) -> None:
    """Main function for benchmarking converging paces.

    Args:
        dataset (MasterDataset): dataset to benchmark.
        ranker_space (list[str]): list of rankers to benchmark.
        n_runs (int): number of runs.
        l_batches (int): length of batches.
        show_figs (bool, optional): show figures ?. Defaults to False.
    """

    # Assess converging paces
    converging_paces = benchmarking_converging_paces.assess_converging_paces(dataset=dataset, 
        ranker_space=ranker_space,  
        n_runs=n_runs, 
        l_batches=l_batches, 
    )

    # Format & Enrich & Compute converging paces
    df_converging_paces = (
        benchmarking_converging_paces.format_converging_paces(
            converging_paces,
            dataset_params_columns=list(dataset.params.keys()),
        )
        .pipe(
            benchmarking_converging_paces.enrich_converging_paces,
        )
        .pipe(
            benchmarking_converging_paces.compute_converging_paces,
            dataset_params_columns=list(dataset.params.keys()),
        )
    )

    # Plot converging paces
    dataset_infos = {"dataset": dataset.name, **dataset.params}
    benchmark_infos = {"n_runs": n_runs}
    fig = visualizing_benchmark.plot_converging_paces(df_converging_paces, 
        dataset_infos=dataset_infos, benchmark_infos=benchmark_infos, 
    )

    # Save converging paces
    df_converging_paces.to_csv(
        utils_paths.compose_path_result_converging_paces(
            config.PATH_RESULTS, dataset.id, "tables/converging_paces.csv", 
        ),
    )
    utils_figs.save_figure(fig,
        utils_paths.compose_path_result_converging_paces(
            config.PATH_RESULTS, dataset.id, "figs/converging_paces", 
        ),
    )

    # Show converging paces
    if show_figs:
        fig.show()



if __name__ == "__main__":
    logging.info("Main benchmarking...")

    # Parse arguments
    args = setup_argparser().parse_args()

    # Get dataset
    dataset = MasterDataset(dataset_name=args.dataset_name, 
        # params for authentic dataset
        path_data_processed=config.PATH_DATA_PROCESSED, 
        # params for synthetic dataset
        n_principles=args.n_principles, n_comparisons=args.n_comparisons, 
        voting=args.voting, sampling=args.sampling, 
            # params for noisy dataset
            noise=args.noise, 
    )
    logging.info("dataset: %s", dataset.id)

    
    # Benchmark fitting durations
    if "fitting_duration" in args.benchmarks:
        logging.info("benchmarking fitting durations...")
        main_fitting_duration(dataset=dataset, 
            ranker_space=args.ranker_space, 
            n_runs=args.n_runs, n_runs_warmup=args.n_runs_warmup, 
            offline="offline" in args.learnings, online="online" in args.learnings, l_batches=args.l_batches, 
            show_figs=args.show_figs, 
        )


    # Benchmark ranking relevances
    if "ranking_relevance" in args.benchmarks:
        logging.info("benchmarking ranking relevances...")
        main_ranking_relevance(dataset=dataset,
            ranker_space=args.ranker_space, 
            n_runs=args.n_runs, 
            show_figs=args.show_figs, 
        )


    # Benchmark converging paces
    if "converging_paces" in args.benchmarks:
        logging.info("benchmarking converging paces...")
        main_converging_paces(dataset=dataset, 
            ranker_space=args.ranker_space, 
            n_runs=args.n_runs, 
            l_batches=args.l_batches, 
            show_figs=args.show_figs, 
        )
