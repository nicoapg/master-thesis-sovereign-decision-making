import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .visualizing_utils import (
    LABELS, LABEL_ALIASES, CATEGORY_ORDERS, COLOR_DISCRETE_MAP, filter_category_orders, 
    format_name, format_traces_names, compose_fig_title, 
)



# Fitting durations

def plot_fitting_durations(df: pd.DataFrame, 
    dataset_infos: dict|None=None, benchmark_infos: dict|None=None) -> go.Figure:
    """Plot figure of fitting durations.

    Args:
        df (pd.DataFrame): dataframe of fitting durations.
        dataset_infos (dict, optional): dataset informations. Defaults to None.
        benchmark_infos (dict, optional): benchmark informations. Defaults to None.

    Returns:
        go.Figure: figure of fitting durations.
    """

    if dataset_infos is None:
        dataset_infos = {}
    if benchmark_infos is None:
        benchmark_infos = {}

    fig = (
        px.box(
            df.reset_index(), 
            x="ranker", 
            y="fitting_duration", 
            color="ranker", 
            facet_row="learning", 

            labels=LABELS, 
            category_orders=filter_category_orders(CATEGORY_ORDERS, df), 
            color_discrete_map=COLOR_DISCRETE_MAP, 
            boxmode="overlay", 

            title=compose_fig_title(
                "Fitting durations", 
                **dataset_infos, 
                **benchmark_infos, 
            ), 
        )
        .update_yaxes(rangemode="tozero")
    )
    fig = format_traces_names(fig)
    fig = fig.update_xaxes(labelalias=LABEL_ALIASES)
    return fig

def plot_fitting_durations_learning(df: pd.DataFrame, 
    dataset_infos: dict|None=None, benchmark_infos: dict|None=None) -> go.Figure:
    """Plot figure of fitting durations learning.

    Args:
        df (pd.DataFrame): dataframe of fitting durations.
        dataset_infos (dict, optional): dataset informations. Defaults to None.
        benchmark_infos (dict, optional): benchmark informations. Defaults to None.

    Returns:
        go.Figure: figure of fitting durations.
    """

    if dataset_infos is None:
        dataset_infos = {}
    if benchmark_infos is None:
        benchmark_infos = {}

    fig = (
        px.box(
            df.reset_index(), 
            x="ranker", 
            y="fitting_duration", 
            color="learning", 

            labels=LABELS, 
            category_orders=filter_category_orders(CATEGORY_ORDERS, df), 
            color_discrete_map=COLOR_DISCRETE_MAP, 
            boxmode="group", 

            title=compose_fig_title(
                "Fitting durations", 
                **dataset_infos, 
                **benchmark_infos, 
            ), 
        )
        .update_yaxes(rangemode="tozero")
    )
    fig = format_traces_names(fig)
    fig = fig.update_xaxes(labelalias=LABEL_ALIASES)
    return fig


# Ranking relevances

def plot_ranking_relevances(df: pd.DataFrame, 
    dataset_infos: dict|None=None, benchmark_infos: dict|None=None) -> go.Figure:
    """Plot figure of ranking relevances.

    Args:
        df (pd.DataFrame): dataframe.
        dataset_infos (dict, optional): dataset informations. Defaults to None.
        benchmark_infos (dict, optional): benchmark informations. Defaults to None.

    Returns:
        go.Figure: figure of ranking relevances.
    """

    if dataset_infos is None:
        dataset_infos = {}
    if benchmark_infos is None:
        benchmark_infos = {}

    fig = (
        px.box(
            df.reset_index(), 
            x="ranker", 
            y="ranking_relevance", 
            color="ranker", 
            facet_row="reference", 

            labels=LABELS, 
            category_orders=filter_category_orders(CATEGORY_ORDERS, df), 
            color_discrete_map=COLOR_DISCRETE_MAP, 
            boxmode="overlay", 

            title=compose_fig_title(
                "Ranking relevances", 
                **dataset_infos, 
                **benchmark_infos, 
            ), 
        )
    )
    fig = format_traces_names(fig)
    fig = fig.update_xaxes(labelalias=LABEL_ALIASES)
    return fig

def plot_ranking_relevances_reference(df: pd.DataFrame, 
    dataset_infos: dict|None=None, benchmark_infos: dict|None=None) -> go.Figure:
    """Plot figure of fitting durations reference.

    Args:
        df (pd.DataFrame): dataframe.
        dataset_infos (dict, optional): dataset informations. Defaults to None.
        benchmark_infos (dict, optional): benchmark informations. Defaults to None.
    
    Returns:
        go.Figure: figure of ranking relevance reference.
    """

    if dataset_infos is None:
        dataset_infos = {}
    if benchmark_infos is None:
        benchmark_infos = {}

    fig = (
        px.box(
            df.reset_index(), 
            x="ranker", 
            y="ranking_relevance", 
            color="reference", 
            
            labels=LABELS, 
            category_orders=filter_category_orders(CATEGORY_ORDERS, df), 
            color_discrete_map=COLOR_DISCRETE_MAP, 
            boxmode="group", 

            title=compose_fig_title(
                "Ranking relevances", 
                **dataset_infos, 
                **benchmark_infos, 
            ), 
        )
    )
    fig = format_traces_names(fig)
    fig = fig.update_xaxes(labelalias=LABEL_ALIASES)
    return fig

def plot_ranking_relevance_matrix(df: pd.DataFrame, ranker_name: str, reference: str, 
    dataset_infos: dict|None=None, benchmark_infos: dict|None=None) -> go.Figure:
    """Plot figure of ranking relevance matrix.

    Args:
        df (pd.DataFrame): dataframe.
        ranker_name (str): name of the ranker.
        reference (str): reference.
        dataset_infos (dict, optional): dataset informations. Defaults to None.
        benchmark_infos (dict, optional): benchmark informations. Defaults to None.

    Returns:
        go.Figure: figure of ranking relevance matrix.
    """

    if dataset_infos is None:
        dataset_infos = {}
    if benchmark_infos is None:
        benchmark_infos = {}

    fig = (
        px.imshow(
            df, 
            text_auto=".0%", 
            
            labels={
                "x": LABELS["rank"], 
                "y": LABELS["principle"], 
                "color": LABELS["ranking_relevance"], 
            }, 
            color_continuous_scale=[
                (0, "white"), 
                (1, COLOR_DISCRETE_MAP[ranker_name]), 
            ], range_color=[0, 1], 
            aspect="auto", 

            title=compose_fig_title(
                "Ranking Relevances", format_name(ranker_name), 
                **dataset_infos, 
                reference=format_name(reference), 
                **benchmark_infos, 
            ), 
        )
        .update_yaxes(showticklabels=False)
        .update_coloraxes(colorbar_tickformat=".0%")
        .update_traces(
            hoverongaps=False, 
            hovertemplate=(
                "%{y}"
                "<extra>%{z:.1%}</extra>"
            ), 
        )
    )
    return fig


# Converging paces

def plot_converging_paces(df: pd.DataFrame, 
    dataset_infos: dict|None=None, benchmark_infos: dict|None=None) -> go.Figure:
    """Plot figure of converging paces.

    Args:
        df (pd.DataFrame): dataframe.
        dataset_infos (dict, optional): dataset informations. Defaults to None.
        benchmark_infos (dict, optional): benchmark informations. Defaults to None.

    Returns:
        go.Figure: figure of converging paces.
    """

    if dataset_infos is None:
        dataset_infos = {}
    if benchmark_infos is None:
        benchmark_infos = {}

    df = (
        df
        
        .groupby(["dataset", "ranker", "n_votes", "reference"])
        ["converging_pace"]
        .agg(["mean", "sem"])
        .assign(
            converging_pace=lambda df: df["mean"], 
            moe=lambda df: 1.96 * df["sem"], 
        )
    )

    fig = (
        px.line(
            df.reset_index(), 
            x="n_votes", 
            y="converging_pace", error_y="moe", 
            color="ranker", 
            facet_row="reference", 

            labels=LABELS, 
            category_orders=filter_category_orders(CATEGORY_ORDERS, df), 
            color_discrete_map=COLOR_DISCRETE_MAP, 

            title=compose_fig_title(
                "Converging paces", 
                **dataset_infos, 
                **benchmark_infos, 
            ), 
        )
    )
    fig = format_traces_names(fig)
    return fig
