import pandas as pd
import plotly.graph_objects as go



LABELS = {
    "n_votes": "<b>Number of votes</b>", 
    "n_principles": "<b>Number of principles</b>", 
    "n_comparisons": "<b>Number of comparisons</b>", 

    "rank": "<b>Rank</b>", 
    "ranker": "<b>Ranker</b>", 
    "learning": "<b>Learning</b>", 
    "principle": "<b>Principle</b>", 
    "reference": "<b>Reference</b>", 

    "fitting_duration": "<b>Fitting duration</b><br><sup><i>(seconds)</i></sup>",
    "ranking_relevance": "<b>Ranking relevance</b><br><sup><i>(kendall's τ)</i></sup>", 
    "converging_pace": "<b>Converging paces</b><br><sup><i>(kendall's τ)</i></sup>", 
}

LABEL_ALIASES = {
    ranker: ranker.capitalize().replace("_", "")
    for ranker in ["win_rate", "elo", "true_skill", "eigenvector_centrality", "bradley_terry"]
}

CATEGORY_ORDERS = {
    "learning": ["offline", "online"], 
    "reference": ["rank_last", "rank_best", "rank_real"], 

    "ranker": ["win_rate", "elo", "true_skill", "eigenvector_centrality", "bradley_terry"], 
}

COLOR_DISCRETE_MAP = {
    "win_rate": "red", 
    "elo": "orange", 
    "true_skill": "green", 
    "eigenvector_centrality": "blue", 
    "bradley_terry": "purple", 

    "offline": "lightcoral", 
    "online": "lightgreen", 

    "rank_last": "darkred", 
    "rank_best": "darkorange", 
    "rank_real": "darkgreen", 
}

def filter_category_orders(category_orders: dict[str, list], df: pd.DataFrame) -> dict[str, list]:
    """Filter category orders.
    
    Args:
        category_orders (dict[str, list]): category orders.
        df (pd.DataFrame): dataframe.
        
    Returns:
        dict[str, list]: filtered category orders.
    """

    df = df.reset_index()
    category_orders = {
        key: [value for value in values if value in df[key].unique()]
        for key, values in category_orders.items()
        if key in df.columns
    }
    return category_orders


def format_name(name: str) -> str:
    """Format name."""

    return name.title().replace("_", "")

def format_variable(name: str) -> str:
    """Format variable name."""

    return name.capitalize().replace("_", " ")


def format_traces_names(fig: go.Figure) -> go.Figure:
    """Format traces names.

    Args:
        fig (go.Figure): figure to format.
    
    Returns:
        go.Figure: formatted figure.
    """

    fig = fig.for_each_trace(lambda trace: trace.update(name=format_name(trace.name)))
    return fig

def compose_fig_title(title: str, subtitle: str|None=None, **params) -> str:
    """Compose figure title.

    Args:
        title (str): title.
        subtitle (str, optional): subtitle. Defaults to None.
        **params: parameters.
    
    Returns:
        str: composed title.
    """

    title = f"<b>{title}</b>"

    if subtitle is not None:
        title += f" <b> - </b>{subtitle}"

    if len(params) > 0:
        title_params = ", ".join([
            f"{format_variable(variable)}: <i>{value}</i>"
            for variable, value in params.items()
            if value is not None
        ])
        title += f"<br><sup>{title_params}</sup>"
        
    return title
