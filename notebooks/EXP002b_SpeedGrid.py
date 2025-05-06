#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("..")


# # Imports

# In[2]:


import tqdm
import itertools
import plotly.express as px

from src import config
from src.data import MasterDataset
from src.benchmarking import benchmarking_fitting_duration
from src.visualizing import visualizing_utils
from src.utils import utils_paths, utils_figs


# In[3]:


import plotly.io as pio
pio.renderers.default = "notebook"


# # Config

# In[4]:


EXPERIMENT_NAME = "speed"


# # Parameters

# In[5]:


DATASET_NAME = "dummy"
N_PRINCIPLES_SPACE = [5, 10, 50, 100]
N_COMPARISONS_SPACE = [10, 100, 1_000, 10_000]

RANKER_SPACE = config.RANKER_SPACE

N_RUNS = config.N_RUNS
N_RUNS_WARMUP = config.N_RUNS_WARMUP


# # main

# In[6]:


dataset_params_columns = ["n_principles", "n_comparisons"]

dataset_infos = {"dataset": DATASET_NAME}
benchmark_infos = {"n_runs": N_RUNS}


# ## assessing

# In[7]:


# Initialize fitting durations
fitting_durations = []

# Initialize progress
progress = tqdm.tqdm(total=len(N_PRINCIPLES_SPACE)*len(N_COMPARISONS_SPACE))

# Iterate over datasets
for n_principles, n_comparisons in itertools.product(N_PRINCIPLES_SPACE, N_COMPARISONS_SPACE):
    dataset = MasterDataset(DATASET_NAME, n_principles=n_principles, n_comparisons=n_comparisons)

    # Assess fitting durations
    fitting_durations.extend(
        benchmarking_fitting_duration.assess_fitting_durations(dataset=dataset, 
            ranker_space=RANKER_SPACE, 
            n_runs=N_RUNS, n_runs_warmup=N_RUNS_WARMUP, 
            offline=True, online=False, 
            show_progress=False, 
        )
    )
    
    # Update progress
    progress.update()

# Format fitting durations
df_fitting_durations = (
    benchmarking_fitting_duration.format_fitting_durations(fitting_durations, 
        dataset_params_columns=dataset_params_columns, 
    )
    
    .xs("offline", level="learning")
)
display(df_fitting_durations.head())


# ## visualizing

# ### grid

# In[8]:


df_plot = df_fitting_durations

fig_grid = (
    px.box(
        df_plot.reset_index(), 
        x="ranker", 
        y="fitting_duration", 
        color="ranker", 
        facet_col="n_principles", facet_row="n_comparisons", 

        labels={
            **visualizing_utils.LABELS, 
            "n_principles": "<b>p</b>", 
            "n_comparisons": "<b>c</b>", 
        }, 
        category_orders={
            **visualizing_utils.filter_category_orders(visualizing_utils.CATEGORY_ORDERS, df_plot), 
            "n_principles": N_PRINCIPLES_SPACE, 
            "n_comparisons": N_COMPARISONS_SPACE[::-1], 
        }, 
        color_discrete_map=visualizing_utils.COLOR_DISCRETE_MAP, 
        boxmode="overlay", 

        title=visualizing_utils.compose_fig_title(
            "Speed", 
            **dataset_infos, **benchmark_infos, 
        ), 
    )
    .update_xaxes(visible=False)
    .update_yaxes(rangemode="tozero")
)
fig_grid = visualizing_utils.format_traces_names(fig_grid)
fig_grid.show()


# ### matrixes

# In[9]:


df_plot = (
    df_fitting_durations
    
    .groupby(["ranker", "n_principles", "n_comparisons"])
    ["fitting_duration"]
    .mean()
    
    .unstack("n_comparisons")

    .sort_index(ascending=False, axis=0)
    .sort_index(ascending=True, axis=1)

    .round(3)
)

figs_matrix = {}
for ranker_name in RANKER_SPACE:
    fig_matrix = (
        px.imshow(
            df_plot.xs(ranker_name, level="ranker"), 
            text_auto=".1f", 
            
            labels={
                "x": visualizing_utils.LABELS["n_comparisons"], 
                "y": visualizing_utils.LABELS["n_principles"], 
                "color": visualizing_utils.LABELS["fitting_duration"], 
            }, 
            color_continuous_scale=[
                (0, "white"), 
                (1, visualizing_utils.COLOR_DISCRETE_MAP[ranker_name]), 
            ], range_color=[0, df_plot.xs(ranker_name, level="ranker").max().max()], 
            aspect="auto", 

            title=visualizing_utils.compose_fig_title(
                "Speed", visualizing_utils.format_name(ranker_name), 
                **dataset_infos, 
                **benchmark_infos, 
            ), 
        )
        .update_xaxes(type="category")
        .update_yaxes(type="category")
        .update_coloraxes(colorbar_tickformat=".1f")
        .update_traces(
            hoverongaps=False, 
            hovertemplate=(
                "<b>Principles:</b>%{y:,}"
                "<br>"
                "<b>Comparisons:</b>%{x:,}"
                "<extra>%{z:.3f}s</extra>"
            ), 
        )
    )
    fig_matrix.show()
    figs_matrix[ranker_name] = fig_matrix


# ## saving

# ### table

# In[10]:


df_fitting_durations.to_csv(
    utils_paths.compose_path_result_file(
        config.PATH_RESULTS, EXPERIMENT_NAME, "speed", "tables/speed.csv", 
    ), 
)


# ### figures

# In[11]:


utils_figs.save_figure(fig_grid, 
    utils_paths.compose_path_result_file(
        config.PATH_RESULTS, EXPERIMENT_NAME, "speed", "figs/speed_grid", 
    ), 
)


# In[12]:


for ranker_name, fig_matrix in figs_matrix.items():
    utils_figs.save_figure(fig_matrix, 
        utils_paths.compose_path_result_file(
            config.PATH_RESULTS, EXPERIMENT_NAME, "speed", f"figs/speed_matrix_{ranker_name}", 
        ), 
    )


# In[ ]:




