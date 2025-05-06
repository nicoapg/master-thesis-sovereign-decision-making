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
from src.benchmarking import benchmarking_converging_paces
from src.visualizing import visualizing_utils
from src.utils import utils_paths, utils_figs


# In[3]:


import plotly.io as pio
pio.renderers.default = "notebook"


# # Config

# In[4]:


EXPERIMENT_NAME = "accuracy"


# # Parameters

# In[5]:


DATASET_NAME = "synthetic"
N_PRINCIPLES, N_COMPARISONS = 25, 2_500
SAMPLING, VOTING = "uniform", "noisy"
NOISE_SPACE = [0, 0.1, 0.2, 0.4]

RANKER_SPACE = config.RANKER_SPACE

N_RUNS = config.N_RUNS
L_BATCHES = 100


# # main

# In[6]:


dataset_params_columns = ["sampling", "voting", "noise"]

dataset_infos = {"dataset": DATASET_NAME, "n_principles": N_PRINCIPLES, "n_comparisons": N_COMPARISONS}
benchmark_infos = {"n_runs": N_RUNS}


# ## assessing

# In[7]:


# Initialize ranking relevances
ranking_relevances = []

# Initialize progress
progress = tqdm.tqdm(total=len(NOISE_SPACE))

# Iterate over datasets
for noise in NOISE_SPACE:
    dataset = MasterDataset(
        DATASET_NAME, sampling=SAMPLING, voting=VOTING, noise=noise, 
        n_principles=N_PRINCIPLES, n_comparisons=N_COMPARISONS, 
    )

    # Assess ranking relevances
    ranking_relevances.extend(
        benchmarking_converging_paces.assess_converging_paces(dataset=dataset, 
            ranker_space=RANKER_SPACE, 
            n_runs=N_RUNS, l_batches=L_BATCHES, 
            show_progress=False, 
        )
    )
    
    # Update progress
    progress.update()

# Format ranking relevances
df_ranking_relevances = (
    benchmarking_converging_paces.format_converging_paces(
        ranking_relevances, 
        dataset_params_columns=dataset_params_columns, 
    )
    .pipe(
        benchmarking_converging_paces.enrich_converging_paces, 
    )
    .pipe(
        benchmarking_converging_paces.compute_converging_paces, 
        dataset_params_columns=dataset_params_columns, 
    )
    .xs("rank_real", level="reference")
)
display(df_ranking_relevances.head())


# ## visualizing

# ### grid

# In[1]:


df_plot = df_ranking_relevances.groupby(["dataset", "ranker", *dataset_params_columns, "n_votes"]).mean()

fig_grid = (
    px.line(
        df_plot.reset_index(), 
        x="n_votes", 
        y="converging_pace", 
        color="ranker", 
        facet_row="noise", 

        labels={
            **visualizing_utils.LABELS, 
            "noise": "<b>Noise</b>", 
            "n_votes": "<b>Votes</b>", 
        }, 
        category_orders={
            **visualizing_utils.filter_category_orders(visualizing_utils.CATEGORY_ORDERS, df_plot), 
            "noise": NOISE_SPACE, 
        }, 
        color_discrete_map=visualizing_utils.COLOR_DISCRETE_MAP, 

        title=visualizing_utils.compose_fig_title(
            "Accuracy", 
            **dataset_infos, **benchmark_infos, 
        ), 
    )
    .update_yaxes(matches=None)
)
fig_grid = visualizing_utils.format_traces_names(fig_grid)
fig_grid.show()


# ## saving

# ### table

# In[9]:


df_ranking_relevances.to_csv(
    utils_paths.compose_path_result_file(
        config.PATH_RESULTS, EXPERIMENT_NAME, "accuracy", "tables/accuracy.csv", 
    ), 
)


# ### figures

# In[10]:


utils_figs.save_figure(fig_grid, 
    utils_paths.compose_path_result_file(
        config.PATH_RESULTS, EXPERIMENT_NAME, "accuracy", "figs/accuracy_grid", 
    ), 
)


# In[ ]:




