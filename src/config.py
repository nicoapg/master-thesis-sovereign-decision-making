import os
import logging
import plotly.io as pio



# ==================================================
# LOGGER
# ==================================================

logging.basicConfig(
    level=logging.DEBUG, 

    style="{", 
    format="[{asctime}] {levelname}: {message}", 
    datefmt="%Y-%m-%d %H:%M:%S", 

    handlers=[
        logging.StreamHandler(), 
        logging.FileHandler(
            filename="logs/logs.log", 
            mode="a", 
        ), 
    ], 
)

# ==================================================
# PATHS
# ==================================================

PATH_DATA = "data"
PATH_DATA_RAW = os.path.join(PATH_DATA, "raw")
PATH_DATA_PROCESSED = os.path.join(PATH_DATA, "processed")

PATH_RESULTS = "results"

# ==================================================
# PROCESSING
# ==================================================

# datasets
DATASETS_REFERENTIAL = {
    "experiment001": {
        "PATH_FILE_PRINCIPLES": "experiment001-dummy-data.xlsx", 
        "PATH_FILES_COMPARISONS": "experiment001-10comparison-expert-{name}-labeled.xlsx", 
        "USERS_VARIABLES": ["name"], 
        "USERS_VALUES": [
            ("Alex",), 
            ("Christoph",), 
            ("Dominic",), 
            ("Fiona",), 
            ("Matthias",), 
            ("Michael",), 
        ], 
    }, 
    "experiment002": {
        "PATH_FILE_PRINCIPLES": "experiment002-dummy-data.xlsx", 
        "PATH_FILES_COMPARISONS": "experiment002-28comparison-{name}-background-{field}.xlsx", 
        "USERS_VARIABLES": ["name", "field"], 
        "USERS_VALUES": [
            ("anna", "physician-internal-medicine"), 
            ("gustavo", "data-scientist"), 
            ("martin", "physician"), 
            ("michael", "regulator"), 
            ("nicolas", "data-scientist"), 
            ("samuel-stalder", "law"), 
        ], 
    }, 
    "experiment003": {
        "PATH_FILE_PRINCIPLES": "DATA_sgmlp_dataset_principles_V2.xlsx", 
        "PATH_FILES_COMPARISONS": "experiment003-10comparisons-{name}-background-{field}___{sample}.xlsx", 
        "USERS_VARIABLES": ["name", "field", "sample"], 
        "USERS_VALUES": [
            ("ANONYM", "TEAC", "04"), 
            ("FABIO", "PHYSICIAN", "01"), 
            ("MARTIN", "LAWYER", "03"), 
            ("NICO", "DATASCIENTIST", "02"), 
        ], 
    }, 
}

# principles
PRINCIPLE_COLUMNS = ["principle", "PRINCIPLE"]

# comparions
LEFT_COLUMNS, RIGHT_COLUMNS = ["option_1"], ["option_2"]
VOTE_COLUMNS = ["vote_left_or_right", "vote_left_right_or_tie", "vote_left_or_right_or_tie"]
LEFT_VALUES, RIGHT_VALUES, TIE_VALUES, NAN_VALUE = ["left", "LEFT"], ["right", "RIGHT"], ["tie", "TIE"], "tie"

# ==================================================
# BENCHMARKING
# ==================================================

RANKER_SPACE = [
    "win_rate", 
    "elo", 
    "true_skill", 
    "eigenvector_centrality", 
    "bradley_terry", 
]
DATASET_SPACE = [
    "experiment001", "experiment002", "experiment003", 
    "dummy", 
    "synthetic", 
]
SAMPLING_SPACE, VOTING_SPACE = ["uniform"], ["absolute", "noisy"]

LEARNING_OFFLINE, LEARNING_ONLINE = True, True
L_BATCHES = 25

N_RUNS = 100
N_RUNS_WARMUP = 1

# ==================================================
# VISUALIZING
# ==================================================

pio.renderers.default = "notebook" #FIXME "browser"
pio.templates.default = "plotly_white"
