# master-thesis-sovereign-decision-making

## Setup project

Our environment is setup with poetry, this means that you can activate it by doing:
```sh
poetry shell
```

If you want to add additional packages, inside of poetry shell, you can do it so by  running:

> poetry add package-name

## Check quality

> python -m pylint src

> python -m mypy src --ignore-missing-import

## Addressing the data used in the project:

- All data used is exported from our database, running online. Files are structured as follows:

- Experiments.csv: Contains information about experiments and probes. These are containers for a specific "question" regarding specific principles.
- Feedbacks.csv: Generally contains feedback given by users be it, "like", "dislike" or "free text". 
- Results.csv: This is one of the most relevant tables. It contains the following columns: [id, experiment_id, lhs_id, rhs_id, user_id, user_groups,choice, insertion_date, update_date, processed]

