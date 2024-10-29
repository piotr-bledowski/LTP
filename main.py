import argparse
import os
import sys
import warnings
from typing import Union
import optuna
import wandb
import pickle
from perform_experiment import perform_experiment

# the only warning raised is ConvergenceWarning for linear SVM, which is
# acceptable (max_iter is already higher than default); unfortunately, we
# have to do this globally for all warnings to affect child processes in
# cross-validation
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # also affect subprocesses


def ensure_bool(data: Union[bool, str]) -> bool:
    if isinstance(data, bool):
        return data
    elif data.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif data.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Local Topological Profile")
    parser.add_argument(
        "--dataset_name",
        choices=[
            "all",
            "DD",
            "NCI1",
            "PROTEINS_full",
            "ENZYMES",
            "IMDB-BINARY",
            "IMDB-MULTI",
            "REDDIT-BINARY",
            "REDDIT-MULTI-5K",
            "COLLAB",
        ],
        default="all",
        help="Dataset name, use 'all' to run the entire benchmark.",
    )
    parser.add_argument(
        "--degree_sum",
        type=ensure_bool,
        default=False,
        help="Add degree sum feature from LDP?",
    )
    parser.add_argument(
        "--shortest_paths",
        type=ensure_bool,
        default=False,
        help="Add shortest paths feature from LDP?",
    )
    parser.add_argument(
        "--edge_betweenness",
        type=ensure_bool,
        default=True,
        help="Add edge betweenness centrality proposed in LTP?",
    )
    parser.add_argument(
        "--jaccard_index",
        type=ensure_bool,
        default=True,
        help="Add Jaccard Index proposed in LTP?",
    )
    parser.add_argument(
        "--local_degree_score",
        type=ensure_bool,
        default=True,
        help="Add Local Degree Score proposed in LTP?",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=50,
        help="Number of bins for aggregation.",
    )
    parser.add_argument(
        "--normalization",
        choices=[
            "none",
            "graph",
            "dataset",
        ],
        default="none",
        help="Normalization scheme.",
    )
    parser.add_argument(
        "--aggregation",
        choices=[
            "histogram",
            "EDF",
        ],
        default="histogram",
        help="Aggregation scheme.",
    )
    parser.add_argument(
        "--log_degree",
        type=bool,
        default=False,
        help="Use log scale for degree features from LDP?",
    )
    parser.add_argument(
        "--model_type",
        choices=[
            "LinearSVM",
            "KernelSVM",
            "RandomForest",
        ],
        default="RandomForest",
        help="Classification algorithm to use.",
    )
    parser.add_argument(
        "--tune_feature_extraction_hyperparams",
        type=bool,
        default=False,
        help="Perform hyperparameter tuning for feature extraction?",
    )
    parser.add_argument(
        "--tune_model_hyperparams",
        type=bool,
        default=False,
        help="Perform hyperparameter tuning for classification model?",
    )
    parser.add_argument(
        "--use_features_cache",
        type=bool,
        default=True,
        help="Cache calculated features to speed up subsequent experiments?",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Should print out verbose output?",
    )

    return parser.parse_args()


def run_experiment(trial, dataset_name):
    config = {
        'dataset_name': dataset_name,
        'degree_sum': trial.suggest_categorical("degree_sum", [1, 0]),
        'shortest_paths': trial.suggest_categorical("shortest_paths", [1, 0]),
        'edge_betweenness': trial.suggest_categorical("edge_betweenness", [1]),
        'degree_centrality': trial.suggest_categorical("degree_centrality", [1, 0]),
        #'closeness': trial.suggest_categorical("closeness", [1, 0]),
        'local_clustering_coefficient': trial.suggest_categorical("local_clustering_coefficient", [1, 0]),
        'pagerank': trial.suggest_categorical("pagerank", [1, 0]),
        'eigenvector_centrality': trial.suggest_categorical("eigenvector_centrality", [1, 0]),
        'algebraic_distance': trial.suggest_categorical("algebraic_distance", [1, 0]),
        'diameter': trial.suggest_categorical("diameter", [1, 0]),
        'density': trial.suggest_categorical("density", [1, 0]),
        'preferential_attachment': trial.suggest_categorical("preferential_attachment", [1, 0]),
        'common_neighbor': trial.suggest_categorical("common_neighbor", [1, 0]),
        'katz_index': trial.suggest_categorical("katz_index", [1, 0]),
        'jaccard_index': trial.suggest_categorical("jaccard_index", [1, 0]),
        'adjusted_rand': trial.suggest_categorical("adjusted_rand", [1, 0]),
        'adamic_adar': trial.suggest_categorical("adamic_adar", [1, 0]),
        'local_degree_score': trial.suggest_categorical("local_degree_score", [1, 0]),
        'local_similarity_score': trial.suggest_categorical("local_similarity_score", [1, 0]),
        'scan': trial.suggest_categorical("scan", [1, 0]),
        'n_bins': trial.suggest_categorical("n_bins", [args.n_bins]),
        'normalization': trial.suggest_categorical("normalization", [args.normalization]),
        'aggregation': trial.suggest_categorical("aggregation", [args.aggregation]),
        'log_degree': trial.suggest_categorical("log_degree", [args.log_degree]),
        'model_type': trial.suggest_categorical("model_type", [args.model_type])
    }

    with wandb.init(config=config, project="LTP"):
        acc_mean, acc_stddev = perform_experiment(**config)

        wandb.log({
            'acc_mean': acc_mean,
            'acc_std': acc_stddev,
        })
    return acc_mean


def objective(trial, dataset):
    # Run the experiment for a specific trial and dataset
    acc_mean = run_experiment(trial, dataset)
    return acc_mean


if __name__ == "__main__":
    args = parse_args()

    datasets = [
        "DD",
        "NCI1",
        "PROTEINS_full",
        "REDDIT-BINARY",
        "REDDIT-MULTI-5K"
    ]

    best_trials = {}

    for dataset in datasets:
        print(f"Starting study for dataset: {dataset}")

        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(direction='maximize', sampler=sampler)

        study.optimize(lambda trial: objective(trial, dataset), n_trials=100)

        best_trials[dataset] = {
            'params': study.best_trial.params,
            'acc_mean': study.best_trial.value
        }

        print(f"Best trial for {dataset}: {study.best_trial.params}")
        print(f"Best acc_mean for {dataset}: {study.best_trial.value}")

    print("\nSummary of best trials for all datasets:")
    for dataset, result in best_trials.items():
        print(f"{dataset}: Best params = {result['params']}, Best acc_mean = {result['acc_mean']}")

    with open(os.path.join('results', 'results.pkl'), 'wb') as f:
        pickle.dump(best_trials, f)
