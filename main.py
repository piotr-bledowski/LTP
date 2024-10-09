import argparse
import os
import sys
import time
import warnings
from typing import Union
import wandb

from data_loading import DATASET_NAMES
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


def run_experiment(config=None):
    with wandb.init(config=config):
        config = wandb.config
        acc_mean, acc_stddev = perform_experiment(**config)
        wandb.log({
            'acc_mean': acc_mean,
            'acc_std': acc_stddev,
        })


if __name__ == "__main__":
    args = parse_args()

    if args.dataset_name == "all":
        datasets = DATASET_NAMES
    else:
        datasets = [args.dataset_name]

    sweep_configuration = {
        'method': 'bayes',
        'metric': {'name': 'acc_mean', 'goal': 'maximize'},
        'parameters': {
            'dataset_name': {'value': "DD"},
            'degree_sum': {'values': [True, False]},
            'shortest_paths': {'values': [True, False]},
            'edge_betweenness': {'values': [True, False]},
            'degree_centrality': {'values': [True, False]},
            'closeness': {'values': [True, False]},
            'local_clustering_coefficient': {'values': [True, False]},
            'pagerank': {'values': [True, False]},
            'eigenvector_centrality': {'values': [True, False]},
            'algebraic_distance': {'values': [True, False]},
            'diameter': {'values': [True, False]},
            'density': {'values': [True, False]},
            'preferential_attachment': {'values': [True, False]},
            'common_neighbor': {'values': [True, False]},
            'katz_index': {'values': [True, False]},
            'jaccard_index': {'values': [True, False]},
            'adjusted_rand': {'values': [True, False]},
            'adamic_adar': {'values': [True, False]},
            'local_degree_score': {'values': [True, False]},
            'local_similarity_score': {'values': [True, False]},
            'scan': {'values': [True, False]},
            'n_bins': {'values': [args.n_bins]},
            'normalization': {'values': [args.normalization]},
            'aggregation': {'values': [args.aggregation]},
            'log_degree': {'values': [args.log_degree]},
            'model_type': {'values': [args.model_type]},
        }
    }

    sweep_id = wandb.sweep(sweep_configuration, project="LTP")
    wandb.agent(sweep_id, function=run_experiment)  # TODO add count


    # for dataset_name in DATASET_NAMES:
    #     for descriptor_combination in descriptors:
    #         start = time.time()
    #         print(dataset_name)
    #         acc_mean, acc_stddev = perform_experiment(
    #             dataset_name=dataset_name,
    #             degree_sum=args.degree_sum,
    #             shortest_paths=descriptor_combination[0],
    #             edge_betweenness=descriptor_combination[1],
    #             jaccard_index=descriptor_combination[2],
    #             adjusted_rand=descriptor_combination[3],
    #             adamic_adar=descriptor_combination[4],
    #             local_degree_score=descriptor_combination[5],
    #             local_similarity_score=descriptor_combination[6],
    #             n_bins=args.n_bins,
    #             normalization=args.normalization,
    #             aggregation=args.aggregation,
    #             log_degree=args.log_degree,
    #             model_type=args.model_type,
    #             tune_feature_extraction_hyperparams=args.tune_feature_extraction_hyperparams,
    #             tune_model_hyperparams=args.tune_model_hyperparams,
    #             use_features_cache=args.use_features_cache,
    #             verbose=args.verbose,
    #         )

