import argparse
import os
import sys
import warnings
import pickle
from pathlib import Path
from typing import Union
from time import time
import matplotlib.pyplot as plt
from feature_extraction import extract_features
from data_loading import DATASET_NAMES
from perform_experiment import perform_experiment, perform_experiment_calculate_importance
import pandas as pd
from data_loading import load_dataset
from caching import cache_features
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

def create_cached_features():
    # create cache table
    params = {
            'degree sum': False,
            'shortest paths': False,
            'edge betweenness': False,
            'degree centrality': False,
            'local clustering coefficient': False,
            'pagerank': False,
            'eigenvector centrality': False,
            'algebraic distance': False,
            'diameter': False,
            'density': False,
            'preferential attachment': False,
            'common neighbor': False,
            'katz index': False,
            'jaccard index': False,
            'adjusted rand': False,
            'adamic adar': False,
            'local degree score': False,
            'local similarity score': False,
            'scan': False,
        }
    for dataset_name in datasets:
        for feature_name in params.keys():
            params[feature_name] = True
            extracted_data = extract_features(
                        dataset= load_dataset(dataset_name),
                        degree_sum=params['degree sum'],
                        shortest_paths=params['shortest paths'],
                        edge_betweenness=params['edge betweenness'],
                        degree_centrality=params['degree centrality'],
                        local_clustering_coefficient=params['local clustering coefficient'],
                        pagerank=params['pagerank'],
                        eigenvector_centrality=params['eigenvector centrality'],
                        algebraic_distance=params['algebraic distance'],
                        diameter=params['diameter'],
                        density=params['density'],
                        preferential_attachment=params['preferential attachment'],
                        common_neighbor=params['common neighbor'],
                        katz_index=params['katz index'],
                        jaccard_index=params['jaccard index'],
                        adjusted_rand=params['adjusted rand'],
                        adamic_adar=params['adamic adar'],
                        local_degree_score=params['local degree score'],
                        local_similarity_score=params['local similarity score'],
                        scan=params['scan'],
                        verbose=False,
                    )
            print("Udało sie!!")
            print(extracted_data.shape)
            cache_features(
                extracted_data,
                dataset_name,
                params["degree sum"],
                params["shortest paths"],
                params["edge betweenness"],
                params["degree centrality"],
                False,
                False,
                params["pagerank"],
                params["eigenvector centrality"],
                params["algebraic distance"],
                params["diameter"],
                params["density"],
                params["preferential attachment"],
                params["common neighbor"],
                params["katz index"],
                params["jaccard index"],
                params["adjusted rand"],
                params["adamic adar"],
                params["local degree score"],
                params["local similarity score"],
                params["scan"]
            )
            params[feature_name] = False



if __name__ == "__main__":
    plots_dir = Path("plots") / "feature_importance"
    plots_dir.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    plots_dir = Path("plots") / "feature_importance"
    plots_dir.mkdir(parents=True, exist_ok=True)

    datasets = ['DD'] #['DD', 'NCI1', 'PROTEINS_full', 'ENZYMES', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']
    ldp_features = ['deg max', 'deg', 'deg min', 'deg mean', 'deg stddev']


    create_cached_features()


    for dataset_name in datasets:
        best_params = {
            'degree sum': False,
            'shortest paths': False,
            'edge betweenness': False,
            'degree centrality': False,
            'local clustering coefficient': False,
            'pagerank': False,
            'eigenvector centrality': False,
            'algebraic distance': False,
            'diameter': False,
            'density': False,
            'preferential attachment': False,
            'common neighbor': False,
            'katz index': False,
            'jaccard index': False,
            'adjusted rand': False,
            'adamic adar': False,
            'local degree score': False,
            'local similarity score': False,
            'scan': False,
        }

        best_acc = 0
        best_acc_std = 0

        start = time()

        with open(os.path.join('plots', 'feature_importance', f'{dataset_name}.pkl'), 'rb') as handle:
            b = pickle.load(handle)
            d = b.to_dict('records')[0]
            d = sorted(d.items(), key=lambda x: x[1], reverse=True)

            imp = [x for x in d if x[0] not in ldp_features]

            for i in range(len(imp)):
                params = best_params.copy()
                next_descriptor = imp[i][0]
                params[next_descriptor] = True

                acc_mean, acc_std = perform_experiment(
                    dataset_name=dataset_name,
                    verbose=False,
                    degree_sum=params['degree sum'],
                    shortest_paths=params['shortest paths'],
                    edge_betweenness=params['edge betweenness'],
                    degree_centrality=params['degree centrality'],
                    local_clustering_coefficient=params['local clustering coefficient'],
                    pagerank=params['pagerank'],
                    eigenvector_centrality=params['eigenvector centrality'],
                    algebraic_distance=params['algebraic distance'],
                    diameter=params['diameter'],
                    density=params['density'],
                    preferential_attachment=params['preferential attachment'],
                    common_neighbor=params['common neighbor'],
                    katz_index=params['katz index'],
                    jaccard_index=params['jaccard index'],
                    adjusted_rand=params['adjusted rand'],
                    adamic_adar=params['adamic adar'],
                    local_degree_score=params['local degree score'],
                    local_similarity_score=params['local similarity score'],
                    scan=params['scan'],
                    plots_dir=plots_dir
                )

                if acc_mean > best_acc:
                    best_acc = acc_mean
                    best_params = params
                    best_acc_std = acc_std

        total_time = round(time() - start, 2)

        best_params['time'] = total_time
        best_params['acc_mean'] = best_acc
        best_params['acc_std'] = best_acc_std

        os.makedirs('results', exist_ok=True)

        with open(os.path.join('results', f'{dataset_name}_best_features.pkl'), 'wb') as f:
            pickle.dump(best_params, f)

    # all_feature_importances = []
    #
    # for dataset_name in datasets:
    #     print(dataset_name)
    #     importances = perform_experiment_calculate_importance(
    #         dataset_name=dataset_name,
    #         verbose=False,
    #         degree_sum=True,
    #         shortest_paths=True,
    #         edge_betweenness=True,
    #         degree_centrality=True,
    #         local_clustering_coefficient=True,
    #         pagerank=True,
    #         eigenvector_centrality=True,
    #         algebraic_distance=True,
    #         diameter=True,
    #         density=True,
    #         preferential_attachment=True,
    #         common_neighbor=True,
    #         katz_index=True,
    #         jaccard_index=True,
    #         adjusted_rand=True,
    #         adamic_adar=True,
    #         local_degree_score=True,
    #         local_similarity_score=True,
    #         scan=True,
    #         plots_dir=plots_dir
    #     )
    #     all_feature_importances.append(importances)
    #
    # df = pd.concat(all_feature_importances, ignore_index=True)
    # df = pd.DataFrame(df.mean(axis=0)).transpose()
    # df.index = [""]
    # df.to_pickle(plots_dir / "DD.pkl")
    # df.plot.bar(rot=0)
    # plt.tight_layout()
    # plt.savefig(plots_dir / "DD.pdf")
