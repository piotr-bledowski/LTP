import os
import pickle
import sys
import warnings
from pathlib import Path
from time import time
from data_loading import load_dataset
from feature_extraction import calculate_features_matrix, extract_features
from models import get_model
from perform_experiment import perform_experiment, perform_experiment_calculate_importance
import numpy as np


if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


def perform_recursive_feature_elimination(dataset_name: str, plots_dir: Path):
    """Performs Recursive Feature Elimination to identify the best features."""
    
    start = time()

    # Initial feature set (all features enabled)
    initial_params = {
        'degree_sum': True,
        'shortest_paths': True,
        'edge_betweenness': True,
        'degree_centrality': True,
        'local_clustering_coefficient': True,
        'pagerank': True,
        'eigenvector_centrality': True,
        'algebraic_distance': True,
        'diameter': True,
        'density': True,
        'preferential_attachment': True,
        'common_neighbor': True,
        'katz_index': True,
        'jaccard_index': True,
        'adjusted_rand': True,
        'adamic_adar': True,
        'local_degree_score': True,
        'local_similarity_score': True,
        'scan': True,
    }

    params = initial_params.copy()
    
    with open(os.path.join('plots', 'feature_importance', f'{dataset_name}.pkl'), 'rb') as handle:
        importance_df = pickle.load(handle)
        d = importance_df.to_dict('records')[0]
    
    ldp_features = ['deg max', 'deg', 'deg min', 'deg mean', 'deg stddev']
    importances_dict = {k: v for k, v in d.items() if k not in ldp_features}
    
    # Get features matrix and labels (needed for final accuracy evaluation)
    features = extract_features(
        load_dataset(dataset_name),
        **initial_params
    )
    
    # Custom RFE implementation using pre-calculated importances
    n_features = len(initial_params)
    n_features_to_select = 5
    remaining_features = list(initial_params.keys())
    ranking = {feature: -1 for feature in initial_params.keys()}  # Initialize ranking dictionary
    current_rank = 1
    
    while len(remaining_features) > n_features_to_select:

        importance_df = perform_experiment_calculate_importance(
            dataset_name,
            **params
        )

        importances_dict = importance_df.to_dict('records')[0]

        # Find feature with minimum importance directly from dictionary
        feature_to_remove = min(
            remaining_features,
            key=lambda f: importances_dict[f.replace('_', ' ')]
        )
        
        # Update ranking
        ranking[feature_to_remove] = current_rank
        current_rank += 1
        
        # Remove feature
        remaining_features.remove(feature_to_remove)
        params[feature_to_remove] = False
    
    # Assign top rank to remaining features
    for feature in remaining_features:
        ranking[feature] = 1
    
    # Feature ranks dictionary is now just our ranking
    feature_ranks = ranking
    
    # Sort features by ranking
    sorted_features = sorted(feature_ranks.items(), key=lambda x: x[1])
    
    # Rest of the function remains the same...
    results = {
        'feature_ranks': feature_ranks,
        'selected_features': [f for f, r in sorted_features if r == 1],
        'feature_order': [f for f, _ in sorted_features]
    }
    
    selected_features_params = {feature: True for feature in results['selected_features']}
    selected_features_params.update({feature: False for feature in initial_params.keys() 
                                   if feature not in results['selected_features']})
    
    acc_mean, acc_std = perform_experiment(
        dataset_name,
        **selected_features_params,
    )
    
    results.update({
        'accuracy_mean': acc_mean,
        'accuracy_std': acc_std,
        'time': time() - start
    })
    
    with open(os.path.join('results', f'{dataset_name}_rfe_results_fixed.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    return results


if __name__ == "__main__":
    plots_dir = Path("plots") / "feature_importance"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path("plots") / "feature_importance"
    plots_dir.mkdir(parents=True, exist_ok=True)

    datasets = ['NCI1', 'PROTEINS_full', 'ENZYMES', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']
    ldp_features = ['deg max', 'deg', 'deg min', 'deg mean', 'deg stddev']
    
    # Run RFE for each dataset
    for dataset_name in datasets:
        print(f"Running RFE for {dataset_name}")
        rfe_results = perform_recursive_feature_elimination(dataset_name, plots_dir)
        print(f"Selected features: {rfe_results['selected_features']}")
        print(f"Feature order: {rfe_results['feature_order']}")
        print(f"Accuracy: {rfe_results['accuracy_mean']} ± {rfe_results['accuracy_std']}")
        print(f"Time: {rfe_results['time']} seconds")
        
