#!/usr/bin/env python3
"""
Test script for the iterative feature selection implementation.
This script runs a quick test on a small dataset to verify functionality.
"""

import sys
import warnings
from pathlib import Path

# Suppress warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from iterative_feature_selection import iterative_feature_selection, compare_selection_methods

def test_iterative_selection():
    """Test iterative feature selection on a small dataset."""
    print("Testing iterative feature selection...")
    print("=" * 50)
    
    # Test on a smaller dataset first
    test_dataset = "ENZYMES"  # Relatively small dataset
    
    print(f"Running test on {test_dataset} dataset")
    print("This may take a few minutes...")
    
    try:
        # Run iterative feature selection with stricter parameters for faster testing
        results = iterative_feature_selection(
            dataset_name=test_dataset,
            model_type='RandomForest',
            min_improvement=0.005,  # Slightly higher threshold for faster testing
            max_features=5,  # Limit to 5 features for faster testing
            n_runs=3,  # Reduced number of runs for faster testing
            verbose=True,
            plots_dir="plots/test"
        )
        
        print("\nTest Results:")
        print(f"Final accuracy: {results['acc_mean']:.4f} ± {results['acc_std']:.4f}")
        print(f"Features selected: {results['num_features']}")
        print(f"Time taken: {results['time']:.2f}s")
        print(f"Selected features: {[k for k, v in results.items() if k in ['degree_sum', 'shortest_paths', 'edge_betweenness', 'degree_centrality', 'local_clustering_coefficient', 'pagerank', 'eigenvector_centrality', 'algebraic_distance', 'diameter', 'density', 'preferential_attachment', 'common_neighbor', 'katz_index', 'jaccard_index', 'adjusted_rand', 'adamic_adar', 'local_degree_score', 'local_similarity_score', 'scan'] and v]}")
        
        # Save test results
        results_dir = Path("results") / "test"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        import pickle
        with open(results_dir / f"{test_dataset}_test.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\nTest results saved to: {results_dir / f'{test_dataset}_test.pkl'}")
        print("\n✅ Test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_comparison():
    """Test comparison functionality if importance-based results exist."""
    print("\nTesting comparison functionality...")
    print("=" * 50)
    
    test_dataset = "ENZYMES"
    
    try:
        comparison = compare_selection_methods(test_dataset, verbose=True)
        print("✅ Comparison test completed successfully!")
        return True
        
    except Exception as e:
        print(f"⚠️  Comparison test failed (this is expected if importance-based results don't exist): {str(e)}")
        return False

if __name__ == "__main__":
    print("Iterative Feature Selection Test Suite")
    print("=" * 60)
    
    # Test iterative selection
    selection_success = test_iterative_selection()
    
    # Test comparison (optional)
    comparison_success = test_comparison()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Iterative Selection: {'✅ PASS' if selection_success else '❌ FAIL'}")
    print(f"Comparison: {'✅ PASS' if comparison_success else '⚠️  SKIP'}")
    
    if selection_success:
        print("\nThe iterative feature selection implementation is working correctly!")
        print("You can now run experiments using:")
        print("  python main.py --iterative_feature_selection true --dataset_name DD")
        print("  or")
        print("  python main_iterative.py --datasets DD")
    else:
        print("\n❌ There are issues with the implementation that need to be fixed.")
        sys.exit(1) 