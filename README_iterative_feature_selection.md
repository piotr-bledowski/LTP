# Iterative Feature Selection for Local Topological Profile

This implementation provides an alternative to the existing importance-based feature selection method. Instead of using precalculated feature importances, it performs **forward feature selection** starting from no topological features and iteratively adding the feature that provides the most improvement.

## Overview

### Traditional Approach (existing)

1. Calculate feature importances using all features
2. Sort features by importance
3. Iteratively add features in order of importance
4. Stop when adding a feature doesn't improve performance

### New Iterative Approach

1. Start with only base LDP features (no topological features)
2. Evaluate each remaining topological feature individually
3. Add the feature that provides the best improvement
4. Repeat until no feature provides sufficient improvement
5. Stop when improvement threshold is not met

## Key Differences

- **No pre-calculated importances**: Features are evaluated dynamically
- **True forward selection**: Each feature is tested in the context of currently selected features
- **Greedy optimization**: Always selects the feature that provides the best immediate improvement
- **Multiple runs**: The entire selection process is run multiple times (default: 10) to account for variability
- **Robust statistics**: Mean and standard deviation calculated across multiple complete runs
- **Configurable stopping criteria**: Can set minimum improvement threshold and maximum features

## Files Added

### Core Implementation

- `iterative_feature_selection.py` - Main implementation
- `main_iterative.py` - Standalone script for running iterative selection
- `test_iterative_selection.py` - Test suite

### Updated Files

- `main.py` - Added `--iterative_feature_selection` option

## Usage

### Option 1: Using the updated main.py

```bash
# Run iterative feature selection on DD dataset
python main.py --iterative_feature_selection true --dataset_name DD

# Run with custom parameters
python main.py --iterative_feature_selection true --dataset_name DD \
    --min_improvement 0.002 --max_features_iterative 10 --model_type RandomForest
```

### Option 2: Using the dedicated script

```bash
# Run on single dataset
python main_iterative.py --datasets DD

# Run on multiple datasets
python main_iterative.py --datasets DD NCI1 PROTEINS_full

# Run on all datasets
python main_iterative.py --datasets all

# Custom parameters
python main_iterative.py --datasets DD --model_type RandomForest \
    --min_improvement 0.001 --max_features 15 --verbose
```

### Option 3: Programmatic usage

```python
from iterative_feature_selection import iterative_feature_selection

results = iterative_feature_selection(
    dataset_name="DD",
    model_type="RandomForest",
    min_improvement=0.001,
    max_features=10,
    verbose=True
)
```

## Parameters

- `dataset_name`: Name of the dataset to process
- `model_type`: Classification model ("RandomForest", "LinearSVM", "KernelSVM")
- `min_improvement`: Minimum accuracy improvement to continue (default: 0.001)
- `max_features`: Maximum number of topological features to select (default: None)
- `n_runs`: Number of times to run the entire selection process (default: 10)
- `verbose`: Print detailed progress information

## Available Topological Features

The algorithm can select from 19 topological features:

1. degree_sum
2. shortest_paths
3. edge_betweenness
4. degree_centrality
5. local_clustering_coefficient
6. pagerank
7. eigenvector_centrality
8. algebraic_distance
9. diameter
10. density
11. preferential_attachment
12. common_neighbor
13. katz_index
14. jaccard_index
15. adjusted_rand
16. adamic_adar
17. local_degree_score
18. local_similarity_score
19. scan

Note: Base LDP features (degree distributions) are always included and not subject to selection.

## Output and Results

### Results Structure

```python
{
    # Feature selection flags
    'degree_sum': True/False,
    'shortest_paths': True/False,
    # ... for all topological features

    # Performance metrics
    'acc_mean': 0.8234,
    'acc_std': 0.0156,
    'baseline_acc': 0.7890,
    'total_improvement': 0.0344,

    # Selection metadata
    'num_features': 3,
    'iterations': 3,
    'time': 234.56,

    # Detailed history
    'selection_history': [
        {'features': [], 'accuracy': 0.7890, 'improvement': 0.0},
        {'features': ['edge_betweenness'], 'accuracy': 0.8012, 'improvement': 0.0122},
        # ...
    ]
}
```

### File Locations

- Results: `results/iterative_feature_selection/`
- Summary files: `results/iterative_feature_selection/iterative_selection_summary_*.txt` and `.csv`
- Comparisons: `results/comparisons/`
- Plots: `plots/iterative_feature_selection/`

### Summary Files

After running experiments on multiple datasets, the system automatically generates comprehensive summary files:

**Text Summary (`iterative_selection_summary_randomforest.txt`)**:

- Overall statistics across all datasets
- Detailed results per dataset (accuracy, time, features selected)
- Feature selection frequency analysis
- Features never selected

**CSV Summary (`iterative_selection_summary_randomforest.csv`)**:

- Machine-readable format for analysis
- One row per dataset with all metrics
- Individual columns for each topological feature (True/False)
- Easy to import into spreadsheet applications or analysis tools

**Example summary content**:

```
ITERATIVE FEATURE SELECTION SUMMARY
==================================================
Model Type: RandomForest
Number of Datasets: 3
Generated: 2024-01-15 14:30:22

OVERALL STATISTICS
--------------------
Average Accuracy: 0.7845
Average Time: 156.32 seconds
Average Features Selected: 3.7
Average Improvement: 0.0234

DATASET RESULTS
--------------------

Dataset: DD
  Accuracy: 0.7823 ± 0.0234
  Baseline: 0.7589
  Improvement: +0.0234
  Time: 234.56 seconds
  Features Selected (3): edge_betweenness, jaccard_index, pagerank
  Iterations: 3

Dataset: NCI1
  Accuracy: 0.8234 ± 0.0156
  Baseline: 0.8012
  Improvement: +0.0222
  Time: 189.45 seconds
  Features Selected (4): degree_centrality, edge_betweenness, local_clustering_coefficient, pagerank
  Iterations: 4

FEATURE SELECTION FREQUENCY
------------------------------
  edge_betweenness: 3/3 datasets (100.0%)
  pagerank: 2/3 datasets (66.7%)
  jaccard_index: 1/3 datasets (33.3%)

  Features never selected: diameter, density, scan
```

## Testing

Run the test suite to verify the implementation:

```bash
python test_iterative_selection.py
```

This will:

1. Test iterative selection on the ENZYMES dataset
2. Test comparison functionality (if importance-based results exist)
3. Save test results and provide usage examples

## Comparison with Importance-Based Method

Use the comparison functionality to analyze differences:

```bash
# Compare methods for specific dataset
python main_iterative.py --datasets DD --mode compare

# Run both selection and comparison
python main_iterative.py --datasets DD --mode both
```

The comparison will show:

- Accuracy differences between methods
- Number of features selected by each method
- Time taken by each approach
- Selected feature sets

## Performance Considerations

### Time Complexity

- **Iterative method**: O(n²) where n is number of features
- **Importance-based**: O(n) after initial importance calculation

### Memory Usage

- Uses existing feature caching system efficiently
- Loads only necessary feature combinations
- Minimal additional memory overhead

### Scalability

- Works well for datasets with cached features
- Time increases quadratically with number of available features
- Can be limited with `max_features` parameter for faster execution

## Algorithm Details

### Multiple Run Strategy

The algorithm runs the entire feature selection process multiple times (default: 10 runs) and calculates statistics across these runs:

1. **Per Run**: Each run independently performs forward selection from scratch
2. **Final Accuracy**: Each run produces a final accuracy based on the selected feature set
3. **Statistics**: Mean and standard deviation calculated across all run final accuracies
4. **Feature Selection**: Final feature set determined by consensus (features selected in >50% of runs) or best performing run

This approach provides more robust estimates of performance and accounts for variability in the selection process itself.

### Forward Selection Process (Per Run)

1. **Initialization**: Start with empty feature set
2. **Evaluation**: For each remaining feature:
   - Create feature set = current_features + candidate_feature
   - Run cross-validation experiment
   - Record accuracy
3. **Selection**: Choose feature with best accuracy improvement
4. **Stopping**: Continue until improvement < threshold or max features reached

### Stopping Criteria

- **Improvement threshold**: No feature improves accuracy by at least `min_improvement`
- **Feature limit**: Reached `max_features` topological features
- **No remaining features**: All features have been considered

### Feature Integration

- Uses existing `perform_experiment()` function for consistent evaluation
- Leverages cached feature system for efficiency
- Maintains compatibility with all model types and parameters

## Examples

### Quick Test

```bash
# Fast test with limited features and fewer runs
python main_iterative.py --datasets ENZYMES --max_features 3 --n_runs 3
```

### Production Run

```bash
# Full run on important datasets with default 10 runs
python main_iterative.py --datasets DD NCI1 PROTEINS_full --model_type RandomForest

# Custom number of runs
python main_iterative.py --datasets DD --n_runs 15 --min_improvement 0.002
```

### Comparison Study

```bash
# Compare both methods with custom parameters
python main_iterative.py --datasets all --mode both --n_runs 10
```

### Using main.py integration

```bash
# Run with custom parameters
python main.py --iterative_feature_selection true --dataset_name DD --n_runs 5
```

This implementation provides a robust alternative to importance-based feature selection while maintaining compatibility with the existing codebase and leveraging all existing optimizations.
