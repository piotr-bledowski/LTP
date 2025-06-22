# Single-Split Feature Selection Modification

## Overview

This document describes the modifications made to the importance-based feature selection method in `main.py` and `perform_experiment.py` to support **single-split sequential feature selection**, as requested.

## Problem Statement

The original importance-based feature selection method had the following workflow:

1. Calculate feature importance by averaging across **all dataset splits**
2. Use this averaged importance to rank features
3. Perform greedy feature selection by evaluating each feature on **all splits**

The user requested a modification to evaluate the whole algorithm on all dataset splits in sequence and provide mean and std of accuracy across these splits per dataset in a convenient CSV format.

## Solution

### New Functions Added

#### `perform_experiment_calculate_importance_single_split()` in `perform_experiment.py`

- Calculates feature importance using only a single specified dataset split
- Takes a `split_idx` parameter to specify which split to use
- Returns importance scores from that split only (no averaging)

#### `perform_experiment_single_split()` in `perform_experiment.py`

- Evaluates a feature configuration on a single specified dataset split
- Returns accuracy for that split only
- Used during the greedy selection process

### Modified Main Logic

The `main.py` file now implements a **sequential approach** when `--single_split_selection=True`:

1. **For each dataset:**

   - Load all available splits
   - **For each split:**
     - Calculate feature importance on that split only
     - Perform greedy feature selection on that split only
     - Record best accuracy, selected features, and execution time
   - **Aggregate results** across all splits for that dataset
   - Calculate mean ± std accuracy, feature selection frequencies, timing statistics

2. **Generate comprehensive CSV** with all metrics across all datasets

### New Command Line Arguments

#### `--single_split_selection` (boolean, default: False)

- Enables the new single-split sequential mode
- When `True`: Runs algorithm on ALL splits sequentially, aggregates results
- When `False`: Uses original multi-split averaging approach

#### `--split_idx` (integer, deprecated)

- No longer used in the current implementation
- Kept for backward compatibility but has no effect

## Usage Examples

### Single-Split Sequential Mode (NEW)

```bash
# Run on single dataset - processes ALL splits automatically
python main.py --dataset_name ENZYMES --single_split_selection true --model_type RandomForest

# Run on all datasets - processes ALL splits for each dataset
python main.py --dataset_name all --single_split_selection true --model_type RandomForest

# With verbose output to see per-split progress
python main.py --dataset_name DD --single_split_selection true --model_type RandomForest --verbose true
```

### Multi-Split Averaging Mode (ORIGINAL)

```bash
# Traditional approach (requires pre-computed importance files)
python main.py --dataset_name ENZYMES --single_split_selection false --model_type RandomForest
```

## Output Format

### CSV Results (`single_split_results_{model}.csv`)

The main output is a CSV file with the following columns:

| Column                          | Description                                      |
| ------------------------------- | ------------------------------------------------ |
| `Dataset`                       | Dataset name                                     |
| `Accuracy_Mean`                 | Mean accuracy across all split runs              |
| `Accuracy_Std`                  | Standard deviation of accuracy across splits     |
| `Baseline_Accuracy`             | LDP-only baseline performance                    |
| `Total_Improvement`             | Improvement over baseline                        |
| `Time_Mean_Seconds`             | Average execution time per split                 |
| `Time_Std_Seconds`              | Standard deviation of execution times            |
| `Total_Time_Seconds`            | Total time for all splits                        |
| `Num_Splits_Processed`          | Number of splits successfully processed          |
| `Selected_Features_Frequencies` | Summary of feature selection frequencies         |
| `Model_Type`                    | Classification model used                        |
| `Feature_{feature_name}`        | Individual feature selection frequency (0.0-1.0) |

### Example CSV Output

```csv
Dataset,Accuracy_Mean,Accuracy_Std,Baseline_Accuracy,Total_Improvement,Time_Mean_Seconds,Time_Std_Seconds,Total_Time_Seconds,Num_Splits_Processed,Selected_Features_Frequencies,Model_Type,Feature_degree_sum,Feature_shortest_paths,...
ENZYMES,0.485,0.061,0.365,0.120,62.2,5.4,621.9,10,shortest_paths(0.80); jaccard_index(0.60),RandomForest,0.0,0.8,...
```

## Key Differences

1. **Feature Importance Calculation**: Pre-computed importance from `plots/feature_importance/{dataset}.pkl` is used across all splits, ensuring consistency

2. **Evaluation Strategy**: Feature evaluation is performed on one split at a time, not averaged across all splits

3. **Statistical Robustness**: Results from all splits are aggregated to provide mean ± std statistics

4. **Efficiency**: Significantly faster than multi-split averaging, and uses consistent pre-computed importance rankings

## Workflow Comparison

### Original Multi-Split Approach

1. Load pre-computed feature importance (averaged across splits)
2. Rank features by averaged importance
3. For each feature in ranking:
   - Evaluate feature set on **all splits**
   - Calculate mean accuracy across splits
   - Keep if best so far
4. Output single result per dataset

### New Single-Split Sequential Approach

1. **For each split in dataset:**

   - Calculate feature importance on **this split only**
   - Rank features by single-split importance
   - For each feature in ranking:
     - Evaluate feature set on **same single split**
     - Keep if accuracy on that split is best so far
   - Record: best accuracy, selected features, execution time

2. **Aggregate across all splits:**

   - Calculate mean ± std accuracy
   - Calculate feature selection frequencies
   - Calculate timing statistics
   - Generate comprehensive CSV row

3. **Output aggregated results** for all datasets in CSV format

## Benefits

1. **Robustness Evaluation**: Tests algorithm consistency across different data splits
2. **Statistical Rigor**: Provides mean ± std accuracy instead of single values
3. **Feature Stability**: Shows how consistently features are selected
4. **Comprehensive Metrics**: Includes timing, improvement, and selection frequencies
5. **Self-Contained**: No dependency on pre-computed files
6. **CSV Output**: Easy to analyze and compare results
7. **Scalable**: Works with any number of datasets and splits

## Use Cases

- **Algorithm Validation**: Assess how robust the feature selection is
- **Comparative Studies**: Compare performance across different datasets with statistical measures
- **Feature Analysis**: Understand which features are consistently selected
- **Performance Benchmarking**: Get comprehensive metrics including timing and improvements

## Testing

Run the updated test script:

```bash
python test_single_split_selection.py
```

This will demonstrate the new functionality and show the expected CSV output format.
