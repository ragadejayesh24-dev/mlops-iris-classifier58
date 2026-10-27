# Practical 7: Hyperparameter Tuning and Model Comparison

## Aim
To build a baseline classification model, perform Grid Search and Random Search hyperparameter tuning, track experiments using MLflow, and compare model performance.

## Dataset
Iris dataset stored in `data/processed/iris_features.csv`, containing 149 rows and 9 columns. Seven numerical features were used for training, and `species` was used as the target variable.

## Methodology
1. Developed a baseline Decision Tree classifier.
2. Applied Grid Search to a Random Forest classifier using 72 parameter combinations and 5-fold cross-validation, resulting in 360 model fits.
3. Applied Random Search to a Random Forest classifier using 30 configurations and 5-fold cross-validation, resulting in 150 model fits.
4. Recorded experiment parameters and evaluation metrics using MLflow.

## Results

| Metric | Baseline | Grid Search | Random Search |
|---|---:|---:|---:|
| Cross-validation Macro F1 | 0.9663 | 0.9663 | 0.9663 |
| Test accuracy | 90.00% | 93.33% | 96.67% |
| Test Macro F1 | Not recorded | 0.9333 | 0.9666 |

## Best Hyperparameters

**Grid Search**
- `n_estimators`: 50
- `max_depth`: None
- `max_features`: sqrt
- `min_samples_leaf`: 1
- `min_samples_split`: 2

**Random Search**
- `n_estimators`: 93
- `max_depth`: 5
- `max_features`: log2
- `min_samples_leaf`: 1
- `min_samples_split`: 3

## Conclusion
All three experiments completed successfully. Random Search achieved the highest test accuracy of 96.67% and a test Macro F1 score of 0.9666 among the evaluated models. Grid Search achieved 93.33% test accuracy, while the baseline Decision Tree achieved 90.00%.

MLflow was used to record the experiment parameters and evaluation metrics for comparison.
