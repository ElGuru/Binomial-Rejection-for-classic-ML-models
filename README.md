# Binomial-Rejection-for-classic-ML-models
Machine Learning Classifier Rejection System
A comprehensive Python library for implementing prediction rejection strategies in machine learning classifiers. This system allows models to abstain from making predictions when confidence is low, improving overall reliability and performance.
Overview
This library implements rejection-based classification for three popular machine learning algorithms:

    Random Forest (RF)
    K-Nearest Neighbors (KNN)
    Logistic Regression

The core concept is to identify and reject predictions with low confidence scores, thereby improving accuracy on the remaining accepted predictions.
Features

    Probability Vector Generation: Extract prediction confidence scores from trained classifiers
    Adaptive Thresholding: Compute optimal rejection thresholds using binomial statistical tests
    Performance Evaluation: Comprehensive metrics including coverage, selected accuracy, and rejection effectiveness
    Multi-Classifier Support: Unified interface for different classifier types

Installation

# Required dependencies
pip install numpy pandas scipy scikit-learn

Core Functions
1. Probability Vector Extraction
rf_vectors(X_val, classifier)
Extracts class probability vectors from a trained Random Forest classifier.
Parameters:

    X_val: Validation data (DataFrame or array-like)
    classifier: Trained RandomForestClassifier

Returns:

    DataFrame with probability vectors (columns=classes, rows=samples)

knn_vectors(X_val, classifier, return_mapping=False)
Transforms validation data into prediction vectors based on neighbor class counts.
Parameters:

    X_val: Validation features
    classifier: Fitted K-NN classifier
    return_mapping: If True, returns class-to-number mapping

Returns:

    DataFrame with neighbor class counts (normalized by k for probabilities)

log_regression_vectors(X_val, classifier)
Extracts prediction probabilities from logistic regression classifier.
Parameters:

    X_val: Validation features
    classifier: Trained LogisticRegression classifier

Returns:

    DataFrame with prediction probability vectors

2. Prediction Analysis
sorted_rf_numeric(X, y, classifier)
sorted_knn_numeric(X, y, classifier)
sorted_logreg_numeric(X, y, classifier)
Generate sorted DataFrames with prediction analysis for each classifier type.
Parameters:

    X: Input features
    y: True labels
    classifier: Trained classifier

Returns:

    DataFrame with columns:
        max_prob: Maximum class probability
        predicted: Predicted class (numeric index)
        true: True class (numeric index)

3. Threshold Computation
compute_thresholds(X_val, y_val, delta, classifier) (KNN)
compute_thresholds_lr(X_val, y_val, delta, classifier) (Logistic Regression)
compute_thresholds_rf(X_val, y_val, delta, classifier) (Random Forest)
Compute optimal rejection thresholds using binomial statistical criterion.
Parameters:

    X_val: Validation features
    y_val: True validation labels
    delta: Significance level for binomial test (0-1)
    classifier: Trained classifier

Returns:

    List of optimal thresholds (one per class)

Algorithm:

    For each class, identify misclassified instances
    Test different threshold values
    Use binomial test to validate rejection effectiveness
    Select threshold that maximizes selected accuracy while maintaining statistical significance

4. Rejection Strategy
reject_predictions(X, thresh)
Identifies predictions to reject based on class-specific thresholds.
Parameters:

    X: DataFrame with 'max_prob' and 'predicted' columns
    thresh: Threshold vector (index corresponds to predicted classes)

Returns:

    Binary rejection vector (1=rejected, 0=accepted)

5. Performance Evaluation
calculate_metrics(y_true, y_pred, reject_vector)
Computes comprehensive performance metrics.
Returns:

    coverage: Proportion of predictions accepted (1 - rejection_rate)
    select_accuracy: Accuracy on accepted predictions
    reject_accuracy: Proportion of errors among rejected predictions

evaluate_knn_performance(X_test, y_test, classifier, thresholds_list=None)
evaluate_logreg_performance(X_test, y_test, classifier, thresholds_list=None)
evaluate_rf_performance(X_test, y_test, classifier)
Comprehensive evaluation functions for each classifier type.
Usage Examples
Basic Usage

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from funciones_final import *

# Load your data
X, y = load_your_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Train classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Compute rejection thresholds
delta = 0.05  # Significance level
thresholds = compute_thresholds_rf(X_val, y_val, delta, rf_classifier)

# Evaluate performance
results = evaluate_rf_performance(X_test, y_test, rf_classifier)

Advanced Workflow

# 1. Extract probability vectors
prob_vectors = rf_vectors(X_val, rf_classifier)

# 2. Get sorted predictions with confidence scores
sorted_preds = sorted_rf_numeric(X_test, y_test, rf_classifier)

# 3. Apply rejection strategy
rejected = reject_predictions(sorted_preds, thresholds)

# 4. Calculate detailed metrics
coverage, sel_acc, rej_acc = calculate_metrics(
    sorted_preds["true"], 
    sorted_preds["predicted"], 
    rejected
)

print(f"Coverage: {coverage:.4f}")
print(f"Selected Accuracy: {sel_acc:.4f}")
print(f"Rejection Accuracy: {rej_acc:.4f}")

Key Concepts
Rejection Strategy
The system implements a confidence-based rejection approach:

    Train classifier on training data
    Compute class-specific rejection thresholds using validation data
    Reject test predictions below their respective thresholds
    Evaluate performance on accepted vs. rejected predictions

Statistical Foundation

    Uses binomial tests to ensure rejection effectiveness
    Threshold selection balances coverage and accuracy
    Statistical significance level (delta) controls trade-off

Performance Metrics

    Coverage: Proportion of predictions accepted
    Selected Accuracy: Accuracy on non-rejected predictions
    Rejection Accuracy: Quality of rejection decisions (higher = better rejection of errors)

File Structure

funciones_final.py
├── Probability Extraction Functions
│   ├── rf_vectors()
│   ├── knn_vectors()
│   └── log_regression_vectors()
├── Prediction Analysis Functions
│   ├── sorted_rf_numeric()
│   ├── sorted_knn_numeric()
│   └── sorted_logreg_numeric()
├── Threshold Computation Functions
│   ├── compute_thresholds() [KNN]
│   ├── compute_thresholds_lr()
│   └── compute_thresholds_rf()
├── Rejection Strategy Functions
│   ├── reject_predictions()
│   └── calculate_metrics()
└── Evaluation Functions
    ├── evaluate_knn_performance()
    ├── evaluate_logreg_performance()
    └── evaluate_rf_performance()

Dependencies

    numpy: Numerical computations
    pandas: Data manipulation and analysis
    scipy.stats: Statistical functions (binomial tests)
    scikit-learn: Machine learning classifiers (optional, for usage examples)

Important Notes
KNN Specific Requirements

    KNN classifier must store training labels in _y attribute
    Handles class mapping automatically
    Normalizes neighbor counts by k-value for probability estimation

Threshold Computation

    Uses validation set to compute thresholds (avoid overfitting)
    Binomial test ensures statistical significance of rejection regions
    Per-class thresholds allow fine-tuned rejection strategies

Error Handling

    Validates classifier attributes and data consistency
    Handles missing class mappings
    Provides informative error messages for debugging

Contributing

    Fork the repository
    Create a feature branch
    Add tests for new functionality
    Ensure all tests pass
    Submit a pull request

License
This project is open source. Please check the repository for specific license terms.
Citation
If you use this code in your research, please cite:

Sarmiento Amdré and Analuisa Daniela, "Machine Learning Classifier Rejection System", GitHub Repository, 2025

