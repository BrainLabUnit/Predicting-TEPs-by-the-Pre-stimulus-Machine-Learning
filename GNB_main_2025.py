#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 11:41:02 2025

@author: sadafmoaveninejad
"""

# Create feature dataframe
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import rankdata, spearmanr
from GNB_functions_2025 import compute_targets, visualize_quartiles_comparison_fixed_ylim, interpret_model_with_importance
#%% Initializing 

CONFIG = {
    "channel_index": 5,  # C3 (5th in 0-based indexing)
    "target_method": "peak",#"area",
    'random_state': 42,  # Random seed for reproducibility
    'test_size': 0.2,  # Train-test split ratio
    'features_type': "allbands" # "Alpha"#
   
}
#------------------
# Define the data directory
data_directory = os.path.join(os.getcwd(), 'RogashData')

# Load the NumPy arrays
alpha = np.load(os.path.join(data_directory, "alpha.npy"))  # Shape (1560, 1)
beta = np.load(os.path.join(data_directory, "beta.npy"))    # Shape (1560, 1)
delta = np.load(os.path.join(data_directory, "delta.npy"))  # Shape (1560, 1)
gamma = np.load(os.path.join(data_directory, "gamma.npy"))  # Shape (1560, 1)
theta = np.load(os.path.join(data_directory, "theta.npy"))  # Shape (1560, 1)
pre_stimulus = np.load(os.path.join(data_directory, "pre_stimulus_c3.npy"))  # Shape (1000,1560)
post_stimulus = np.load(os.path.join(data_directory, "post_stimulus_c3.npy"))  # Shape (1000,1560)


if CONFIG['features_type'] == "allbands":
    # Feature names
    feature_labels = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]#, "HFD"]

    # Stack the arrays together (Now all have shape (1560, 1))
    features_to_combine = np.hstack([delta, theta, alpha, beta, gamma])#, HFD])  # Shape (1560, 6)

    # Convert to Pandas DataFrame
    feature_df = pd.DataFrame(features_to_combine, columns=feature_labels)
elif CONFIG['features_type'] == "alpha":
    # Feature names
    feature_labels = ["Alpha"]

    # Stack the arrays together (Now all have shape (1560, 1))
    features_to_combine = np.hstack([alpha])#, HFD])  # Shape (1560, 6)

    # Convert to Pandas DataFrame
    feature_df = pd.DataFrame(features_to_combine, columns=feature_labels)
    

# --- Load EEG data ---
# data = np.load(os.path.join(data_directory, "EEG_Measurements.npy"))

# Load Excel
excel_path = os.path.join(data_directory, "TrialsVsSubjects.xlsx")
trial_subject_df = pd.read_excel(excel_path, header=None)
trial_subject_df.columns = ['trial_id', 'subject_id']
assert trial_subject_df.shape[0] == feature_df.shape[0]
subject_ids = trial_subject_df['subject_id'].values

# Extract targets
# channel_id = CONFIG['channel_index'] - 1
# post_stimulus = data[channel_id, 1000:, :]
targets = compute_targets(post_stimulus, method=CONFIG['target_method'] )


#%% --- Subject-level Stratified K-Fold with Hyperparameters Tunning ---

subject_ids = trial_subject_df['subject_id'].values
unique_subjects = np.unique(subject_ids)
subject_df = pd.DataFrame({'subject_id': unique_subjects})
subject_df['stratify_label'] = 0  # dummy label


# Store results
all_spearman_corrs = []
all_nmses = []
all_spearman_cis = []
all_nmse_cis = []
all_predicted_ranks = []
all_true_ranks = []
all_test_indices = []

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# ---- Start SKF:

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG['random_state'])

for fold, (train_subj_idx, test_subj_idx) in enumerate(skf.split(subject_df['subject_id'], subject_df['stratify_label'])):
    print(f"\n=== Fold {fold + 1} ===")

    # Get subject IDs for this fold
    train_subjects = subject_df.loc[train_subj_idx, 'subject_id'].values
    test_subjects = subject_df.loc[test_subj_idx, 'subject_id'].values

    # Map to trial indices
    trial_subject_ids = trial_subject_df['subject_id'].values
    train_mask = np.isin(trial_subject_ids, train_subjects)
    test_mask = np.isin(trial_subject_ids, test_subjects)

    # Extract data
    X_train = feature_df[train_mask]
    X_test = feature_df[test_mask]
    y_train = targets[train_mask]
    y_test = targets[test_mask]
    test_indices = np.where(test_mask)[0]

    # --- Hyperparameter tuning ---
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=CONFIG['random_state']),
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # --- Predict and evaluate ---
    predicted_targets = model.predict(X_test)
    predicted_ranks = rankdata(predicted_targets, method="ordinal")
    true_ranks = rankdata(y_test, method="ordinal")

    # Save ranks across folds
    all_predicted_ranks.extend(predicted_ranks)
    all_true_ranks.extend(true_ranks)
    all_test_indices.extend(test_indices)

    # Spearman correlation
    spearman_corr, p_value = spearmanr(true_ranks, predicted_ranks)
    print(f"Spearman's ρ: {spearman_corr:.4f}, p = {p_value:.2e}")
    all_spearman_corrs.append(spearman_corr)

    # nMSE
    mse = mean_squared_error(y_test, predicted_targets)
    nmse = mse / np.var(y_test)
    print(f"nMSE: {nmse:.4f}")
    all_nmses.append(nmse)

    # --- Confidence intervals ---
    # Bootstrapped nMSE CI
    nmse_boot = []
    for _ in range(1000):
        idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
        mse_sample = mean_squared_error(y_test[idx], predicted_targets[idx])
        nmse_boot.append(mse_sample / np.var(y_test[idx]))
    nmse_ci = np.percentile(nmse_boot, [2.5, 97.5])
    all_nmse_cis.append(tuple(nmse_ci))
    print(f"nMSE 95% CI: [{nmse_ci[0]:.4f}, {nmse_ci[1]:.4f}]")

    # Bootstrapped Spearman CI
    spearman_boot = []
    for _ in range(1000):
        idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
        r, _ = spearmanr(true_ranks[idx], predicted_ranks[idx])
        spearman_boot.append(r)
    spearman_ci = np.percentile(spearman_boot, [2.5, 97.5])
    all_spearman_cis.append(tuple(spearman_ci))
    print(f"Spearman 95% CI: [{spearman_ci[0]:.4f}, {spearman_ci[1]:.4f}]")


# === Global-level Ture-Predicted Over All Folds ===
all_true_ranks = np.array(rankdata(all_true_ranks, method="ordinal"))
all_predicted_ranks = np.array(rankdata(all_predicted_ranks, method="ordinal"))

#%% --- Results - Part 1: Spearmanr  ---
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
import numpy as np


# --- Spearmanr Global Level ---
spearman_final, p_spearman = spearmanr(all_true_ranks, all_predicted_ranks)

# Bootstrap Spearman CI
spearman_bootstrap = []
for _ in range(1000):
    idx = np.random.choice(len(all_true_ranks), size=len(all_true_ranks), replace=True)
    r, _ = spearmanr(all_true_ranks[idx], all_predicted_ranks[idx])
    spearman_bootstrap.append(r)

spearman_ci_lower = np.percentile(spearman_bootstrap, 2.5)
spearman_ci_upper = np.percentile(spearman_bootstrap, 97.5)

#  --- nMSE Global Level ---
global_mse = mean_squared_error(all_true_ranks, all_predicted_ranks)
global_nmse = global_mse / np.var(all_true_ranks)

# Bootstrap nMSE CI
nmse_bootstrap = []
for _ in range(1000):
    idx = np.random.choice(len(all_true_ranks), size=len(all_true_ranks), replace=True)
    mse_sample = mean_squared_error(all_true_ranks[idx], all_predicted_ranks[idx])
    nmse_sample = mse_sample / np.var(all_true_ranks[idx])
    nmse_bootstrap.append(nmse_sample)

nmse_ci_lower = np.percentile(nmse_bootstrap, 2.5)
nmse_ci_upper = np.percentile(nmse_bootstrap, 97.5)

# --- Final Output ---
print("\n===== Evaluation Over All Test Sets Combined =====")
print(f"Spearman's Correlation (ρ): {spearman_final:.3f}")
print(f"95% CI: [{spearman_ci_lower:.3f}, {spearman_ci_upper:.3f}], p-value: {p_spearman:.2e}")
print(f"Normalized MSE: {global_nmse:.3f}")
print(f"95% CI: [{nmse_ci_lower:.3f}, {nmse_ci_upper:.3f}]")

#%% --- Results - Part 2: Plotting Quartiles ---

all_predicted_ranks = np.array(all_predicted_ranks)
all_true_ranks = np.array(all_true_ranks)
all_test_indices = np.array(all_test_indices)

# Define quartiles for prediction and ground truth
qt_pred = np.percentile(all_predicted_ranks, [25, 50, 75])
qt_true = np.percentile(all_true_ranks, [25, 50, 75])

predicted_quartiles = [np.where(all_predicted_ranks <= qt_pred[0])[0],
                       np.where((all_predicted_ranks > qt_pred[0]) & (all_predicted_ranks <= qt_pred[1]))[0],
                       np.where((all_predicted_ranks > qt_pred[1]) & (all_predicted_ranks <= qt_pred[2]))[0],
                       np.where(all_predicted_ranks > qt_pred[2])[0]]

true_quartiles = [np.where(all_true_ranks <= qt_true[0])[0],
                  np.where((all_true_ranks > qt_true[0]) & (all_true_ranks <= qt_true[1]))[0],
                  np.where((all_true_ranks > qt_true[1]) & (all_true_ranks <= qt_true[2]))[0],
                  np.where(all_true_ranks > qt_true[2])[0]]

# Extract EEG signals
post_stimulus_combined_test = post_stimulus[:, all_test_indices].T  # (trials, samples)
pre_stimulus_combined_test = pre_stimulus[:, all_test_indices].T
pre_post_stimulus_test_c3 = np.concatenate([pre_stimulus_combined_test, post_stimulus_combined_test], axis=1).T  # shape: (2000, 1560)

# Visualize
visualize_quartiles_comparison_fixed_ylim(pre_post_stimulus_test_c3, predicted_quartiles, true_quartiles,
                                          "All Folds - Predicted vs True Quartile EEG Signals (Fixed Y-Lim)")


#%% --- Results - Part 3: SHAP 
# Split data
from sklearn.model_selection import train_test_split

indices = np.arange(feature_df.shape[0])
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    feature_df, targets, indices, test_size=CONFIG['test_size'], random_state=CONFIG['random_state']
)

# --- Hyperparameter tuning using grid search ---
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=CONFIG['random_state']),
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_
print(f"Selected RF Parameters: {grid_search.best_params_}")

# --- Evaluate ---
interpret_model_with_importance(model, X_test, y_test, feature_labels)

#%% statistical analyses comparing nMSE and Spearman correlation between 'Area' and 'Peak' targets
from scipy.stats import ttest_rel

# Example placeholder values (replace these with your actual 5-fold results)
nmse_area = [0.6, 0.62, 0.73, 0.61, 0.54]
nmse_peak = [1.05, 0.87, 0.99, 1.1, 0.79]

spearman_area = [0.76, 0.65, 0.72, 0.85, 0.78]
spearman_peak = [0.52, 0.54, 0.48, 0.69, 0.64]

# Paired t-tests
nmse_ttest = ttest_rel(nmse_area, nmse_peak)
spearman_ttest = ttest_rel(spearman_area, spearman_peak)

print("\n===== Statistical Comparison Between Targets =====")
print(f"nMSE: t = {nmse_ttest.statistic:.4f}, p = {nmse_ttest.pvalue:.4e}")
print(f"Spearman's ρ: t = {spearman_ttest.statistic:.4f}, p = {spearman_ttest.pvalue:.4e}")

