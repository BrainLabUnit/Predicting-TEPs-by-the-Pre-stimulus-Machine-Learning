#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 11:46:22 2025

@author: sadafmoaveninejad
"""

from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

def interpret_model_with_importance(model, X_test, y_test, feature_labels):
    """Assess model feature importance using SHAP and Permutation Feature Importance (PFI)."""

    # Set font properties
    rcParams['font.size'] = 16
    rcParams['font.weight'] = 'bold'
    rcParams['axes.labelweight'] = 'bold'

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)


    # SHAP Summary Plot
    plt.figure(figsize=(14, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_labels)
    plt.title("SHAP Summary Plot", fontsize=18, fontweight='bold')
    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.show()


# Compute post-stimulus target
def compute_targets(data, method="area", fs=1000):
    """
    Compute post-stimulus target values based on the selected method.

    Parameters:
    - data: numpy array of shape (channels, samples, trials).
    - method: str, method to compute targets (e.g., "area", "peak", "auc", "tep_slope", "entropy").
    - channel_idx: int, channel to use for computation.

    Returns:
    - numpy array of target values for each trial.
    """
    post_stimulus = data

    if method == "area":
        return np.trapz(np.abs(post_stimulus), axis=0)

    elif method == "auc":
        return np.trapz(post_stimulus, axis=0)

    elif method == "peak":
        return np.max(post_stimulus, axis=0)
    
    elif method == "peak_latency":
        return np.array([np.argmax(np.abs(post_stimulus[:, i])) / fs for i in range(post_stimulus.shape[1])])
    
    elif method == "tep_energy":  # New target
        return np.sum(post_stimulus ** 2, axis=0)

    elif method == "tep_slope":
        slopes = []
        for i in range(post_stimulus.shape[1]):
            signal = post_stimulus[:, i]
            max_idx = np.argmax(np.abs(signal))  # Index of max absolute peak
            min_idx = np.argmin(np.abs(signal))  # Index of min absolute peak

            if max_idx > min_idx:
                slope = (signal[max_idx] - signal[min_idx]) / (max_idx - min_idx)
            else:
                slope = (signal[min_idx] - signal[max_idx]) / (min_idx - max_idx)

            slopes.append(slope)
        return np.array(slopes)


import numpy as np
import matplotlib.pyplot as plt

def visualize_quartiles_comparison_fixed_ylim(data, predicted_quartiles, true_quartiles, title):
    """
    Visualize quartile comparisons in a 2x2 subplot layout with y-axis limited to [-19, 19].

    Parameters:
    - data: numpy array of shape (samples, trials), pre- and post-stimulus data.
    - predicted_quartiles: list of arrays, each containing indices of trials in the predicted quartile.
    - true_quartiles: list of arrays, each containing indices of trials in the true quartile.
    - title: str, title of the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)  # 2x2 layout
    quartile_labels = ["1st-Quartile", "2nd-Quartile", "3rd-Quartile", "4th-Quartile"]

    # Define x-axis range (-1000 to 1000)
    time_range = np.linspace(-1000, 1000, data.shape[0])

    for i, ax in enumerate(axes.flat):
        pred_mean = np.mean(data[:, predicted_quartiles[i]], axis=1)
        true_mean = np.mean(data[:, true_quartiles[i]], axis=1)

        # Thicker Curves
        ax.plot(time_range, pred_mean, label="Predicted", linestyle='-', linewidth=3, color='blue')
        ax.plot(time_range, true_mean, label="True", linestyle='--', linewidth=3, color='red')

        # Set fixed y-axis limits
        ax.set_ylim(-16, 13)

        ax.set_title(quartile_labels[i], fontsize=22, fontweight="bold")

        # Set X and Y labels/ticks correctly
        if i >= 2:  
            ax.set_xlabel("Time (ms)", fontsize=22, fontweight="bold")

        if i % 2 == 0: 
            ax.set_ylabel("Amplitude", fontsize=22, fontweight="bold")

        ax.grid(alpha=0.5)
        ax.legend(fontsize=18, loc="lower left")  # Ensure legend is visible

    # Force display of tick labels
    plt.setp(axes[-1, :], xticks=np.linspace(-1000, 1000, 5))  # Ensure bottom row has x-ticks
    plt.setp(axes[:, 0], yticks=np.linspace(-16, 13, 5))  # Ensure left column has y-ticks

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to prevent legend clipping
    plt.show()



