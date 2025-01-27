#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:03:25 2025

@author: sadafmoaveninejad
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, rankdata, skew, kurtosis
from scipy.signal import welch, coherence
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pywt
import os
import mne
import shap

# Configurations
CONFIG = {
    'fs': 1000,  # Sampling frequency
    'target_method': 'hurst',  # Method to compute targets: "peak", "beta_power", "gamma_power", "area", "higuchi_fd", "dfa", "hurst", "delta_power", "theta_power", "alpha_power", "entropy"
    'test_size': 0.2,  # Train-test split ratio
    'random_state': 42,  # Random seed for reproducibility
    'feature_types': ['frequency_domain', 'entropy', 'higuchi_FD', 'dfa', 'hurst_exponent'],  # Feature types to include
    'sliding_window_size': 200,  # Window size for temporal analysis (in samples)
    'sliding_window_step': 100,  # Step size for sliding windows (in samples)
    'connectivity_analysis': False,  # Include functional connectivity analysis for more than one channnel
    'cfc_analysis': True,  # Include cross-frequency coupling analysis
    'channels': [4]#'all'  # Either 'all' or a list of specific channels (e.g., [0, 1, 4, 5])
}


# Feature Extraction Functions
def extract_features(signal, fs=CONFIG['fs']):
    """Extract features for a single EEG signal based on selected types in CONFIG."""
    features = []

    if 'time_domain' in CONFIG['feature_types']:
        # Time-domain features
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        min_val = np.min(signal)
        max_val = np.max(signal)
        skew_val = skew(signal)
        kurtosis_val = kurtosis(signal)
        features.extend([mean_val, std_val, min_val, max_val, skew_val, kurtosis_val])

    if 'frequency_domain' in CONFIG['feature_types']:
        # Frequency-domain features
        nperseg = min(len(signal), 256)  # Ensure nperseg is valid
        freqs, psd = welch(signal, fs, nperseg=nperseg)
        delta_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])
        theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
        alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 12)])
        beta_power = np.sum(psd[(freqs >= 12) & (freqs <= 30)])
        gamma_power = np.sum(psd[(freqs >= 30) & (freqs <= 100)])
        features.extend([delta_power, theta_power, alpha_power, beta_power, gamma_power])

    if 'entropy' in CONFIG['feature_types']:
        # Entropy
        prob_dist, _ = np.histogram(signal, bins=10, density=True)
        prob_dist /= prob_dist.sum()  # Normalize histogram
        signal_entropy = entropy(prob_dist)
        features.append(signal_entropy)

    if 'wavelet_transform' in CONFIG['feature_types']:
        # Wavelet Transform Features
        coeffs = pywt.wavedec(signal, 'db4', level=4, mode='smooth')
        wavelet_means = [np.mean(c) for c in coeffs]
        wavelet_stds = [np.std(c) for c in coeffs]
        features.extend(wavelet_means + wavelet_stds)

    if 'higuchi_FD' in CONFIG['feature_types']:
        # Higuchi Fractal Dimension (HFD)
        hfd = compute_FD(signal, max_k=50)
        features.append(hfd)

    if 'dfa' in CONFIG['feature_types']:
        # Detrended Fluctuation Analysis (DFA)
        dfa_alpha = compute_fractal_dimension('DFA', signal, fs)
        features.append(dfa_alpha)

    if 'hurst_exponent' in CONFIG['feature_types']:
        # Hurst Exponent
        hurst_exp = compute_fractal_dimension('HE', signal, fs)
        features.append(hurst_exp)

    return np.array(features)

def compute_FD(time_series, max_k=50):
    """Compute Higuchi Fractal Dimension (HFD) for a time series."""
    def Lmk(time_series, m, k):
        N = len(time_series)
        num_segments = int(np.floor((N - m) / k))
        summation = sum(abs(time_series[m + i * k - 1] - time_series[m + (i - 1) * k - 1]) for i in range(1, num_segments + 1))
        return (summation / num_segments) / k

    L_values = [np.mean([Lmk(time_series, m, k) for m in range(1, k + 1)]) for k in range(1, max_k + 1)]
    log_k = np.log10(np.clip(range(1, max_k + 1), a_min=1e-10, a_max=None))
    log_L = np.log10(np.clip(L_values, a_min=1e-10, a_max=None))
    return np.polyfit(log_k, log_L, 1)[0]

def dfa(data, win_length, order=1):
    N = len(data)
    n = N // win_length  # Number of windows
    N1 = n * win_length  # Adjusted length of the data for the windows
    y = np.cumsum(data[:N1] - np.mean(data[:N1]))  # Cumulative sum adjusted by the mean

    # Fit a polynomial to each window of the data
    fitcoef = np.zeros((n, order + 1))
    Yn = np.zeros(N1)
    for j in range(n):
        window_slice = slice(j * win_length, (j + 1) * win_length)
        fitcoef[j, :] = np.polyfit(np.arange(1, win_length + 1), y[window_slice], order)
        Yn[window_slice] = np.polyval(fitcoef[j, :], np.arange(1, win_length + 1))

    # Calculate the root mean square fluctuation
    rms_fluctuation = np.sqrt(np.mean((y - Yn) ** 2))

    return rms_fluctuation

def DFA(DATA, fs):
    win_lengths = np.arange(10, len(DATA), 5)  # Window lengths
    F_n = np.array([dfa(DATA, int(wl), 1) for wl in win_lengths])  # Calculate DFA for each window length

    # Perform log-log linear fit
    A = np.polyfit(np.log(win_lengths), np.log(F_n), 1)
    Alpha1 = A[0]
    return Alpha1

def HE(S, fs, max_T=19):
    """
    Calculate the Fractal Dimension (FD) using the Hurst exponent.

    Parameters:
    - S: array-like
        The input signal (1D array).
    - fs: int
        Sampling frequency of the signal (not used in this implementation but kept for compatibility).
    - max_T: int
        Maximum scale for the fluctuation analysis (default is 30).

    Returns:
    - FD: float
        The fractal dimension calculated as FD = 2 - H.
    """
    # Convert to numpy array
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 1:
        raise ValueError("S must be a 1D array.")

    L = len(S)
    H = np.zeros(max_T - 4)

    for Tmax in range(5, max_T + 1):
        x = np.arange(1, Tmax + 1)
        mcord = np.zeros(Tmax)

        for tt in range(1, Tmax + 1):
            dV = S[tt:] - S[:-tt]
            VV = S[:-tt]
            N = len(dV)
            X = np.arange(1, N + 1)
            Y = VV
            mx = np.mean(X)
            SSxx = np.sum((X - mx)**2)
            my = np.mean(Y)
            SSxy = np.sum((X - mx) * (Y - my))
            cc1 = SSxy / SSxx
            cc2 = my - cc1 * mx
            dVd = dV - cc1
            VVVd = VV - cc1 * X - cc2

            mcord[tt - 1] = np.mean(np.abs(dVd)**2) / np.mean(np.abs(VVVd)**2)

        mx = np.mean(np.log10(x))
        SSxx = np.sum((np.log10(x) - mx)**2)
        my = np.mean(np.log10(mcord))
        SSxy = np.sum((np.log10(x) - mx) * (np.log10(mcord) - my))
        H[Tmax - 5] = SSxy / SSxx

    # Format H values to two decimal places
    H = np.round(H, 2)
    mH = np.mean(H)
    FD = 2 - mH
    return FD

def compute_fractal_dimension(method, signal, fs):
    """
    Compute the fractal dimension of a signal using the specified method.

    :param method: String representing the fractal dimension estimation method.
    :param signal: 1D numpy array representing the time series.
    :return: Estimated fractal dimension.
    """
    methods = {
        'DFA': DFA,
        'HE': HE,
        'HFD': compute_FD
    }

    if method in methods:
        return methods[method](signal, fs)
    else:
        raise ValueError(f"Method {method} is not recognized.")


def functional_connectivity_analysis(data, fs=CONFIG['fs']):
    """Compute functional connectivity using coherence between all channel pairs."""
    n_channels = data.shape[0]
    connectivity_matrix = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            freqs, coherence_values = coherence(data[i], data[j], fs=fs, nperseg=256)
            connectivity_matrix[i, j] = np.mean(coherence_values[(freqs >= 4) & (freqs <= 30)])  # Theta to Beta range
            connectivity_matrix[j, i] = connectivity_matrix[i, j]

    return connectivity_matrix

import numpy as np
import matplotlib.pyplot as plt
from pactools import Comodulogram

# Function for cross-frequency coupling
def cross_frequency_coupling(signal, fs=1000):
    """Compute phase-amplitude coupling (PAC) as a measure of cross-frequency coupling."""
    pac = Comodulogram(fs=fs, low_fq_range=np.arange(4, 13, 1), high_fq_range=np.arange(30, 100, 5), method='tort')
    pac.fit(signal)
    return pac.comod_  # Returns PAC matrix

# Plot function with increased font size and bold styling for colorbar ticks
def plot_cfc_matrices(pre_cfc, post_cfc, freq_low, freq_high, title_pre, title_post, title_diff):
    """Plot pre-stimulus, post-stimulus, and difference CFC matrices."""
    plt.figure(figsize=(18, 6))

    # Pre-Stimulus PAC
    plt.subplot(1, 3, 1)
    plt.imshow(pre_cfc, cmap='hot', aspect='auto', extent=[freq_high[0], freq_high[-1], freq_low[0], freq_low[-1]], origin='lower')
    cbar = plt.colorbar(label="PAC")
    cbar.ax.set_ylabel("PAC", fontsize=20, fontweight='bold')  # Colorbar label bold
    cbar.ax.tick_params(labelsize=16, labelcolor='black', width=1.5)  # Colorbar ticks font size and bold
    plt.title(title_pre, fontsize=20, fontweight='bold')
    plt.xlabel("High Frequency (Hz)", fontsize=20, fontweight='bold')
    plt.ylabel("Low Frequency (Hz)", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')  # Tick font size and bold
    plt.yticks(fontsize=16, fontweight='bold')

    # Post-Stimulus PAC
    plt.subplot(1, 3, 2)
    plt.imshow(post_cfc, cmap='hot', aspect='auto', extent=[freq_high[0], freq_high[-1], freq_low[0], freq_low[-1]], origin='lower')
    cbar = plt.colorbar(label="PAC")
    cbar.ax.set_ylabel("PAC", fontsize=20, fontweight='bold')  # Colorbar label bold
    cbar.ax.tick_params(labelsize=16, labelcolor='black', width=1.5)  # Colorbar ticks font size and bold
    plt.title(title_post, fontsize=20, fontweight='bold')
    plt.xlabel("High Frequency (Hz)", fontsize=20, fontweight='bold')
    plt.ylabel("Low Frequency (Hz)", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')  # Tick font size and bold
    plt.yticks(fontsize=16, fontweight='bold')

    # Difference (Post - Pre)
    plt.subplot(1, 3, 3)
    diff_cfc = post_cfc - pre_cfc
    plt.imshow(diff_cfc, cmap='bwr', aspect='auto', extent=[freq_high[0], freq_high[-1], freq_low[0], freq_low[-1]], origin='lower')
    cbar = plt.colorbar(label="PAC Difference (Post - Pre)")
    cbar.ax.set_ylabel("PAC Difference (Post - Pre)", fontsize=20, fontweight='bold')  # Colorbar label bold
    cbar.ax.tick_params(labelsize=16, labelcolor='black', width=1.5)  # Colorbar ticks font size and bold
    plt.title(title_diff, fontsize=20, fontweight='bold')
    plt.xlabel("High Frequency (Hz)", fontsize=20, fontweight='bold')
    plt.ylabel("Low Frequency (Hz)", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')  # Tick font size and bold
    plt.yticks(fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.show()

def interpret_model_with_shap(model, X_test, feature_labels):
    """Use SHAP to interpret the machine learning model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_labels, plot_type="bar")
    plt.title("SHAP Feature Importance")
    plt.show()

    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_labels)
    plt.title("SHAP Summary Plot")
    plt.show()

# Feature Extraction Functions
def extract_features_all_channels(data, fs=CONFIG['fs']):
    """Extract features from specified EEG channels for all trials."""
    if len(data.shape) != 3:
        raise ValueError("Input data must have dimensions (channels, samples, trials).")

    n_channels, n_samples, n_trials = data.shape
    all_features = []  # Numerical feature matrix
    feature_labels = []

    # Determine channels to use
    if CONFIG['channels'] == 'all':
        channel_indices = range(n_channels)
    else:
        channel_indices = CONFIG['channels']

    for channel_idx in channel_indices:
        channel_id = channel_idx + 1
        channel_features = []

        if 'time_domain' in CONFIG['feature_types']:
            channel_features.extend([f"Ch{channel_id}-" + feat for feat in [
                "mean", "std", "min", "max", "skew", "kurtosis"]])

        if 'frequency_domain' in CONFIG['feature_types']:
            channel_features.extend([f"Ch{channel_id}-" + feat for feat in [
                "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power"]])

        if 'entropy' in CONFIG['feature_types']:
            channel_features.append(f"Ch{channel_id}-entropy")

        if 'wavelet_transform' in CONFIG['feature_types']:
            channel_features.extend([f"Ch{channel_id}-wavelet_mean_{i+1}" for i in range(5)] +
                                     [f"Ch{channel_id}-wavelet_std_{i+1}" for i in range(5)])

        if 'higuchi_FD' in CONFIG['feature_types']:
            channel_features.append(f"Ch{channel_id}-hfd")

        if 'dfa' in CONFIG['feature_types']:
            channel_features.append(f"Ch{channel_id}-dfa")

        if 'hurst_exponent' in CONFIG['feature_types']:
            channel_features.append(f"Ch{channel_id}-hurst_exp")

        feature_labels.extend(channel_features)

    for trial_idx in range(n_trials):
        trial_features = []
        for channel_idx in channel_indices:
            signal = data[channel_idx, :1000, trial_idx]  # Pre-stimulus only
            trial_features.extend(extract_features(signal, fs))
        all_features.append(trial_features)

    return np.array(all_features), feature_labels


# Compute post-stimulus target
def compute_targets(data, method="area", channel_idx=4):
    """
    Compute post-stimulus target values based on the selected method.

    Parameters:
    - data: numpy array of shape (channels, samples, trials).
    - method: str, method to compute targets (e.g., "area", "peak", "entropy").
    - channel_idx: int, channel to use for computation.

    Returns:
    - numpy array of target values for each trial.
    """
    post_stimulus = data[channel_idx, :, :]
    if method == "area":
        return np.array([np.trapz(np.abs(post_stimulus[:, i])) for i in range(post_stimulus.shape[1])])
    elif method == "peak":
        return np.array([np.max(post_stimulus[:, i]) for i in range(post_stimulus.shape[1])])
    elif method in ["beta_power", "gamma_power"]:
        freqs, psd = welch(post_stimulus, fs=CONFIG['fs'], nperseg=256, axis=0)  # Adjust nperseg for better resolution
        if method == "beta_power":
            beta_band = (freqs >= 12) & (freqs <= 30)
            return np.sum(psd[beta_band], axis=0)
        elif method == "gamma_power":
            gamma_band = (freqs >= 30) & (freqs <= 100)
            return np.sum(psd[gamma_band], axis=0)
    elif method == "higuchi_fd":
        return np.array([compute_FD(post_stimulus[:, i]) for i in range(post_stimulus.shape[1])])
    elif method == "dfa":
        return np.array([DFA(post_stimulus[:, i], CONFIG['fs']) for i in range(post_stimulus.shape[1])])
    elif method == "hurst":
        return np.array([HE(post_stimulus[:, i], CONFIG['fs']) for i in range(post_stimulus.shape[1])])
    elif method == "entropy":
        return np.array([
            entropy(np.histogram(post_stimulus[:, i], bins=10, density=True)[0])
            for i in range(post_stimulus.shape[1])
        ])
    elif method == "delta_power":
        freqs, psd = welch(post_stimulus, fs=CONFIG['fs'], nperseg=256, axis=0)
        delta_band = (freqs >= 0.5) & (freqs <= 4)
        return np.sum(psd[delta_band], axis=0)
    elif method == "theta_power":
        freqs, psd = welch(post_stimulus, fs=CONFIG['fs'], nperseg=256, axis=0)
        theta_band = (freqs >= 4) & (freqs <= 8)
        return np.sum(psd[theta_band], axis=0)
    elif method == "alpha_power":
        freqs, psd = welch(post_stimulus, fs=CONFIG['fs'], nperseg=256, axis=0)
        alpha_band = (freqs >= 8) & (freqs <= 12)
        return np.sum(psd[alpha_band], axis=0)
    else:
        raise ValueError(f"Invalid method for target computation: {method}")


# Visualize quartiles with bold and larger fonts
def visualize_quartiles(data, quartiles, title):
    plt.figure(figsize=(12, 8))
    for i, quartile in enumerate(quartiles):
        avg_signal = np.mean(data[:, quartile], axis=1)
        std_signal = np.std(data[:, quartile], axis=1)
        plt.plot(avg_signal, label=f"Quartile {i + 1}")
        # Optionally add shaded error regions for standard deviation
        # plt.fill_between(range(len(avg_signal)), avg_signal - std_signal, avg_signal + std_signal, alpha=0.2)
    
    # Set title and labels with bold font
    #plt.title(title, fontsize=20, fontweight="bold")
    plt.xlabel("Time (samples)", fontsize=22, fontweight="bold")
    plt.ylabel("Amplitude", fontsize=22, fontweight="bold")
    
    # Customize legend
    # Bold and large legend text
    plt.legend(
        loc="upper left", 
        prop={"size": 20 ,"weight": "bold"}  # Ensure both size and bold weight are applied
    )
    
    # Customize ticks
    plt.xticks(fontsize=22, fontweight="bold")
    plt.yticks(fontsize=22, fontweight="bold")
    
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()

    

from sklearn.metrics import mean_squared_error, r2_score

# Evaluate Model
def evaluate_model(model, X_test, y_test, feature_labels):
    y_pred = model.predict(X_test)
    
    # Compute MSE
    mse = mean_squared_error(y_test, y_pred)
    
    # Compute Normalized MSE
    variance = np.var(y_test)
    nmse = mse / variance if variance > 0 else np.nan  # Avoid division by zero
    
    # Compute R²
    r2 = r2_score(y_test, y_pred)
    
    print(f"MSE: {mse}, Normalized MSE (nMSE): {nmse}, R²: {r2}")
    
    # Feature Importance
    feature_importances = model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_labels = [feature_labels[i] for i in sorted_indices]
    sorted_importances = feature_importances[sorted_indices]

    # Plot Feature Importances
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_labels[:20], sorted_importances[:20])  # Plot top 20 features
    plt.title("Top 20 Feature Importances (Train-Test Split)", fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=10)
    plt.ylabel("Importance", fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.show()


# Main Code
if __name__ == "__main__":
    # Load data
    data_directory = '/Volumes/SADAF/Unipd/TMS_EEG'
    data = np.load(os.path.join(data_directory, "EEG_Measurements.npy"))

    # Preprocess data
    pre_stimulus = data[:, :1000, :]
    post_stimulus = data[:, 1000:, :]

    # Extract features and compute targets
    features, feature_labels = extract_features_all_channels(pre_stimulus, fs=CONFIG['fs'])
    scaler = StandardScaler()
    features = scaler.fit_transform(features)  # Normalize features
    targets = compute_targets(post_stimulus, method=CONFIG['target_method'])

    # Train-Test Split with indices tracking
    indices = np.arange(features.shape[0])  # Create an array of indices
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        features, targets, indices, test_size=CONFIG['test_size'], random_state=CONFIG['random_state']
    )

    # Train model
    model = RandomForestRegressor(random_state=CONFIG['random_state'])
    model.fit(X_train, y_train)

    # Evaluate model and visualize results
    evaluate_model(model, X_test, y_test, feature_labels)

    # Interpret model using SHAP
    interpret_model_with_shap(model, X_test, feature_labels)

    # Predict and rank trials (Test Set Only)
    predicted_targets = model.predict(X_test)
    predicted_ranks = rankdata(predicted_targets, method="ordinal")
    quartile_thresholds = np.percentile(predicted_ranks, [25, 50, 75])
    predicted_quartiles = [
        np.where(predicted_ranks <= quartile_thresholds[0])[0],
        np.where((predicted_ranks > quartile_thresholds[0]) & (predicted_ranks <= quartile_thresholds[1]))[0],
        np.where((predicted_ranks > quartile_thresholds[1]) & (predicted_ranks <= quartile_thresholds[2]))[0],
        np.where(predicted_ranks > quartile_thresholds[2])[0]
    ]

    # Extract post-stimulus signals for Channel 4 (Test Set Only)
    post_stimulus_test_channel4 = post_stimulus[4, :, test_indices]  # Extract only Channel 4 test set signals
    pre_post_stimulus_test_channel4 = data[4, :, test_indices]  # Extract pre and post signals for Channel 4 (Test Set Only)

    # Transpose the arrays to match the expected dimensions (time samples, trials)
    post_stimulus_test_channel4 = post_stimulus_test_channel4.T  # Now (1000, test_trials)
    pre_post_stimulus_test_channel4 = pre_post_stimulus_test_channel4.T

    # Visualize quartiles for predicted ranks
    visualize_quartiles(pre_post_stimulus_test_channel4, predicted_quartiles, f"Predicted Post-Stimulus Signals by Quartile (Channel 4 - Test Set Only, Target Method: {CONFIG['target_method']})")
    visualize_quartiles(pre_post_stimulus_test_channel4, predicted_quartiles)#, f"Predicted Post-Stimulus Signals by Quartile (Channel 4 - Test Set Only, Target Method: {CONFIG['target_method']})")


    # Compute and visualize original ranks (Test Set Only)
    original_ranks = rankdata(y_test, method="ordinal")
    original_quartile_thresholds = np.percentile(original_ranks, [25, 50, 75])
    original_quartiles = [
        np.where(original_ranks <= original_quartile_thresholds[0])[0],
        np.where((original_ranks > original_quartile_thresholds[0]) & (original_ranks <= original_quartile_thresholds[1]))[0],
        np.where((original_ranks > original_quartile_thresholds[1]) & (original_ranks <= original_quartile_thresholds[2]))[0],
        np.where(original_ranks > original_quartile_thresholds[2])[0]
    ]

    # Visualize original quartiles for Channel 4 (Test Set Only)
    visualize_quartiles(pre_post_stimulus_test_channel4, original_quartiles, f"Original Pre & Post-Stimulus Signals by Quartile (Channel 4 - Test Set Only, Target Method: {CONFIG['target_method']})")
    visualize_quartiles(pre_post_stimulus_test_channel4, original_quartiles)#, f"Original Pre & Post-Stimulus Signals by Quartile (Channel 4 - Test Set Only, Target Method: {CONFIG['target_method']})")

    from scipy.stats import spearmanr
    spearman_corr, _ = spearmanr(original_ranks, predicted_ranks)
    print(f"Spearman's Rank Correlation: {spearman_corr:.4f}")

    # Define channel and sampling frequency
    channel_idx = 4  # Adjust to your target channel
    fs = CONFIG['fs']

    # Extract pre-stimulus and post-stimulus signals for the channel
    pre_stimulus_signal = data[channel_idx, :1000, :].mean(axis=1)  # Averaging across trials
    post_stimulus_signal = data[channel_idx, 1000:, :].mean(axis=1)  # Averaging across trials

    # Compute CFC matrices
    pre_cfc_matrix = cross_frequency_coupling(pre_stimulus_signal, fs)
    post_cfc_matrix = cross_frequency_coupling(post_stimulus_signal, fs)

    # Define frequency ranges (for visualization)
    low_freq_range = np.arange(4, 13, 1)  # Low frequencies (Theta to Alpha)
    high_freq_range = np.arange(30, 100, 5)  # High frequencies (Gamma)

    # Plot CFC matrices
    plot_cfc_matrices(
        pre_cfc_matrix,
        post_cfc_matrix,
        low_freq_range,
        high_freq_range,
        "Pre-Stimulus",#Cross-Frequency Coupling (PAC)
        "Post-Stimulus",#Cross-Frequency Coupling (PAC)
        "Difference"#(Post-Stimulus - Pre-Stimulus)
    )


