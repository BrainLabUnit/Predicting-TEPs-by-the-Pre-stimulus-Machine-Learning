#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:27:28 2025

@author: sadafmoaveninejad
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the actual data
data_directory = '/Volumes/SADAF/Unipd/TMS_EEG'
data = np.load(os.path.join(data_directory, "EEG_Measurements.npy"))

# Extract the signal from Channel 4 for a single trial (replace 0 with your trial index)
channel_4_signal = data[4, :, 1500]  # Channel 4, trial 0
pre_stimulus = channel_4_signal[:1000]  # First 1000 samples as pre-stimulus
post_stimulus = channel_4_signal[1000:]  # Next 1000 samples as post-stimulus
time = np.arange(-1000, 1000, 1)  # Time in ms, 1 ms per sample

# Smooth the signal using a moving average
def smooth_signal(signal, window_size=10):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

pre_stimulus_smoothed = smooth_signal(pre_stimulus)
post_stimulus_smoothed = smooth_signal(post_stimulus)

# Plot the smoothed signal
plt.figure(figsize=(14, 8))

# Pre-stimulus
plt.plot(time[:1000], pre_stimulus_smoothed, label="Pre-Stimulus", color="green")
# Post-stimulus
plt.plot(time[1000:], post_stimulus_smoothed, label="Post-Stimulus", color="blue")

# Add vertical line for stimulus point
plt.axvline(0, color="red", linestyle="--", linewidth=4, label="Stimulus")

# Add shaded regions
plt.axvspan(-1000, 0, facecolor="gray", alpha=0.1)
plt.axvspan(0, 1000, facecolor="blue", alpha=0.1)

# Annotate the regions
features_list = "Input Features: \nFreq, Entropy,\nHFD, DFA, HE"
targets_list = "Targets: \nArea, Peak,\nBeta Power, Gamma Power,\nAlpha Power, Theta Power,\nHFD, DFA, HE"

plt.text(-350, np.max(pre_stimulus_smoothed) * 1.3, features_list, color="green",
         fontsize=16, fontweight="bold", bbox=dict(facecolor="white", edgecolor="green", boxstyle="round,pad=1"))
plt.text(150, np.max(pre_stimulus_smoothed) * 1.3, targets_list, color="blue",
         fontsize=16, fontweight="bold", bbox=dict(facecolor="white", edgecolor="blue", boxstyle="round,pad=1"))

# Labels and legend
#plt.title("TMS-EEG Signal Pre and Post-Stimulus\nFeatures (Input) and Target (Output)", fontsize=16, fontweight="bold")
plt.xlabel("Time (ms)", fontsize=24, fontweight="bold")
plt.ylabel("Amplitude", fontsize=24, fontweight="bold")

# Explicitly set tick labels to bold and increase font size
plt.xticks(fontsize=24, fontweight="bold")
plt.yticks(fontsize=24, fontweight="bold")

# Bold and large legend text
plt.legend(
    loc="upper left", 
    prop={"size": 16, "weight": "bold"}  # Ensure both size and bold weight are applied
)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%%

# Extract the signal from Channel 4 for a single trial (replace 0 with your trial index)
channel_4_signal = data[4, :, 16]  # Channel 4, trial 0
pre_stimulus = channel_4_signal[:1000]  # First 1000 samples as pre-stimulus
post_stimulus = channel_4_signal[1000:]  # Next 1000 samples as post-stimulus
time = np.arange(-1000, 1000, 1)  # Time in ms, 1 ms per sample

# Smooth the signal using a moving average
def smooth_signal(signal, window_size=10):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

pre_stimulus_smoothed = smooth_signal(pre_stimulus)
post_stimulus_smoothed = smooth_signal(post_stimulus)

# Plot the smoothed signal
plt.figure(figsize=(14, 8))

# Pre-stimulus
plt.plot(time[:1000], pre_stimulus_smoothed, color="green")
# Post-stimulus
plt.plot(time[1000:], post_stimulus_smoothed, label="Post-Stimulus", color="blue")

# Add vertical line for stimulus point
plt.axvline(0, color="red", linestyle="--", linewidth=4, label="Stimulus")

#plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
