# Predicting TMS-Evoked Potentials from Pre-Stimulus EEG Features

![Python](https://img.shields.io/badge/python-3.8+-blue)
![Repo size](https://img.shields.io/github/repo-size/BrainLabUnit/Predicting-TEPs-by-the-Pre-stimulus-Machine-Learning)
![Last commit](https://img.shields.io/github/last-commit/BrainLabUnit/Predicting-TEPs-by-the-Pre-stimulus-Machine-Learning)

This repository contains Python scripts and supporting data used in the study:

**“Prediction of TMS-evoked Potentials from Pre-stimulus Spectral Features: A Machine Learning Approach”**  
*Presented at GNB 2025 (June 16–18), Palermo, Italy*  
*Authors: Sadaf Moaveninejad, Antonio Luigi Bisogno, Simone Cauzzo, Maurizio Corbetta, Camillo Porcaro*

---

## 📘 Project Summary

Transcranial Magnetic Stimulation (TMS) combined with EEG enables non-invasive assessment of brain responsiveness.  
This project explores how **pre-stimulus spectral features** (from delta to gamma bands) can **predict TMS-evoked potentials (TEPs)** — specifically, the **peak amplitude** and **signal area** of post-stimulus activity.

A **Random Forest Regressor** was trained using **subject-stratified 5-fold cross-validation** to ensure generalization.  
Results indicate that predicting **signal area** outperforms peak amplitude in accuracy and correlation.

🔍 This work provides insight into the role of brain state in shaping neural responses, and guides more personalized and efficient TMS interventions.

🖼️ The poster associated with this study is included in this repository as:
**`GNB25_poster_SM.pdf`**

---

## 📂 Repository Structure

```
├── GNB_main_2025.py         # Main script that performs:
│   ├── Configuration and loading of EEG features
│   ├── Preprocessing and data-frame construction
│   ├── Target extraction from post-stimulus signal
│   ├── Subject-stratified 5-fold cross-validation
│   ├── Hyperparameter tuning using GridSearchCV
│   ├── Model evaluation (nMSE, Spearman’s ρ, CI bootstrapping)
│   ├── Quartile-based EEG signal visualization
│   └── SHAP-based feature importance analysis
│
├── GNB_functions_2025.py    # Helper functions:
│   ├── compute_targets: extract TEP metrics (area, peak, latency, etc.)
│   ├── interpret_model_with_importance: SHAP-based feature interpretation
│   └── visualize_quartiles_comparison_fixed_ylim: EEG quartile group comparison
│
├── GNB25_poster_SM.pdf      # Conference poster (GNB 2025, Palermo)
│
├── RogashData/              # Pre-extracted features and EEG data
│   ├── alpha.npy
│   ├── beta.npy
│   ├── delta.npy
│   ├── gamma.npy
│   ├── theta.npy
│   ├── pre_stimulus_c3.npy
│   ├── post_stimulus_c3.npy
│   └── TrialsVsSubjects.xlsx
```

⚠️ The data in `RogashData/` is derived from the **public dataset** available at:  
🔗 [BMHLab/TEPs-PEPs](https://github.com/BMHLab/TEPs-PEPs)

---

## ⚙️ Configuration

The following configuration can be adjusted at the top of `GNB_main_2025.py`:

```
CONFIG = {
    "channel_index": 5,           # EEG channel index for C3
    "target_method": "area",      # or "peak"
    "random_state": 42,
    "test_size": 0.2,
    "features_type": "allbands"   # or "alpha"
}
```

---

## 📊 Outputs

- Spearman’s ρ and normalized MSE (nMSE) across 5-fold subject-stratified CV  
- SHAP plots showing feature importance  
- Quartile-based signal plots for predicted vs. true trial groups  
- Statistical comparisons between prediction targets

---

## 🧩 GNB_functions_2025.py Overview

This script contains reusable functions used in the main pipeline, including:

- `compute_targets`:  
  Extracts post-stimulus features such as **area**, **peak amplitude**, **AUC**, **latency**, **slope**, and **energy** from EEG trials.

- `interpret_model_with_importance`:  
  Uses **SHAP** (Shapley Additive Explanations) to visualize the contribution of each frequency band feature to model predictions.

- `visualize_quartiles_comparison_fixed_ylim`:  
  Generates a 2×2 subplot visualization comparing **true vs. predicted quartile groups** based on EEG signal response (with fixed y-axis limits).

These utility functions support model interpretability, visualization, and target extraction for TMS-EEG regression tasks.

---

## 📦 Citation

If you use this code or data in your work, please cite:

S. Moaveninejad et al., “Prediction of TMS-evoked Potentials from Pre-stimulus Spectral Features: A Machine Learning Approach,” *GNB 2025*, Palermo, Italy.

---

## 🚀 How to Run

1. **Clone this repository**

```
git clone https://github.com/BrainLabUnit/Predicting-TEPs-by-the-Pre-stimulus-Machine-Learning.git
cd Predicting-TEPs-by-the-Pre-stimulus-Machine-Learning
```

2. **Ensure `RogashData/` folder is present**

The pre-extracted features and EEG arrays must be placed in the `RogashData/` folder. If you are missing this data, please refer to the original dataset link above or contact the repository maintainer.

3. **Install required packages (optional)**

```
pip install -r requirements.txt
```

4. **Run the main pipeline**

```
python GNB_main_2025.py
```

---
