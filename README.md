# Predicting TMS-Evoked Potentials from Pre-Stimulus EEG Features

This repository contains Python scripts to reproduce the experiments from our GNB 2025 conference paper:

**“Prediction of TMS-evoked Potentials from Pre-stimulus Spectral Features: A Machine Learning Approach”**  
*Authors: Sadaf Moaveninejad et al.*

---

## 📂 Repository Structure

```
├── GNB_main_2025.py         # Main script: ML pipeline, evaluation, SHAP
├── GNB_functions_2025.py    # Helper functions: target extraction, SHAP, visualization
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
