# Predicting TMS-Evoked Potentials from Pre-Stimulus EEG Features

This repository contains Python scripts to reproduce the experiments from our GNB 2025 conference paper:

**“Prediction of TMS-evoked Potentials from Pre-stimulus Spectral Features: A Machine Learning Approach”**  
*Authors: Sadaf Moaveninejad et al.*

---

## 📂 Repository Structure
├── GNB_main_2025.py # Main script: ML pipeline, evaluation, SHAP
├── GNB_functions_2025.py # Helper functions: target extraction, SHAP, visualization
├── RogashData/ # Pre-extracted features and EEG data
│ ├── alpha.npy
│ ├── beta.npy
│ ├── delta.npy
│ ├── gamma.npy
│ ├── theta.npy
│ ├── pre_stimulus_c3.npy
│ ├── post_stimulus_c3.npy
│ └── TrialsVsSubjects.xlsx


> ⚠️ The data in `RogashData/` is derived from the **public dataset** available at  
> 🔗 [BMHLab/TEPs-PEPs](https://github.com/BMHLab/TEPs-PEPs)  

---

## 🚀 How to Run

1. **Clone this repository**
   ```bash
   git clone https://github.com/BrainLabUnit/Predicting-TEPs-by-the-Pre-stimulus-Machine-Learning.git
   cd Predicting-TEPs-by-the-Pre-stimulus-Machine-Learning

2. **Ensure RogashData/ folder is present

3. **Run the main pipeline
   python GNB_main_2025.py
   
