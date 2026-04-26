# ECON 3916 Final Project — Employee Attrition Prediction

**Live Dashboard:** https://finalproject-datdo-econ3916.streamlit.app/

**Author:** Dat Do | Northeastern University | Spring 2026

---

## Project Overview

This project predicts voluntary employee attrition using the IBM HR Analytics Employee Attrition dataset (N=1,470). The prediction question is: can we predict whether an employee will voluntarily leave the organization based on their demographic, compensation, and job-satisfaction characteristics?

Two models are compared — Logistic Regression and Random Forest — using 5-fold stratified cross-validation with bootstrap confidence intervals. Logistic Regression is selected as the preferred final model (CV F1: 0.5552).

**This is a prediction tool, not causal inference. The dashboard should be used for HR risk-screening only, not for automated employment decisions.**

---

## Repository Structure

```
FinalProject-Employee-Attrition-Prediction/
├── Checkpoint 1/
│   └── Final_Project_Dat_Do.ipynb       # Checkpoint submission (April 19)
├── Data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   └── Readme.md                        # Data access instructions
├── Final Submission/
│   └── Final_Project_Dat_Do.ipynb       # Final notebook with full analysis
├── P.R.I.M.E Prompts/
│   ├── Readme.md
│   ├── Repo_Readme.md
│   ├── Data_Readme.md
│   ├── Final_deliverables.md
│   └── Streamlit_app.md                 # AI prompt documentation
├── Streamlit/
│   ├── app.py                           # Streamlit dashboard
│   ├── requirements.txt                 # Python dependencies
│   └── Readme.md                        # Dashboard access instructions
└── README.md                            # This file
```

---

## Data Access Instructions

The dataset is included in this repository at `Data/WA_Fn-UseC_-HR-Employee-Attrition.csv`.

**Source:** IBM HR Analytics Employee Attrition Dataset  
**Kaggle URL:** https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset  
**Access date:** April 2026

### Option 1: Use the file already in this repo (recommended)

The CSV is already in the `Data/` folder. The notebook and Streamlit app load it directly from the raw GitHub URL — no download or API key required.

```python
import pandas as pd

url = "https://raw.githubusercontent.com/ddoneu/FinalProject-Employee-Attrition-Prediction/refs/heads/main/Data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(url)
print(df.shape)
```

### Option 2: Download fresh from Kaggle

Install kagglehub first:

```bash
pip install kagglehub
```

Then run:

```python
import kagglehub
import pandas as pd
from pathlib import Path

path = Path(kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset"))
df = pd.read_csv(path / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(df.shape)
```

### Expected file

```
WA_Fn-UseC_-HR-Employee-Attrition.csv
Shape: (1470, 35)
```

---

## How to Run the Notebook

1. Open `Final Submission/Final_Project_Dat_Do.ipynb` in Google Colab or Jupyter
2. Run all cells top to bottom
3. Data loads automatically via the raw GitHub URL — no setup needed

---

## How to Launch the Streamlit App Locally

```bash
cd Streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Model Summary

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8605 | 0.6154 | 0.3404 | 0.4384 | 0.8079 |
| Random Forest | 0.8265 | 0.3333 | 0.0851 | 0.1356 | 0.8009 |

**5-Fold CV F1:** Logistic Regression 0.5552 vs Random Forest 0.3044

Logistic Regression outperforms Random Forest on all five metrics and in every cross-validation fold and is selected as the preferred final model.

---

## Reproducibility

All random operations use `random_state=42`. Train/test split uses `test_size=0.20` with `stratify=y` to preserve the 16% attrition rate in both sets.

---

## AI Usage

AI tools were used during the final submission stage. All prompts are documented in the `P.R.I.M.E Prompts/` folder following the P.R.I.M.E. framework required by the course. The checkpoint analysis was completed before AI assistance.
