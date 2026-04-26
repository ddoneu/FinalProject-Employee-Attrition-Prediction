# Employee Attrition Prediction — ECON 3916 Final Project

**Live Dashboard:** https://finalproject-datdo-econ3916.streamlit.app/

**Author:** Dat Do | Northeastern University | Spring 2026

---

## Project Overview

This project predicts voluntary employee attrition using the IBM HR Analytics Employee Attrition dataset (N=1,470). The prediction question is: can we predict whether an employee will voluntarily leave the organization based on their demographic, compensation, and job-satisfaction characteristics?

Two models are compared — Logistic Regression and Random Forest — using 5-fold stratified cross-validation with bootstrap confidence intervals. Logistic Regression is selected as the preferred final model (CV F1: 0.5552). The Streamlit dashboard allows HR professionals to adjust employee inputs and receive a real-time attrition risk prediction with uncertainty bounds.

**This is a prediction tool, not causal inference. High predicted risk does not prove that any one feature causes attrition. The dashboard should be used for HR risk-screening only, not for automated employment decisions.**

---

## Repository Structure

```
├── Final_Project_Dat_Do.ipynb   # Full analysis notebook
├── Streamlit/
│   ├── app.py                   # Streamlit dashboard
│   └── requirements.txt         # Python dependencies
├── Data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── README.md
└── .python-version
```

---

## Data

**Source:** IBM HR Analytics Employee Attrition Dataset
**URL:** https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
**Size:** 1,470 employees, 35 features, binary target (Attrition Yes/No)
**Access date:** April 2026

The data is also stored in the `Data/` folder of this repository and loaded directly in both the notebook and the Streamlit app via raw GitHub URL.

---

## Environment Setup

**Requirements:** Python 3.11+

Install dependencies:

```bash
pip install -r Streamlit/requirements.txt
```

---

## How to Run the Notebook

1. Open `Final_Project_Dat_Do.ipynb` in Google Colab or Jupyter
2. Run all cells top to bottom
3. The notebook will download the dataset automatically via the raw GitHub URL

---

## How to Launch the Streamlit App Locally

```bash
cd Streamlit
streamlit run app.py
```

The app will open at `http://localhost:8501`. Use the sidebar sliders and dropdowns to adjust the employee profile and see the predicted attrition risk update in real time.

---

## Streamlit Cloud Deployment

The app is deployed at: https://finalproject-datdo-econ3916.streamlit.app/

To redeploy:
1. Push changes to the `main` branch
2. Go to streamlit.io/cloud
3. Select the app and click **Reboot app**

---

## Model Summary

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8605 | 0.6154 | 0.3404 | 0.4384 | 0.8079 |
| Random Forest | 0.8265 | 0.3333 | 0.0851 | 0.1356 | 0.8009 |

**5-Fold CV F1:** Logistic Regression 0.5552 vs Random Forest 0.3044

Logistic Regression is selected as the preferred model. It outperforms Random Forest on all five metrics and in every cross-validation fold.

---

## Reproducibility

All random operations use `random_state=42`. The train/test split uses `test_size=0.20` with `stratify=y` to preserve the 16% attrition rate in both sets.
