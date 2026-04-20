# ECON 3916 — Predicting Employee Attrition

**Author:** Dat Do  
**Course:** ECON 3916 — Statistical Machine Learning for Economics (Prof. Piao)  
**Date:** April 2026

## Project Overview

Can we predict whether an employee will voluntarily leave an organization based on their demographic, compensation, and job-satisfaction characteristics? This is a prediction task, not causal inference. The stakeholder is an HR director deciding where to allocate retention resources.

## Dataset

- **Source:** [IBM HR Analytics Employee Attrition & Performance (Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Size:** 1,470 employees × 35 features
- **Target:** `Attrition` (Yes/No, ~16% positive)

No manual download needed — the notebook loads data directly via `kagglehub`.

## How to Reproduce

1. Open `Final_Project_Dat_Do.ipynb` in Google Colab
2. Run all cells — dependencies install automatically (`kagglehub`)
3. Dataset downloads on first run via Kaggle API

## Checkpoint Contents (April 19)

- **Proposal:** Prediction question, prediction vs. causation distinction, dataset details, stakeholder
- **EDA:** Data types, missing data assessment (MCAR/MAR/MNAR), 5 visualizations with interpretations, data quality summary
- **Baseline Model:** Logistic Regression (80/20 stratified split, `random_state=42`, StandardScaler), metrics reported (accuracy, precision, recall, F1, confusion matrix)

## Final Submission Plan (April 26)

- Second model (Random Forest or Gradient Boosting)
- Cross-validation with confidence intervals
- Streamlit dashboard with interactive predictions
- 5-page SCR report (PDF)
- AI Methodology Appendix (P.R.I.M.E. framework)
