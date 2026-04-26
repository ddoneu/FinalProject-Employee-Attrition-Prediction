# P.R.I.M.E Prompt — Streamlit Dashboard

## Prep
I am completing the Streamlit dashboard requirement for my ECON 3916 final project. My project predicts employee attrition using the IBM HR Analytics Employee Attrition dataset. In my notebook, I compared Logistic Regression and Random Forest. Logistic Regression is my selected model because it performed better on held-out test metrics and 5-fold cross-validated F1-score.

The Streamlit dashboard must satisfy these assignment requirements:
- Deployed to Streamlit Community Cloud
- Parameter controls: sliders, dropdowns, or other input widgets that let users adjust model inputs
- At least 1 interactive visualization that updates dynamically based on user inputs
- Prediction output with uncertainty: point estimate plus confidence interval or prediction interval
- Submit the permanent Streamlit URL

## Request
Act as an expert Python and Streamlit developer helping me create a clean final-project dashboard.

Please write a complete `app.py` file for my employee attrition prediction project. The app should:

1. Load the IBM HR Analytics Employee Attrition dataset using either a local CSV file or KaggleHub.
2. Preprocess the data the same way as my notebook:
   - Drop non-informative columns: `EmployeeCount`, `StandardHours`, `Over18`, and `EmployeeNumber`
   - Encode target `Attrition` as 1 for Yes and 0 for No
   - One-hot encode categorical variables with `drop_first=True`
   - Train/test split using `random_state=42` and `stratify=y`
   - Standardize predictors for Logistic Regression
   - Train Logistic Regression as the selected model
3. Create sidebar parameter controls for a hypothetical employee, including at minimum:
   - Age
   - MonthlyIncome
   - OverTime
   - JobSatisfaction
   - EnvironmentSatisfaction
   - TotalWorkingYears
   - YearsAtCompany
   - DistanceFromHome
   - JobRole
   - BusinessTravel
   - MaritalStatus
4. Output the predicted attrition risk as a percentage.
5. Output an approximate uncertainty range around the predicted risk. Make the uncertainty explanation honest and simple. Do not overclaim statistical precision.
6. Include at least one interactive visualization that changes when the user changes inputs. For example:
   - a bar chart comparing predicted risk under OverTime = Yes vs OverTime = No, holding other inputs fixed, or
   - a line chart showing predicted attrition risk as MonthlyIncome changes, holding other inputs fixed.
7. Include clear text explaining:
   - This is a prediction model, not causal inference.
   - Feature effects are predictive associations, not proof that changing one variable will cause attrition to fall.
   - The dashboard should be used for HR risk-screening and prioritization, not automatic employment decisions.
8. Use clean formatting, section headers, and readable labels.
9. Include comments in the code so I understand what each section does.
10. Also provide a matching `requirements.txt`.

## Iterate
After you generate the first version, I may ask you to revise it if:
- the app does not run locally,
- Streamlit Community Cloud gives a deployment error,
- the dashboard does not meet the assignment requirements,
- the visualization does not update correctly,
- or the prediction output does not match my notebook logic.

## Mechanism Check
Before finalizing the code, explain how I should verify that the app works correctly. Include checks such as:
- running `streamlit run app.py` locally,
- confirming the sidebar inputs update the predicted risk,
- confirming the visualization changes when inputs change,
- confirming the model uses the same preprocessing logic as the notebook,
- confirming the deployed URL works in an incognito browser.

## Evaluate
Prioritize correctness, reproducibility, and satisfying the rubric over making the app overly complex. Keep the dashboard simple but professional. The goal is to earn full credit for the Streamlit requirement.
