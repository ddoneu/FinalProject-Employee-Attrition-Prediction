import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────────
# LOAD & TRAIN (cached so it only runs once)
# ─────────────────────────────────────────────
@st.cache_data
def load_and_train():
    path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
    df = pd.read_csv(path + "/WA_Fn-UseC_-HR-Employee-Attrition.csv")

    # Drop constant / noninformative columns
    drop_cols = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
    df_model = df.drop(columns=drop_cols).copy()

    # Encode target
    df_model['Attrition'] = (df_model['Attrition'] == 'Yes').astype(int)

    # One-hot encode categoricals
    cat_cols = df_model.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

    X = df_encoded.drop(columns=['Attrition'])
    y = df_encoded['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return df, model, scaler, X_train.columns.tolist(), df_encoded

df_raw, model, scaler, feature_cols, df_encoded = load_and_train()

# ─────────────────────────────────────────────
# BOOTSTRAP CI FUNCTION
# ─────────────────────────────────────────────
def bootstrap_prob_ci(input_df, model, scaler, feature_cols, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    probs = []
    x = input_df[feature_cols].values
    for _ in range(n_boot):
        noise = rng.normal(0, 0.01, x.shape)
        x_noisy = x + noise
        x_scaled = scaler.transform(x_noisy)
        probs.append(model.predict_proba(x_scaled)[0][1])
    return np.mean(probs), np.percentile(probs, 2.5), np.percentile(probs, 97.5)

# ─────────────────────────────────────────────
# SIDEBAR — EMPLOYEE PROFILE INPUTS
# ─────────────────────────────────────────────
st.sidebar.header("Employee Profile")
st.sidebar.markdown("Adjust inputs to predict attrition risk.")

age = st.sidebar.slider("Age", 18, 60, 35)
monthly_income = st.sidebar.slider("Monthly Income ($)", 1000, 20000, 5000, step=500)
overtime = st.sidebar.selectbox("Works Overtime?", ["No", "Yes"])
job_satisfaction = st.sidebar.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)
total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 10)
distance_from_home = st.sidebar.slider("Distance from Home (miles)", 1, 29, 10)
num_companies_worked = st.sidebar.slider("Number of Companies Worked", 0, 9, 2)
work_life_balance = st.sidebar.slider("Work-Life Balance (1=Low, 4=High)", 1, 4, 3)
environment_satisfaction = st.sidebar.slider("Environment Satisfaction (1=Low, 4=High)", 1, 4, 3)
job_involvement = st.sidebar.slider("Job Involvement (1=Low, 4=High)", 1, 4, 3)
stock_option_level = st.sidebar.slider("Stock Option Level (0-3)", 0, 3, 1)
years_in_current_role = st.sidebar.slider("Years in Current Role", 0, 18, 3)
years_with_curr_manager = st.sidebar.slider("Years with Current Manager", 0, 17, 3)

# ─────────────────────────────────────────────
# BUILD INPUT ROW MATCHING TRAINING FEATURES
# ─────────────────────────────────────────────
base_row = df_encoded.drop(columns=['Attrition']).iloc[0:1].copy()
base_row[:] = 0

# Fill numeric fields
numeric_map = {
    'Age': age,
    'DailyRate': 800,
    'DistanceFromHome': distance_from_home,
    'Education': 3,
    'EnvironmentSatisfaction': environment_satisfaction,
    'HourlyRate': 66,
    'JobInvolvement': job_involvement,
    'JobLevel': 2,
    'JobSatisfaction': job_satisfaction,
    'MonthlyIncome': monthly_income,
    'MonthlyRate': 14000,
    'NumCompaniesWorked': num_companies_worked,
    'PercentSalaryHike': 14,
    'PerformanceRating': 3,
    'RelationshipSatisfaction': 3,
    'StockOptionLevel': stock_option_level,
    'TotalWorkingYears': total_working_years,
    'TrainingTimesLastYear': 3,
    'WorkLifeBalance': work_life_balance,
    'YearsAtCompany': years_at_company,
    'YearsInCurrentRole': years_in_current_role,
    'YearsSinceLastPromotion': 2,
    'YearsWithCurrManager': years_with_curr_manager,
}

for col, val in numeric_map.items():
    if col in base_row.columns:
        base_row[col] = val

# Overtime encoding
if overtime == "Yes" and 'OverTime_Yes' in base_row.columns:
    base_row['OverTime_Yes'] = 1

# Default categorical encodings (most common values)
defaults = {
    'BusinessTravel_Travel_Frequently': 0,
    'BusinessTravel_Travel_Rarely': 1,
    'Department_Research & Development': 1,
    'Department_Sales': 0,
    'EducationField_Life Sciences': 1,
    'Gender_Male': 1,
    'JobRole_Sales Executive': 0,
    'JobRole_Research Scientist': 1,
    'MaritalStatus_Married': 1,
    'MaritalStatus_Single': 0,
}
for col, val in defaults.items():
    if col in base_row.columns:
        base_row[col] = val

input_df = base_row[feature_cols]

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
input_scaled = scaler.transform(input_df)
prob = model.predict_proba(input_scaled)[0][1]
prediction = model.predict(input_scaled)[0]
prob_mean, ci_lo, ci_hi = bootstrap_prob_ci(input_df, model, scaler, feature_cols)

# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────
st.title("Employee Attrition Risk Predictor")
st.markdown("**ECON 3916 Final Project — Dat Do** | IBM HR Analytics Dataset | Logistic Regression")
st.markdown("---")

# ── ROW 1: Prediction output ──────────────────
col1, col2, col3 = st.columns(3)

risk_label = "High Risk" if prob >= 0.5 else "Low Risk"
risk_color = "red" if prob >= 0.5 else "green"

with col1:
    st.metric("Attrition Probability", f"{prob:.1%}")

with col2:
    st.metric("95% Confidence Interval", f"[{ci_lo:.1%}, {ci_hi:.1%}]")

with col3:
    st.markdown(f"### Risk Level")
    st.markdown(f"<h2 style='color:{risk_color}'>{risk_label}</h2>", unsafe_allow_html=True)

st.markdown("---")

# ── ROW 2: Gauge + Feature contribution ───────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Attrition Probability Gauge")

    fig, ax = plt.subplots(figsize=(6, 3))
    bar_color = '#e74c3c' if prob >= 0.5 else '#2ecc71'
    ci_color = '#c0392b' if prob >= 0.5 else '#27ae60'

    ax.barh(['Risk'], [prob], color=bar_color, height=0.4, label='Predicted probability')
    ax.barh(['Risk'], [ci_hi - ci_lo], left=ci_lo, color=ci_color,
            height=0.15, alpha=0.5, label=f'95% CI [{ci_lo:.1%}, {ci_hi:.1%}]')
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Decision threshold (0.50)')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability of Attrition')
    ax.set_title(f'Predicted Risk: {prob:.1%}', fontweight='bold', fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption(
        "The bar shows the predicted attrition probability. "
        "The shaded band is the 95% bootstrap confidence interval. "
        "The dashed line is the 0.50 decision threshold. "
        "Predictive output only — not causal evidence."
    )

with col_right:
    st.subheader("Top Predictors — This Employee Profile")

    coef_series = pd.Series(model.coef_[0], index=feature_cols)
    input_values = input_df.iloc[0]
    contributions = coef_series * input_values
    top_contrib = contributions.abs().sort_values(ascending=False).head(10)
    top_contrib_signed = contributions[top_contrib.index].sort_values()

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    colors = ['#e74c3c' if v > 0 else '#3498db' for v in top_contrib_signed.values]
    top_contrib_signed.plot.barh(ax=ax2, color=colors, edgecolor='black')
    ax2.axvline(0, color='black', linewidth=1)
    ax2.set_title(
        'Feature Contributions to Prediction\nPredictive association, not causal effect',
        fontweight='bold', fontsize=11
    )
    ax2.set_xlabel('Coefficient × Feature Value')
    ax2.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.caption(
        "Red bars increase predicted attrition risk. Blue bars decrease it. "
        "These reflect the model's prediction logic, not causal effects."
    )

st.markdown("---")

# ── ROW 3: Interactive income vs attrition risk ──
st.subheader("How Monthly Income Affects Attrition Risk (holding other inputs fixed)")

income_range = np.arange(1000, 20500, 500)
probs_income = []

for inc in income_range:
    row = input_df.copy()
    if 'MonthlyIncome' in row.columns:
        row['MonthlyIncome'] = inc
    scaled = scaler.transform(row)
    probs_income.append(model.predict_proba(scaled)[0][1])

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(income_range, probs_income, color='steelblue', linewidth=2)
ax3.axvline(monthly_income, color='salmon', linestyle='--', linewidth=2,
            label=f'Current input: ${monthly_income:,}')
ax3.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, label='Decision threshold (0.50)')
ax3.fill_between(income_range, probs_income, alpha=0.1, color='steelblue')
ax3.set_xlabel('Monthly Income ($)', fontsize=12)
ax3.set_ylabel('Predicted Attrition Probability', fontsize=12)
ax3.set_title(
    'Predicted Attrition Risk vs. Monthly Income\n(all other inputs held at sidebar values)',
    fontweight='bold', fontsize=13
)
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig3)
plt.close()

st.caption(
    "This chart updates dynamically when you change sidebar inputs. "
    "The salmon line marks the current monthly income from the sidebar. "
    "Predictive association only — not causal evidence that raising income reduces attrition."
)

st.markdown("---")

# ── FOOTER ────────────────────────────────────
st.markdown(
    "**Model:** Logistic Regression | **Data:** IBM HR Analytics (Kaggle, N=1,470) | "
    "**Train/Test Split:** 80/20, stratified | **CV F1:** 0.5552 | "
    "**Disclaimer:** Predictive output only. Not for automated employment decisions."
)
