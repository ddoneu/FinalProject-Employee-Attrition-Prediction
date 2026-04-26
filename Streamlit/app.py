import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Risk Dashboard",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────────
# LOAD & TRAIN (cached — runs once on startup)
# ─────────────────────────────────────────────
DATA_URL = "https://raw.githubusercontent.com/ddoneu/FinalProject-Employee-Attrition-Prediction/refs/heads/main/Data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

@st.cache_data
def load_and_train():
    df = pd.read_csv(DATA_URL)

    # Drop constant / noninformative columns
    drop_cols = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
    df_model = df.drop(columns=drop_cols).copy()

    # Encode target: Yes → 1, No → 0
    df_model['Attrition'] = (df_model['Attrition'] == 'Yes').astype(int)

    # One-hot encode categorical features
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

    return df, model, scaler, X_train.columns.tolist(), df_encoded, X_train

df_raw, model, scaler, feature_cols, df_encoded, X_train = load_and_train()

# ─────────────────────────────────────────────
# BOOTSTRAP CI (resamples training data)
# Uses 200 iterations for Streamlit performance
# ─────────────────────────────────────────────
@st.cache_data
def compute_bootstrap_models(n_boot=200, seed=42):
    """Fit n_boot logistic regression models on bootstrap samples of training data."""
    rng = np.random.default_rng(seed)
    boot_models = []
    X_arr = X_train.values
    y_arr = df_encoded['Attrition'].iloc[:X_train.shape[0]].values

    # Re-get y_train aligned to X_train
    drop_cols = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
    df_model = df_raw.drop(columns=drop_cols).copy()
    df_model['Attrition'] = (df_model['Attrition'] == 'Yes').astype(int)
    cat_cols = df_model.select_dtypes(include=['object']).columns.tolist()
    df_enc = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)
    X_full = df_enc.drop(columns=['Attrition'])
    y_full = df_enc['Attrition']
    from sklearn.model_selection import train_test_split as tts
    Xtr, _, ytr, _ = tts(X_full, y_full, test_size=0.20, random_state=42, stratify=y_full)

    for _ in range(n_boot):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        Xb = Xtr.values[idx]
        yb = ytr.values[idx]
        sc = StandardScaler()
        Xb_scaled = sc.fit_transform(Xb)
        m = LogisticRegression(random_state=42, max_iter=500)
        try:
            m.fit(Xb_scaled, yb)
            boot_models.append((m, sc))
        except Exception:
            continue

    return boot_models

boot_models = compute_bootstrap_models()

def bootstrap_prob_ci(input_df, boot_models, feature_cols):
    probs = []
    x = input_df[feature_cols].values
    for m, sc in boot_models:
        try:
            x_scaled = sc.transform(x)
            probs.append(m.predict_proba(x_scaled)[0][1])
        except Exception:
            continue
    if len(probs) == 0:
        return 0.0, 0.0, 0.0
    return np.mean(probs), np.percentile(probs, 2.5), np.percentile(probs, 97.5)

# ─────────────────────────────────────────────
# RISK LABEL FUNCTION (3-tier)
# Low < 20%, Medium 20-40%, High > 40%
# ─────────────────────────────────────────────
def risk_label_color(prob):
    if prob < 0.20:
        return "Low Risk", "green"
    elif prob < 0.40:
        return "Medium Risk", "orange"
    else:
        return "High Risk", "red"

# ─────────────────────────────────────────────
# SIDEBAR — EMPLOYEE PROFILE INPUTS
# ─────────────────────────────────────────────
st.sidebar.header("Employee Profile")
st.sidebar.markdown("Adjust the inputs to predict attrition risk for a hypothetical employee.")

age                     = st.sidebar.slider("Age", 18, 60, 35)
monthly_income          = st.sidebar.slider("Monthly Income ($)", 1000, 20000, 5000, step=500)
overtime                = st.sidebar.selectbox("Works Overtime?", ["No", "Yes"])
job_satisfaction        = st.sidebar.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
environment_satisfaction= st.sidebar.slider("Environment Satisfaction (1=Low, 4=High)", 1, 4, 3)
total_working_years     = st.sidebar.slider("Total Working Years", 0, 40, 10)
years_at_company        = st.sidebar.slider("Years at Company", 0, 40, 5)
distance_from_home      = st.sidebar.slider("Distance from Home (miles)", 1, 29, 10)
job_role                = st.sidebar.selectbox("Job Role", [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Manager",
    "Sales Representative", "Research Director", "Human Resources"
])
business_travel         = st.sidebar.selectbox("Business Travel", [
    "Non-Travel", "Travel_Rarely", "Travel_Frequently"
])
marital_status          = st.sidebar.selectbox("Marital Status", [
    "Single", "Married", "Divorced"
])
num_companies_worked    = st.sidebar.slider("Number of Companies Worked", 0, 9, 2)
work_life_balance       = st.sidebar.slider("Work-Life Balance (1=Low, 4=High)", 1, 4, 3)
job_involvement         = st.sidebar.slider("Job Involvement (1=Low, 4=High)", 1, 4, 3)
stock_option_level      = st.sidebar.slider("Stock Option Level (0-3)", 0, 3, 1)
years_in_current_role   = st.sidebar.slider("Years in Current Role", 0, 18, 3)
years_with_curr_manager = st.sidebar.slider("Years with Current Manager", 0, 17, 3)

# ─────────────────────────────────────────────
# BUILD INPUT ROW MATCHING TRAINING FEATURES
# ─────────────────────────────────────────────
def build_input_row(
    age, monthly_income, overtime, job_satisfaction, environment_satisfaction,
    total_working_years, years_at_company, distance_from_home,
    job_role, business_travel, marital_status,
    num_companies_worked, work_life_balance, job_involvement,
    stock_option_level, years_in_current_role, years_with_curr_manager,
    df_encoded, feature_cols
):
    base_row = df_encoded.drop(columns=['Attrition']).iloc[0:1].copy()
    base_row[:] = 0

    # Numeric features
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

    # Overtime
    if overtime == "Yes" and 'OverTime_Yes' in base_row.columns:
        base_row['OverTime_Yes'] = 1

    # Business travel
    bt_col = f'BusinessTravel_{business_travel}'
    if bt_col in base_row.columns:
        base_row[bt_col] = 1

    # Job role
    jr_col = f'JobRole_{job_role}'
    if jr_col in base_row.columns:
        base_row[jr_col] = 1

    # Marital status
    ms_col = f'MaritalStatus_{marital_status}'
    if ms_col in base_row.columns:
        base_row[ms_col] = 1

    # Default department and education field
    if 'Department_Research & Development' in base_row.columns:
        base_row['Department_Research & Development'] = 1
    if 'EducationField_Life Sciences' in base_row.columns:
        base_row['EducationField_Life Sciences'] = 1
    if 'Gender_Male' in base_row.columns:
        base_row['Gender_Male'] = 1

    return base_row[feature_cols]

input_df = build_input_row(
    age, monthly_income, overtime, job_satisfaction, environment_satisfaction,
    total_working_years, years_at_company, distance_from_home,
    job_role, business_travel, marital_status,
    num_companies_worked, work_life_balance, job_involvement,
    stock_option_level, years_in_current_role, years_with_curr_manager,
    df_encoded, feature_cols
)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
input_scaled = scaler.transform(input_df)
prob = model.predict_proba(input_scaled)[0][1]
prob_mean, ci_lo, ci_hi = bootstrap_prob_ci(input_df, boot_models, feature_cols)
label, color = risk_label_color(prob)

# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────
st.title("Employee Attrition Risk Dashboard")
st.markdown(
    "**ECON 3916 Final Project — Dat Do** | "
    "IBM HR Analytics Dataset | Logistic Regression | "
    "⚠️ Predictive output only — not for automated employment decisions."
)
st.markdown("---")

# ── ROW 1: Key metrics ───────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Attrition Probability", f"{prob:.1%}")
with col2:
    st.metric("95% Uncertainty Range", f"[{ci_lo:.1%}, {ci_hi:.1%}]")
with col3:
    st.markdown("### Risk Level")
    st.markdown(f"<h2 style='color:{color}'>{label}</h2>", unsafe_allow_html=True)

st.caption(
    "Thresholds: Low < 20% | Medium 20–40% | High > 40%. "
    "The uncertainty range is approximate and reflects model instability "
    "from 200 bootstrap resamples of the training data — "
    "not a guarantee about any individual employee."
)

st.markdown("---")

# ── ROW 2: Gauge + Feature contributions ─────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Predicted Risk Gauge")

    fig, ax = plt.subplots(figsize=(6, 3))
    bar_color = {'green': '#2ecc71', 'orange': '#f39c12', 'red': '#e74c3c'}[color]
    ci_color  = {'green': '#27ae60', 'orange': '#d68910', 'red': '#c0392b'}[color]

    ax.barh(['Risk'], [prob], color=bar_color, height=0.4,
            label=f'Predicted: {prob:.1%}')
    ax.barh(['Risk'], [ci_hi - ci_lo], left=ci_lo, color=ci_color,
            height=0.15, alpha=0.5,
            label=f'95% range [{ci_lo:.1%}, {ci_hi:.1%}]')
    ax.axvline(0.20, color='gold',  linestyle=':', linewidth=1.5, label='Low/Medium (20%)')
    ax.axvline(0.40, color='gray',  linestyle='--', linewidth=1.5, label='Medium/High (40%)')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability of Attrition')
    ax.set_title(f'Predicted Risk: {prob:.1%} — {label}', fontweight='bold', fontsize=13)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption(
        "The bar shows the point estimate. "
        "The shaded band is the 95% bootstrap uncertainty range. "
        "Dashed lines mark the Low/Medium/High thresholds."
    )

with col_right:
    st.subheader("Top Predictors — This Profile")

    coef_series = pd.Series(model.coef_[0], index=feature_cols)
    contributions = coef_series * input_df.iloc[0]
    top_idx = contributions.abs().sort_values(ascending=False).head(10).index
    top_contrib = contributions[top_idx].sort_values()

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    colors2 = ['#e74c3c' if v > 0 else '#3498db' for v in top_contrib.values]
    top_contrib.plot.barh(ax=ax2, color=colors2, edgecolor='black')
    ax2.axvline(0, color='black', linewidth=1)
    ax2.set_title(
        'Feature Contributions to Prediction\n⚠️ Predictive association, not causal effect',
        fontweight='bold', fontsize=11
    )
    ax2.set_xlabel('Coefficient × Feature Value')
    ax2.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.caption(
        "Red = increases predicted risk. Blue = decreases predicted risk. "
        "These reflect the model's logic, not causal effects."
    )

st.markdown("---")

# ── ROW 3: Income sweep (interactive) ────────
st.subheader("How Monthly Income Affects Attrition Risk")
st.markdown("All other sidebar inputs are held fixed. The chart updates when you change any input.")

income_range = np.arange(1000, 20500, 500)
probs_income = []
for inc in income_range:
    row = input_df.copy()
    if 'MonthlyIncome' in row.columns:
        row['MonthlyIncome'] = inc
    probs_income.append(model.predict_proba(scaler.transform(row))[0][1])

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(income_range, probs_income, color='steelblue', linewidth=2)
ax3.axvline(monthly_income, color='salmon', linestyle='--', linewidth=2,
            label=f'Current: ${monthly_income:,}')
ax3.axhline(0.40, color='gray',  linestyle='--', linewidth=1, label='High threshold (40%)')
ax3.axhline(0.20, color='gold',  linestyle=':',  linewidth=1, label='Medium threshold (20%)')
ax3.fill_between(income_range, probs_income, alpha=0.1, color='steelblue')
ax3.set_xlabel('Monthly Income ($)', fontsize=12)
ax3.set_ylabel('Predicted Attrition Probability', fontsize=12)
ax3.set_title(
    'Predicted Attrition Risk vs. Monthly Income\n(other inputs fixed at sidebar values)',
    fontweight='bold', fontsize=13
)
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig3)
plt.close()

st.caption(
    "Predictive association only. This does not prove that raising income reduces attrition."
)

st.markdown("---")

# ── ROW 4: Overtime comparison (interactive) ──
st.subheader("Overtime vs. No Overtime — Predicted Risk Comparison")
st.markdown("Holding all other sidebar inputs fixed.")

# Build two rows: one with OT=No, one with OT=Yes
row_no_ot  = build_input_row(
    age, monthly_income, "No", job_satisfaction, environment_satisfaction,
    total_working_years, years_at_company, distance_from_home,
    job_role, business_travel, marital_status,
    num_companies_worked, work_life_balance, job_involvement,
    stock_option_level, years_in_current_role, years_with_curr_manager,
    df_encoded, feature_cols
)
row_yes_ot = build_input_row(
    age, monthly_income, "Yes", job_satisfaction, environment_satisfaction,
    total_working_years, years_at_company, distance_from_home,
    job_role, business_travel, marital_status,
    num_companies_worked, work_life_balance, job_involvement,
    stock_option_level, years_in_current_role, years_with_curr_manager,
    df_encoded, feature_cols
)

prob_no_ot  = model.predict_proba(scaler.transform(row_no_ot))[0][1]
prob_yes_ot = model.predict_proba(scaler.transform(row_yes_ot))[0][1]

fig4, ax4 = plt.subplots(figsize=(6, 4))
bars = ax4.bar(
    ['No Overtime', 'Overtime'],
    [prob_no_ot, prob_yes_ot],
    color=['#3498db', '#e74c3c'],
    edgecolor='black',
    width=0.4
)
for bar, val in zip(bars, [prob_no_ot, prob_yes_ot]):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f'{val:.1%}', ha='center', fontweight='bold', fontsize=12)

ax4.axhline(0.40, color='gray', linestyle='--', linewidth=1, label='High threshold (40%)')
ax4.axhline(0.20, color='gold', linestyle=':',  linewidth=1, label='Medium threshold (20%)')
ax4.set_ylim(0, 1)
ax4.set_ylabel('Predicted Attrition Probability')
ax4.set_title(
    'Overtime vs. No Overtime — Predicted Risk\n(other inputs fixed at sidebar values)',
    fontweight='bold', fontsize=12
)
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)
plt.tight_layout()
st.pyplot(fig4)
plt.close()

st.caption(
    "This chart updates when you change sidebar inputs. "
    "Predictive association only — not causal evidence that eliminating overtime reduces attrition."
)

st.markdown("---")

# ── FOOTER ────────────────────────────────────
st.markdown(
    "**Model:** Logistic Regression | "
    "**Data:** IBM HR Analytics (Kaggle, N=1,470) | "
    "**Split:** 80/20 stratified, random_state=42 | "
    "**CV F1:** 0.5552 | "
    "**Uncertainty:** 200-sample bootstrap | "
    "⚠️ **This tool is for HR risk-screening only. Not for automated employment decisions.**"
)
