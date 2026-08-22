# FactoryLens | Streamlit Dashboard
# Manufacturing Operations Analytics Platform

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="FactoryLens",
    page_icon="🏭",
    layout="wide"
)

# ----------------------------------------
# Load Data and Model
# ----------------------------------------
@st.cache_data
def load_data():
    path = r"C:/Users/yashp/Downloads/MyFactoryLens/MyFactorylens_dataset.csv"
    return pd.read_csv(path)

@st.cache_resource
def load_model():
    path = r"C:/Users/yashp/Downloads/MyFactoryLens/factorylens_model.pkl"
    return joblib.load(path)

@st.cache_data
def load_features():
    path = r"C:/Users/yashp/Downloads/MyFactoryLens/factorylens_features.pkl"
    return joblib.load(path)

df = load_data()
model = load_model()
feature_names = load_features()

print("All files loaded successfully")

# ----------------------------------------
# Header
# ----------------------------------------
st.title("🏭 FactoryLens")
st.markdown("**Manufacturing Operations Analytics Platform**")
st.markdown("---")

# ----------------------------------------
# KPI Cards — Section 1
# ----------------------------------------
st.subheader("Operations Overview")

# Calculate KPI values
total_machines = len(df)
overall_failure_rate = df['machine_failure'].mean() * 100
worst_shift = df.groupby('shift')['machine_failure'].mean().idxmax()
critical_pct = (df['machine_age_category'] == 'Critical').mean() * 100

# Display in 4 columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Observations",
        value=f"{total_machines:,}"
    )

with col2:
    st.metric(
        label="Overall Failure Rate",
        value=f"{overall_failure_rate:.2f}%"
    )

with col3:
    st.metric(
        label="Highest Risk Shift",
        value=worst_shift
    )

with col4:
    st.metric(
        label="Machines in Critical State",
        value=f"{critical_pct:.1f}%"
    )

st.markdown("---")
# ----------------------------------------
# Business Insights Explorer — Section 2
# ----------------------------------------
st.subheader("Business Insights Explorer")
st.markdown("Select a business question to explore:")

# Dropdown menu
question = st.selectbox(
    label="Business Question",
    options=[
        "Which shift has the highest failure rate?",
        "Does tool wear predict failure risk?",
        "Which product type fails most?",
        "How does operator experience affect failure rates?",
        "What temperature conditions are highest risk?",
        "Which machine age category needs most attention?"
    ]
)

# ----------------------------------------
# Chart rendering based on selection
# ----------------------------------------

if question == "Which shift has the highest failure rate?":
    shift_failure = df.groupby('shift')['machine_failure'].mean() * 100
    shift_failure = shift_failure.reset_index()
    shift_failure.columns = ['Shift', 'Failure Rate (%)']
    shift_failure = shift_failure.sort_values('Failure Rate (%)', 
                                              ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x='Shift', y='Failure Rate (%)',
                data=shift_failure, hue='Shift',
                palette=['#e74c3c', '#e67e22', '#2ecc71'],
                legend=False, ax=ax)
    for i, row in shift_failure.reset_index(drop=True).iterrows():
        ax.text(i, row['Failure Rate (%)'] + 0.05,
                f"{row['Failure Rate (%)']:.2f}%",
                ha='center', fontsize=11)
    ax.set_title("Which Shift Has the Highest Failure Rate?")
    ax.set_xlabel("Shift")
    ax.set_ylabel("Failure Rate (%)")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("**Finding:** Evening shift has the highest failure rate at 3.66%.")

elif question == "Does tool wear predict failure risk?":
    df['failure_label'] = df['machine_failure'].map(
        {0: 'No Failure', 1: 'Failure'})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='failure_label', y='tool_wear_min',
                data=df, hue='failure_label',
                palette={'No Failure': '#2ecc71', 
                         'Failure': '#e74c3c'},
                legend=False, ax=ax)
    ax.set_title("Does Tool Wear Predict Failure Risk?")
    ax.set_xlabel("Machine Status")
    ax.set_ylabel("Tool Wear (minutes)")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("**Finding:** Failed machines have median tool wear "
                "58 minutes higher than non-failures.")

elif question == "Which product type fails most?":
    product_failure = df.groupby('product_type')['machine_failure'].mean() * 100
    product_failure = product_failure.reset_index()
    product_failure.columns = ['Product Type', 'Failure Rate (%)']
    product_failure = product_failure.sort_values('Failure Rate (%)', 
                                                   ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x='Product Type', y='Failure Rate (%)',
                data=product_failure, hue='Product Type',
                palette=['#e74c3c', '#e67e22', '#2ecc71'],
                legend=False, ax=ax)
    for i, row in product_failure.reset_index(drop=True).iterrows():
        ax.text(i, row['Failure Rate (%)'] + 0.05,
                f"{row['Failure Rate (%)']:.2f}%",
                ha='center', fontsize=11)
    ax.set_title("Which Product Type Fails Most?")
    ax.set_xlabel("Product Type")
    ax.set_ylabel("Failure Rate (%)")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("**Finding:** L type products fail at 3.92% — "
                "nearly double H type at 2.09%.")

elif question == "How does operator experience affect failure rates?":
    exp_failure = df.groupby('operator_experience')['machine_failure'].mean() * 100
    exp_failure = exp_failure.reset_index()
    exp_failure.columns = ['Operator Experience', 'Failure Rate (%)']
    exp_failure = exp_failure.sort_values('Failure Rate (%)', 
                                           ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x='Operator Experience', y='Failure Rate (%)',
                data=exp_failure, hue='Operator Experience',
                palette=['#e74c3c', '#e67e22', '#2ecc71'],
                legend=False, ax=ax)
    for i, row in exp_failure.reset_index(drop=True).iterrows():
        ax.text(i, row['Failure Rate (%)'] + 0.05,
                f"{row['Failure Rate (%)']:.2f}%",
                ha='center', fontsize=11)
    ax.set_title("How Does Operator Experience Affect Failure Rates?")
    ax.set_xlabel("Operator Experience")
    ax.set_ylabel("Failure Rate (%)")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("**Finding:** Junior operators have highest failure rate "
                "at 3.75%. Experience alone doesn't fully explain failures.")

elif question == "What temperature conditions are highest risk?":
    df['temp_diff'] = df['process_temp_k'] - df['air_temp_k']
    df['failure_label'] = df['machine_failure'].map(
        {0: 'No Failure', 1: 'Failure'})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='failure_label', y='temp_diff',
                data=df, hue='failure_label',
                palette={'No Failure': '#2ecc71', 
                         'Failure': '#e74c3c'},
                legend=False, ax=ax)
    ax.set_title("What Temperature Conditions Are Highest Risk?")
    ax.set_xlabel("Machine Status")
    ax.set_ylabel("Temperature Difference (K)")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("**Finding:** Lower temperature difference correlates "
                "with failure — poor heat dissipation is the risk signal.")

elif question == "Which machine age category needs most attention?":
    category_order = ['New', 'Mid-life', 'Aging', 'Critical']
    age_failure = df.groupby('machine_age_category',
                             observed=True)['machine_failure'].mean() * 100
    age_failure = age_failure.reset_index()
    age_failure.columns = ['Machine Age Category', 'Failure Rate (%)']
    age_failure['Machine Age Category'] = pd.Categorical(
        age_failure['Machine Age Category'],
        categories=category_order, ordered=True)
    age_failure = age_failure.sort_values('Machine Age Category')
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x='Machine Age Category', y='Failure Rate (%)',
                data=age_failure, hue='Machine Age Category',
                palette=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'],
                legend=False, order=category_order, ax=ax)
    for i, row in age_failure.reset_index(drop=True).iterrows():
        ax.text(i, row['Failure Rate (%)'] + 0.05,
                f"{row['Failure Rate (%)']:.2f}%",
                ha='center', fontsize=11)
    ax.set_title("Which Machine Age Category Needs Most Attention?")
    ax.set_xlabel("Machine Age Category")
    ax.set_ylabel("Failure Rate (%)")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("**Finding:** Critical machines fail at 15.49% — "
                "5x higher than all other categories.")

st.markdown("---")



# ----------------------------------------
# Failure Risk Predictor — Section 3
# ----------------------------------------
st.subheader("Failure Risk Predictor")
st.markdown("Adjust machine parameters to predict failure risk:")

# Split into two columns
left_col, right_col = st.columns(2)

# ----------------------------------------
# Left Column — Input Sliders
# ----------------------------------------
with left_col:
    st.markdown("**Machine Parameters**")

    air_temp = st.slider(
        "Air Temperature (K)",
        min_value=295.0, max_value=305.0,
        value=300.0, step=0.1
    )

    process_temp = st.slider(
        "Process Temperature (K)",
        min_value=305.0, max_value=315.0,
        value=310.0, step=0.1
    )

    rot_speed = st.slider(
        "Rotational Speed (RPM)",
        min_value=1168, max_value=2886,
        value=1500, step=10
    )

    torque = st.slider(
        "Torque (Nm)",
        min_value=3.8, max_value=76.6,
        value=40.0, step=0.1
    )

    tool_wear = st.slider(
        "Tool Wear (minutes)",
        min_value=0, max_value=253,
        value=100, step=1
    )

    product_type = st.selectbox(
        "Product Type",
        options=['L', 'M', 'H']
    )

    shift = st.selectbox(
        "Shift",
        options=['Morning', 'Evening', 'Night']
    )

    operator_exp = st.selectbox(
        "Operator Experience",
        options=['Junior', 'Senior', 'Expert']
    )

# ----------------------------------------
# Right Column — Prediction Output
# ----------------------------------------
with right_col:
    st.markdown("**Prediction Results**")
    
    # Engineer features from inputs
    power_kw = (torque * rot_speed) / 9549
    temp_diff = process_temp - air_temp

    # Tool wear category number
    if tool_wear <= 100:
        wear_cat_num = 0
    elif tool_wear <= 160:
        wear_cat_num = 1
    elif tool_wear <= 200:
        wear_cat_num = 2
    else:
        wear_cat_num = 3

    # Build input row as dictionary
    input_dict = {
        'air_temp_k': air_temp,
        'process_temp_k': process_temp,
        'rotational_speed_rpm': rot_speed,
        'torque_nm': torque,
        'tool_wear_min': tool_wear,
        'power_kw': power_kw,
        'temp_diff': temp_diff,
        'tool_wear_category_num': wear_cat_num,
        'product_type_H': 1 if product_type == 'H' else 0,
        'product_type_L': 1 if product_type == 'L' else 0,
        'product_type_M': 1 if product_type == 'M' else 0,
        'shift_Evening': 1 if shift == 'Evening' else 0,
        'shift_Morning': 1 if shift == 'Morning' else 0,
        'shift_Night': 1 if shift == 'Night' else 0,
        'operator_experience_Expert': 1 if operator_exp == 'Expert' else 0,
        'operator_experience_Junior': 1 if operator_exp == 'Junior' else 0,
        'operator_experience_Senior': 1 if operator_exp == 'Senior' else 0
    }

    # Convert to DataFrame in correct column order
    input_df = pd.DataFrame([input_dict])[feature_names]
    
    # Get failure probability
    failure_prob = model.predict_proba(input_df)[0][1]
    failure_pct = failure_prob * 100
    
    # Display probability
    st.metric(
        label="Failure Probability",
        value=f"{failure_pct:.1f}%"
    )

    # Risk card and recommendation
    if failure_prob > 0.35:
        st.error("🔴 HIGH RISK")
        st.markdown("**Recommended Action:**")
        st.markdown("Schedule immediate inspection within 24 hours.")
    elif failure_prob > 0.15:
        st.warning("🟡 MEDIUM RISK")
        st.markdown("**Recommended Action:**")
        st.markdown("Monitor closely — inspect within 72 hours.")

    else:
        st.success("🟢 LOW RISK")
        st.markdown("**Recommended Action:**")
        st.markdown("Continue production — next scheduled maintenance.")

    # Show engineered features calculated
    st.markdown("---")
    st.markdown("**Derived Parameters**")
    st.markdown(f"Power (kW): **{power_kw:.2f}**")
    st.markdown(f"Temp Difference (K): **{temp_diff:.2f}**")
    st.markdown(f"Machine State: "
                f"**{['New','Mid-life','Aging','Critical'][wear_cat_num]}**")