import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import plotly.graph_objects as go

# Load data
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Data Cleaning
df = df.dropna()  # Remove rows with missing values
df = df[(df['hypertension'].isin([0, 1])) & (df['heart_disease'].isin([0, 1]))]
df = df[(df['age'] >= 0) & (df['age'] <= 120)]
df = df[(df['bmi'] >= 10) & (df['bmi'] <= 50)]
df = df[(df['HbA1c_level'] >= 4) & (df['HbA1c_level'] <= 15)]
df = df[(df['blood_glucose_level'] >= 50) & (df['blood_glucose_level'] <= 300)]

# Simplify smoking history into "Has Smoked" and "Didn't Smoked"
df = df[df['smoking_history'] != 'No Info']  # Exclude "No Info"
has_smoked = ['current', 'former', 'even']
didnt_smoked = ['never', 'not current']
df['smoking_history'] = df['smoking_history'].apply(lambda x: 'Has Smoked' if x in has_smoked else "Didn't Smoked")

# Filter gender to exclude "Other"
df = df[df['gender'] != 'Other']

# Streamlit App
st.title("Predicting Diabetes Risk Leveraging Health Indicators")
st.write("""
This dashboard analyzes health indicators to predict diabetes risk. It includes data visualizations, 
a prediction model using logistic regression, and a forecasting tool. Use the sidebar to interact 
with the dashboard and explore your own diabetes risk.
""")

# Data Overview
st.header("Dataset Overview")
st.write(f"Original Data Shape: {df.shape}")
st.write("Sample of Cleaned Data:", df.head())

# Interactive Data Table
st.header("Explore the Data")
age_range = st.slider("Filter by Age Range", 0, 120, (0, 120))
df_filtered = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
st.write("Filtered Data:", df_filtered)

# Visualizations
st.header("Data Visualizations")
st.write("Explore the distribution and relationships in the data interactively.")

# Interactive Pie Chart: Gender Distribution
st.subheader("Gender Distribution")
gender_counts = df_filtered['gender'].value_counts().reset_index()
gender_counts.columns = ['gender', 'count']
fig1 = px.pie(gender_counts, values='count', names='gender', title="Gender Distribution")
st.plotly_chart(fig1, use_container_width=True)

# Interactive Pie Chart: Smoking History (Updated to Has Smoked/Didn't Smoked)
st.subheader("Smoking History Distribution")
smoking_counts = df_filtered['smoking_history'].value_counts().reset_index()
smoking_counts.columns = ['smoking_history', 'count']
fig2 = px.pie(smoking_counts, values='count', names='smoking_history', title="Smoking History Distribution (Has Smoked vs Didn't Smoked)")
st.plotly_chart(fig2, use_container_width=True)

# Interactive Histogram: Age Distribution
st.subheader("Age Distribution")
fig3 = px.histogram(df_filtered, x='age', nbins=20, title="Age Distribution", marginal="rug")
fig3.update_traces(opacity=0.75)
fig3.update_layout(bargap=0.1)
st.plotly_chart(fig3, use_container_width=True)

# Updated Chart: Histogram for BMI vs Diabetes with grouped bins
st.subheader("BMI Distribution by Diabetes Status")
fig4 = px.histogram(df_filtered, x='bmi', color='diabetes', nbins=8,  # 8 bins for BMI range 10-50 with step 5
                    title="BMI Distribution by Diabetes Status",
                    labels={'bmi': 'BMI', 'diabetes': 'Diabetes (0 = No, 1 = Yes)'},
                    opacity=0.5, histnorm='probability density')
fig4.update_layout(bargap=0.1, barmode='overlay')
st.plotly_chart(fig4, use_container_width=True)

# Updated Chart: Heatmap for HbA1c vs Blood Glucose by Diabetes Status with adjustable bins
st.subheader("HbA1c vs Blood Glucose by Diabetes Status")
# Sliders for bin sizes
hba1c_bin_size = st.slider("Select HbA1c Bin Size", min_value=2, max_value=5, value=2, step=1, key='hba1c_bin_size')
glucose_bin_size = st.slider("Select Blood Glucose Bin Size", min_value=10, max_value=50, value=50, step=10, key='glucose_bin_size')

# Compute HbA1c bins dynamically
hba1c_min, hba1c_max = 4, 15
hba1c_bins = np.arange(hba1c_min, hba1c_max + hba1c_bin_size, hba1c_bin_size)
hba1c_labels = [f"{int(start)}-{int(start + hba1c_bin_size)}" for start in hba1c_bins[:-1]]
df_filtered['HbA1c_bin'] = pd.cut(df_filtered['HbA1c_level'], bins=hba1c_bins, labels=hba1c_labels, include_lowest=True)

# Compute Blood Glucose bins dynamically
glucose_min, glucose_max = 50, 300
glucose_bins = np.arange(glucose_min, glucose_max + glucose_bin_size, glucose_bin_size)
glucose_labels = [f"{int(start)}-{int(start + glucose_bin_size)}" for start in glucose_bins[:-1]]
df_filtered['Glucose_bin'] = pd.cut(df_filtered['blood_glucose_level'], bins=glucose_bins, labels=glucose_labels, include_lowest=True)

# Create pivot table for heatmap with observed=True
heatmap_data = df_filtered.groupby(['HbA1c_bin', 'Glucose_bin'], observed=True).size().unstack(fill_value=0)
fig5 = px.imshow(heatmap_data, text_auto=True, aspect="auto",
                 title="HbA1c vs Blood Glucose by Diabetes Status (Count of Cases)",
                 labels={'x': 'Blood Glucose Level (mg/dL)', 'y': 'HbA1c Level (%)', 'color': 'Count'},
                 color_continuous_scale='Viridis')
fig5.update_traces(textfont=dict(size=12))  # Increase text size for visibility
fig5.update_layout(xaxis={'side': 'top'}, margin=dict(l=50, r=50, t=100, b=50))
st.plotly_chart(fig5, use_container_width=True)

# Combined Bar Chart: Diabetes Status by Smoking History and Health Conditions (Dynamic 100% Stacked Bar Chart)
st.subheader("Diabetes Status by Smoking History and Health Conditions")
# Categorize health conditions, explicitly including "No" states
def categorize_health_conditions(row):
    hypertension_status = "Hypertension" if row['hypertension'] == 1 else "No Hypertension"
    heart_disease_status = "Heart Disease" if row['heart_disease'] == 1 else "No Heart Disease"
    return f"{hypertension_status} + {heart_disease_status}"

df_filtered['Health_Conditions'] = df_filtered.apply(categorize_health_conditions, axis=1)

# Create a combined category for Smoking History and Health Conditions
df_filtered['Combined_Category'] = df_filtered['smoking_history'] + " + " + df_filtered['Health_Conditions']

# Calculate counts and percentages for the combined category
combined_counts = df_filtered.groupby(['Combined_Category', 'diabetes']).size().unstack(fill_value=0)
combined_totals = combined_counts.sum(axis=1)
combined_percentages = (combined_counts.T / combined_totals * 100).T.round(1)

# Get all possible combinations for the dropdown
all_combinations = combined_percentages.index.tolist()

# Multi-select dropdown to choose which combinations to display
selected_combinations = st.multiselect(
    "Select Smoking History and Health Condition Combinations to Display",
    options=all_combinations,
    default=all_combinations  # Default to showing all combinations
)

# Filter the data based on selected combinations
if not selected_combinations:
    st.write("Please select at least one combination to display the chart.")
else:
    filtered_percentages = combined_percentages.loc[selected_combinations]

    # Prepare data for 100% stacked bar chart
    categories = filtered_percentages.index.tolist()
    diabetes_0 = filtered_percentages[0].tolist()  # Percentage for diabetes = 0 (No)
    diabetes_1 = filtered_percentages[1].tolist()  # Percentage for diabetes = 1 (Yes)

    # Create text for each bar segment
    text_0 = [f"{pct}%" for pct in diabetes_0]
    text_1 = [f"{pct}%" for pct in diabetes_1]

    # Create 100% stacked bar chart using go.Figure and go.Bar
    fig_combined = go.Figure()

    fig_combined.add_trace(go.Bar(
        x=categories,
        y=diabetes_0,
        name='Diabetes (0 = No)',
        marker_color='lightblue',
        text=text_0,
        textposition='auto',
        textfont=dict(size=14)
    ))

    fig_combined.add_trace(go.Bar(
        x=categories,
        y=diabetes_1,
        name='Diabetes (1 = Yes)',
        marker_color='blue',
        text=text_1,
        textposition='auto',
        textfont=dict(size=14)
    ))

    # Update layout for 100% stacked bar chart
    fig_combined.update_layout(
        barmode='stack',
        title="Diabetes Status by Smoking History and Health Conditions",
        xaxis=dict(title="Smoking History + Health Conditions", tickangle=45),
        yaxis=dict(title="Percentage (%)", range=[0, 100]),  # Ensure y-axis goes from 0 to 100%
        height=600,
        bargap=0.2,
        bargroupgap=0.1,
        width=800  # Increase overall width of the chart
    )

    # Update bar width (closer to 1 makes bars wider)
    fig_combined.update_traces(width=0.6)

    st.plotly_chart(fig_combined, use_container_width=True)

# Prediction Model
st.header("Diabetes Risk Prediction")
st.write("""
Enter your health indicators in the sidebar to predict your diabetes risk using a logistic regression model.
Logistic regression is well-suited for binary outcomes like diabetes prediction.
""")

# Prepare data for modeling
df_encoded = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)
X = df_encoded.drop('diabetes', axis=1)
y = df_encoded['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Sidebar: User Input for Prediction
st.sidebar.header("Predict Your Diabetes Risk")
age_input = st.sidebar.slider("Age", 0, 120, 50)
hypertension_input = st.sidebar.radio("Hypertension", ["No", "Yes"])
heart_disease_input = st.sidebar.radio("Heart Disease", ["No", "Yes"])
bmi_input = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
hba1c_input = st.sidebar.slider("HbA1c Level (%)", 4.0, 15.0, 5.5)
glucose_input = st.sidebar.slider("Blood Glucose Level (mg/dL)", 50, 300, 100)
gender_input = st.sidebar.selectbox("Gender", df['gender'].unique())
smoking_input = st.sidebar.selectbox("Smoking History", ["Has Smoked", "Didn't Smoked"])  # Updated options

# Map user input to model input
hypertension_input = 1 if hypertension_input == "Yes" else 0
heart_disease_input = 1 if heart_disease_input == "Yes" else 0

# Create user input DataFrame
user_data = pd.DataFrame({
    'age': [age_input],
    'hypertension': [hypertension_input],
    'heart_disease': [heart_disease_input],
    'bmi': [bmi_input],
    'HbA1c_level': [hba1c_input],
    'blood_glucose_level': [glucose_input],
    'gender': [gender_input],
    'smoking_history': [smoking_input]
})
user_data_encoded = pd.get_dummies(user_data, columns=['gender', 'smoking_history'], drop_first=True)
for col in X_train.columns:
    if col not in user_data_encoded.columns:
        user_data_encoded[col] = 0
user_data_encoded = user_data_encoded[X_train.columns]
prediction_prob = model.predict_proba(user_data_encoded)[0][1]  # Probability of diabetes
prediction_binary = "Yes" if prediction_prob >= 0.5 else "No"

st.write(f"**Predicted Diabetes Risk Probability:** {prediction_prob:.2f}")
st.write(f"**Diabetes Prediction:** {prediction_binary}")

# Forecasting
st.header("Diabetes Risk Forecast Over Age")
st.write("""
This section forecasts how diabetes risk changes with age, keeping other health indicators constant. 
The chart below is a scatter plot with a regression line showing the trend, and each point is labeled with its probability.
""")

# Sidebar: Forecasting Inputs
st.sidebar.header("Forecast Settings")
hypertension_forecast = st.sidebar.radio("Hypertension (Forecast)", ["No", "Yes"], key='hypertension_forecast')
heart_disease_forecast = st.sidebar.radio("Heart Disease (Forecast)", ["No", "Yes"], key='heart_disease_forecast')
bmi_forecast = st.sidebar.slider("BMI (Forecast)", 10.0, 50.0, 25.0, key='bmi_forecast')
hba1c_forecast = st.sidebar.slider("HbA1c Level (Forecast)", 4.0, 15.0, 5.5, key='hba1c_forecast')
glucose_forecast = st.sidebar.slider("Blood Glucose (Forecast)", 50, 300, 100, key='glucose_forecast')
gender_forecast = st.sidebar.selectbox("Gender (Forecast)", df['gender'].unique(), key='gender_forecast')
smoking_forecast = st.sidebar.selectbox("Smoking History (Forecast)", ["Has Smoked", "Didn't Smoked"], key='smoking_forecast')  # Updated options

# Map forecast inputs
hypertension_forecast = 1 if hypertension_forecast == "Yes" else 0
heart_disease_forecast = 1 if heart_disease_forecast == "Yes" else 0

# Generate forecast data
ages = np.arange(20, 81, 5)
forecast_data = pd.DataFrame({
    'age': ages,
    'hypertension': [hypertension_forecast] * len(ages),
    'heart_disease': [heart_disease_forecast] * len(ages),
    'bmi': [bmi_forecast] * len(ages),
    'HbA1c_level': [hba1c_forecast] * len(ages),
    'blood_glucose_level': [glucose_forecast] * len(ages),
    'gender': [gender_forecast] * len(ages),
    'smoking_history': [smoking_forecast] * len(ages)
})
forecast_data_encoded = pd.get_dummies(forecast_data, columns=['gender', 'smoking_history'], drop_first=True)
for col in X_train.columns:
    if col not in forecast_data_encoded.columns:
        forecast_data_encoded[col] = 0
forecast_data_encoded = forecast_data_encoded[X_train.columns]
forecast_predictions = model.predict_proba(forecast_data_encoded)[:, 1]  # Probability of diabetes

# Compute regression line
z = np.polyfit(ages, forecast_predictions, 1)
p = np.poly1d(z)
regression_line = p(ages)

# Scatter Plot with Regression Line and Labels for Forecast
fig6 = go.Figure()
fig6.add_trace(go.Scatter(x=ages, y=forecast_predictions, mode='markers', name='Risk Probability',
                          marker=dict(size=10)))
fig6.add_trace(go.Scatter(x=ages, y=regression_line, mode='lines', name='Regression Line',
                          line=dict(color='red')))
for i, (age, prob) in enumerate(zip(ages, forecast_predictions)):
    fig6.add_annotation(x=age, y=prob, text=f"{prob:.2f}", showarrow=True, yshift=10)
fig6.update_layout(title="Diabetes Risk Forecast Over Age",
                   xaxis_title="Age",
                   yaxis_title="Predicted Diabetes Risk Probability",
                   showlegend=True)
st.plotly_chart(fig6, use_container_width=True)

# Footer
st.write("""
**Notes:** 
- Data cleaning removed rows with implausible values (e.g., age > 120, BMI < 10).
- Logistic regression is used for binary classification, providing probability scores for diabetes risk.
- The forecast chart updates based on your sidebar inputs.
""")
