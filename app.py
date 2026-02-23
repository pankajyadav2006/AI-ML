import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import os
import sys
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from data_preprocessing import load_data, preprocess
from predict import predict_yield, load_model

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(os.path.join(models_dir, 'best_model.pkl')):
        return None, None, None, None, None, None, None
    model, model_name, encoders, scaler, features = load_model()
    with open(os.path.join(models_dir, 'model_results.json')) as f:
        results = json.load(f)
    log_model = None
    log_path = os.path.join(models_dir, 'logistic_regression.pkl')
    if os.path.exists(log_path):
        log_model = joblib.load(log_path)
    return model, model_name, encoders, scaler, features, results, log_model


st.sidebar.title("🌾 Crop Yield Prediction")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", ["Home", "Data Explorer", "Predict Yield", "Model Performance"])


if page == "Home":
    st.title("🌾 Crop Yield Prediction System")
    st.markdown("""
    This application predicts crop yield using machine learning models trained on 
    historical agricultural data. It uses features like rainfall, temperature, 
    pesticide usage, and crop type to estimate yield in hectograms per hectare (hg/ha).
    """)

    st.subheader("About the Dataset")
    st.markdown("""
    - **Source:** [Kaggle - Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)
    - **Records:** 28,242 entries
    - **Countries:** 101 
    - **Crops:** 10 major crops (Maize, Rice, Wheat, Potatoes, etc.)
    - **Features:** Rainfall, Temperature, Pesticide usage, Year
    - **Target:** Crop yield in hg/ha
    """)

    st.subheader("How it works")
    st.markdown("""
    1. **Data Preprocessing** — Clean data, encode categories, normalize features
    2. **Model Training** — Train ML models (Linear Regression, Decision Tree, Logistic Regression)
    3. **Evaluation** — Compare models using MAE, RMSE, R² and Accuracy metrics
    4. **Prediction** — Use the best model to predict yield for new inputs
    """)

    st.subheader("Sample Data")
    df = get_data()
    st.dataframe(df.head(10))

    st.subheader("Quick Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Countries", df['Area'].nunique())
    col3.metric("Crop Types", df['Item'].nunique())
    col4.metric("Year Range", f"{df['Year'].min()}-{df['Year'].max()}")


elif page == "Data Explorer":
    st.title("📊 Data Explorer")

    use_uploaded = st.checkbox("Upload your own CSV file")

    if use_uploaded:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'Unnamed: 0' in df.columns:
                df.drop(columns=['Unnamed: 0'], inplace=True)
        else:
            st.info("Please upload a file or uncheck the box to use the built-in dataset.")
            st.stop()
    else:
        df = get_data()

    st.subheader("Dataset Info")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(df.describe().round(2))

    st.subheader("Visualizations")

    st.markdown("**Average Yield by Crop Type**")
    avg_by_crop = df.groupby('Item')['hg/ha_yield'].mean().sort_values(ascending=True)
    fig1 = px.bar(
        x=avg_by_crop.values, y=avg_by_crop.index,
        orientation='h', labels={'x': 'Avg Yield (hg/ha)', 'y': 'Crop'},
        color=avg_by_crop.values, color_continuous_scale='Greens'
    )
    fig1.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("**Yield Distribution by Crop**")
    fig2 = px.box(df, x='Item', y='hg/ha_yield', color='Item')
    fig2.update_layout(height=400, showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Average Yield Over Time**")
    yearly = df.groupby('Year')['hg/ha_yield'].mean().reset_index()
    fig3 = px.line(yearly, x='Year', y='hg/ha_yield',
                   labels={'hg/ha_yield': 'Avg Yield (hg/ha)'})
    fig3.update_traces(line_color='green')
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Feature Correlations**")
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    fig4 = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdYlGn', aspect='auto')
    fig4.update_layout(height=450)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("**Top 15 Countries by Average Yield**")
    top = df.groupby('Area')['hg/ha_yield'].mean().sort_values(ascending=False).head(15)
    fig5 = px.bar(x=top.index, y=top.values,
                  labels={'x': 'Country', 'y': 'Avg Yield (hg/ha)'},
                  color=top.values, color_continuous_scale='YlGn')
    fig5.update_layout(height=400, showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig5, use_container_width=True)


elif page == "Predict Yield":
    st.title("🔮 Predict Crop Yield")

    artifacts = get_model()
    if artifacts[0] is None:
        st.error("Model not found! Please train the model first by running:")
        st.code("python src/model_training.py")
        st.stop()

    model, model_name, encoders, scaler, features, results, log_model = artifacts
    df = get_data()

    st.info(f"Using model: **{model_name}** (R² = {results[model_name]['metrics']['R2']})")

    st.subheader("Enter Farm Details")

    col1, col2 = st.columns(2)

    with col1:
        crop = st.selectbox("Crop Type", sorted(encoders['Item'].classes_))
        country = st.selectbox("Country", sorted(encoders['Area'].classes_))
        year = st.slider("Year", 1960, 2030, 2024)

    with col2:
        rainfall = st.number_input("Avg Rainfall (mm/year)",
                                   min_value=0.0, max_value=5000.0,
                                   value=round(float(df['average_rain_fall_mm_per_year'].median()), 1))
        pesticides = st.number_input("Pesticides (tonnes)",
                                     min_value=0.0, max_value=500000.0,
                                     value=round(float(df['pesticides_tonnes'].median()), 1))
        temperature = st.number_input("Avg Temperature (°C)",
                                      min_value=-10.0, max_value=50.0,
                                      value=round(float(df['avg_temp'].median()), 1))

    if st.button("Predict Yield", type="primary"):
        try:
            result = predict_yield(
                area=country, item=crop, year=year,
                rainfall=rainfall, pesticides=pesticides, avg_temp=temperature,
                model=model, encoders=encoders, scaler=scaler, features=features
            )

            st.success(f"Predicted Yield for **{crop}** in **{country}** ({year})")

            col1, col2 = st.columns(2)
            col1.metric("Yield (hg/ha)", f"{result['yield_hg_ha']:,.0f}")
            col2.metric("Yield (tonnes/ha)", f"{result['yield_tonnes_ha']:.2f}")

            if log_model is not None:
                area_enc = encoders['Area'].transform([country])[0]
                item_enc = encoders['Item'].transform([crop])[0]
                input_data = pd.DataFrame(
                    [[area_enc, item_enc, year, rainfall, pesticides, temperature]],
                    columns=features
                )
                input_scaled = scaler.transform(input_data)
                category = log_model.predict(input_scaled)[0]

                if category == 'High':
                    st.success(f"Yield Category (Logistic Regression): **{category}**")
                elif category == 'Medium':
                    st.warning(f"Yield Category (Logistic Regression): **{category}**")
                else:
                    st.error(f"Yield Category (Logistic Regression): **{category}**")

            if result['feature_importance']:
                st.subheader("Feature Importance")
                imp = result['feature_importance']
                imp_sorted = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))

                fig = px.bar(
                    x=list(imp_sorted.values()),
                    y=list(imp_sorted.keys()),
                    orientation='h',
                    labels={'x': 'Importance', 'y': 'Feature'},
                    color=list(imp_sorted.values()),
                    color_continuous_scale='Greens'
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("View Input Summary"):
                st.table(pd.DataFrame({
                    'Parameter': ['Crop', 'Country', 'Year', 'Rainfall (mm)', 'Pesticides (tonnes)', 'Temperature (°C)'],
                    'Value': [crop, country, year, rainfall, pesticides, temperature]
                }))

        except Exception as e:
            st.error(f"Error: {str(e)}")


elif page == "Model Performance":
    st.title("📈 Model Performance")

    artifacts = get_model()
    if artifacts[0] is None:
        st.error("No model results found. Please train models first.")
        st.stop()

    model, model_name, encoders, scaler, features, results, log_model = artifacts

    st.success(f"Best Regression Model: **{model_name}** with R² = {results[model_name]['metrics']['R2']}")

    st.subheader("Regression Models")

    reg_rows = []
    for name, r in results.items():
        if r.get('type') == 'regression':
            m = r['metrics']
            reg_rows.append({
                'Model': name,
                'MAE': m['MAE'],
                'RMSE': m['RMSE'],
                'R² Score': m['R2'],
            })
    reg_df = pd.DataFrame(reg_rows)
    st.dataframe(reg_df, hide_index=True)

    fig1 = px.bar(reg_df, x='Model', y='R² Score', color='R² Score',
                  color_continuous_scale='Greens', text='R² Score')
    fig1.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig1.update_layout(height=350, showlegend=False, title="R² Score Comparison")
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.bar(reg_df, x='Model', y='MAE', color='MAE',
                       color_continuous_scale='Reds', title="MAE Comparison")
        fig2.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = px.bar(reg_df, x='Model', y='RMSE', color='RMSE',
                       color_continuous_scale='Oranges', title="RMSE Comparison")
        fig3.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Classification Model (Logistic Regression)")
    if 'Logistic Regression' in results:
        log_r = results['Logistic Regression']
        st.metric("Accuracy", f"{log_r['metrics']['Accuracy']:.4f}")
        if 'thresholds' in log_r:
            t = log_r['thresholds']
            st.markdown(f"""
            Yield categories:
            - **Low:** ≤ {t['q1']:,.0f} hg/ha
            - **Medium:** {t['q1']:,.0f} – {t['q2']:,.0f} hg/ha
            - **High:** > {t['q2']:,.0f} hg/ha
            """)

    st.subheader("Feature Importance (Best Regression Model)")
    if results[model_name].get('feature_importance'):
        imp = results[model_name]['feature_importance']
        imp_sorted = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))

        fig4 = px.bar(
            x=list(imp_sorted.values()), y=list(imp_sorted.keys()),
            orientation='h', labels={'x': 'Importance', 'y': 'Feature'},
            color=list(imp_sorted.values()), color_continuous_scale='Greens'
        )
        fig4.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    with st.expander("What do these metrics mean?"):
        st.markdown("""
        **Regression metrics:**
        - **MAE:** Average absolute difference between predicted and actual values. Lower = better.
        - **RMSE:** Similar to MAE but penalizes large errors more. Lower = better.
        - **R² Score:** How much variance the model explains (0 to 1). Closer to 1 = better.
        
        **Classification metric:**
        - **Accuracy:** Percentage of correct yield category predictions (Low/Medium/High).
        """)
