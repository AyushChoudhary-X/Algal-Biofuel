import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model_utils import train_all_models  

# Configure page settings
st.set_page_config(page_title="Lipid Productivity Predictor", page_icon="🧪", layout="wide")

st.title("🧪 Microalgae Lipid Productivity Predictor")
st.markdown("Upload your dataset, train models concurrently, and use the **What-If Predictor** to simulate new experiments.")

# --- INITIALIZE APP MEMORY (SESSION STATE) ---
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False
    st.session_state.output_data = None

# ================= SIDEBAR & DATA INPUT =================
st.sidebar.header("1. Data Input & Setup")
uploaded_file = st.sidebar.file_uploader("Upload your Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("👈 Please upload your CSV file in the sidebar to begin. (e.g., '1-s2.0-S...xlsx - Sheet1.csv')")
    st.stop()

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    return df

df = load_data(uploaded_file)

with st.expander("🔍 View Raw Dataset", expanded=False):
    st.dataframe(df.head(50))
    st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

st.sidebar.markdown("---")
st.sidebar.header("2. Model Configuration")

default_col_index = 0
preferred_target = "OUTCOME High/Low LC, LP, BP"
if preferred_target in df.columns:
    default_col_index = list(df.columns).index(preferred_target)

target_col = st.sidebar.selectbox("Select Target Column", options=df.columns, index=default_col_index)

available_models = ["Voting Ensemble (Super Model)", "Random Forest", "SVM", "Logistic Regression", "KNN", "XGBoost", "ANN"]
selected_models = st.sidebar.multiselect("Select Models to Train", available_models, default=["Voting Ensemble (Super Model)", "Random Forest", "XGBoost"])

# ================= TRAINING TRIGGER =================
if st.sidebar.button("🚀 Train Models & Compare", type="primary"):
    if not selected_models:
        st.sidebar.warning("Please select at least one model to train.")
    else:
        with st.spinner("Training models... This might take a few seconds."):
            try:
                # Run the backend training and SAVE to session state
                st.session_state.output_data = train_all_models(df, target=target_col, selected_models=selected_models)
                st.session_state.is_trained = True
                st.sidebar.success("Training Complete!")
            except Exception as e:
                st.sidebar.error(f"An error occurred: {str(e)}")

# ================= RENDER RESULTS (ONLY IF TRAINED) =================
if st.session_state.is_trained:
    output_data = st.session_state.output_data
    best_model_name = output_data["best_model"]
    best_model_obj = output_data["models"][best_model_name]
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Model Comparison", 
        "🔍 Feature Importance", 
        "🧠 SHAP Analysis",
        "📂 Prediction Results", 
        "🧪 What-If Predictor"
    ])
    
    # --- TAB 1: MODEL COMPARISON ---
    with tab1:
        st.subheader("Performance Metrics")
        results_df = pd.DataFrame(output_data["results"])
        st.dataframe(results_df.style.highlight_max(axis=0, subset=['accuracy', 'f1_score'], color='lightgreen'), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_acc = px.bar(results_df, x='name', y='accuracy', color='name', title="Accuracy", text_auto='.3f')
            st.plotly_chart(fig_acc, use_container_width=True)
        with col2:
            fig_f1 = px.bar(results_df, x='name', y='f1_score', color='name', title="F1-Score", text_auto='.3f')
            st.plotly_chart(fig_f1, use_container_width=True)

    # --- TAB 2: FEATURE IMPORTANCE ---
    with tab2:
        st.write(f"🏆 The best performing model overall is: **{best_model_name}**")
        feature_names = output_data["feature_names"]
        importances = None
        
        if hasattr(best_model_obj, "feature_importances_"):
            importances = best_model_obj.feature_importances_
        elif hasattr(best_model_obj, "coef_"):
            importances = np.abs(best_model_obj.coef_[0])
        
        if importances is not None:
            feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False).head(15)
            fig_feat = px.bar(feat_df, x="Importance", y="Feature", orientation='h', color="Importance", color_continuous_scale="Viridis")
            fig_feat.update_layout(yaxis={'categoryorder':'total ascending'}, height=550, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_feat, use_container_width=True)
        else:
            st.info("Feature importance not supported for this model type.")

    # --- TAB 3: SHAP ANALYSIS (ADVANCED EXPLAINABILITY) ---
    with tab3:
        st.subheader("🧠 Advanced AI Explainability (SHAP)")
        st.write("While standard Feature Importance tells us *which* variables matter, SHAP tells us *how* they matter.")
        
        if best_model_name in ["Random Forest", "XGBoost"]:
            with st.spinner("Calculating SHAP values..."):
                import shap
                import matplotlib.pyplot as plt
                
                # Get raw data and scale it so the model can read it
                X_raw = output_data["X"]
                scaler = output_data["scaler"]
                X_scaled = scaler.transform(X_raw)
                
                # Calculate SHAP values using TreeExplainer
                explainer = shap.TreeExplainer(best_model_obj)
                shap_values = explainer.shap_values(X_scaled)
                
                # RF returns a list of arrays (one for each class), XGBoost returns one array
                if isinstance(shap_values, list):
                    shap_to_plot = shap_values[1]  # Index 1 is the positive class (HLP)
                else:
                    shap_to_plot = shap_values
                
                # Generate Plotly-friendly matplotlib chart
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_to_plot, X_raw, show=False)
                st.pyplot(fig)
                
                st.info("""
                **📖 How to read this chart:**
                * **X-axis (Impact):** Dots pushed to the right mean those specific conditions drove the prediction toward **High Lipid Productivity (HLP)**. Dots to the left drove it toward **Low (LLP)**.
                * **Color (Value):** Red dots represent high actual values for that feature. Blue dots represent low values. 
                * *Example:* If 'Temperature' has a cluster of red dots on the far left, it means high temperatures are actively destroying lipid productivity!
                """)
        else:
            st.warning("⚠️ SHAP visualizer in this app is currently optimized for Tree-based models. Please ensure **Random Forest** or **XGBoost** is your top-performing model to view this tab.")

    # --- TAB 4: PREDICTION EXPLORER ---
    with tab4:
        st.subheader("Test Data Predictions")
        y_true = output_data["y_test"]
        best_preds = output_data["predictions"][best_model_name]
        pred_df = pd.DataFrame({"Actual Target": y_true, f"Predicted ({best_model_name})": best_preds})
        pred_df["Correct Prediction?"] = pred_df["Actual Target"] == pred_df[f"Predicted ({best_model_name})"]
        st.dataframe(pred_df)

    # --- TAB 5: WHAT-IF PREDICTOR (SCENARIO TESTING) ---
    with tab5:
        st.subheader("🧪 Interactive Scenario Testing")
        st.markdown(f"Adjust the parameters below. The app will use the **{best_model_name}** model to predict the outcome.")
        
        X_baseline = df.drop(columns=[target_col])
        baseline_row = X_baseline.iloc[0].to_dict()
        user_inputs = {}
        
        with st.form("what_if_form"):
            input_cols = st.columns(3)
            for i, (col_name, default_val) in enumerate(baseline_row.items()):
                with input_cols[i % 3]:
                    if pd.api.types.is_numeric_dtype(df[col_name]):
                        safe_default = 0.0 if pd.isna(default_val) else float(default_val)
                        user_inputs[col_name] = st.number_input(col_name, value=safe_default, format="%.4f")
                    else:
                        unique_vals = df[col_name].dropna().unique().tolist()
                        safe_default = "Unknown" if pd.isna(default_val) else default_val
                        if safe_default not in unique_vals:
                            unique_vals.insert(0, safe_default)
                        user_inputs[col_name] = st.selectbox(col_name, options=unique_vals, index=unique_vals.index(safe_default))
            
            st.markdown("---")
            submitted = st.form_submit_button("🔮 Predict Outcome for this Scenario", type="primary", use_container_width=True)
            
        if submitted:
            user_df = pd.DataFrame([user_inputs])
            combined = pd.concat([X_baseline, user_df], ignore_index=True)
            
            cat_cols = combined.select_dtypes(exclude=np.number).columns
            combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
            user_encoded = combined_encoded.iloc[[-1]]
            
            final_features = output_data["feature_names"]
            for f in final_features:
                if f not in user_encoded.columns:
                    user_encoded[f] = 0
                    
            user_final = user_encoded[final_features]
            user_scaled = output_data["scaler"].transform(user_final)
            pred_value = best_model_obj.predict(user_scaled)[0]
            
            if pred_value == 1:
                st.success("### 🎉 Prediction: **High Lipid Productivity (HLP)**")
                st.balloons()
            else:
                st.error("### 📉 Prediction: **Low Lipid Productivity (LLP)**")