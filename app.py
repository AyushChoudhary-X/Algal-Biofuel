import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model_utils import train_all_models  

# Configure page settings
st.set_page_config(page_title="Lipid Productivity Predictor", page_icon="🧪", layout="wide")

st.title("🧪 Microalgae Lipid Productivity Predictor")
st.markdown("Upload your dataset, train models concurrently, and use the **What-If Predictor** to simulate new experiments.")

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

# Show raw data
with st.expander("🔍 View Raw Dataset", expanded=False):
    st.dataframe(df.head(50))
    st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

st.sidebar.markdown("---")
st.sidebar.header("2. Model Configuration")

# Dynamic target column selector
default_col_index = 0
preferred_target = "OUTCOME High/Low LC, LP, BP"
if preferred_target in df.columns:
    default_col_index = list(df.columns).index(preferred_target)

target_col = st.sidebar.selectbox(
    "Select Target Column", 
    options=df.columns, 
    index=default_col_index
)

available_models = ["Random Forest", "SVM", "Logistic Regression", "KNN", "XGBoost", "ANN"]
selected_models = st.sidebar.multiselect(
    "Select Models to Train", 
    available_models, 
    default=["Random Forest", "Logistic Regression", "XGBoost"]
)

# ================= TRAINING TRIGGER =================
if st.sidebar.button("🚀 Train Models & Compare", type="primary"):
    if not selected_models:
        st.warning("Please select at least one model to train.")
    else:
        with st.spinner("Training models... This might take a few seconds."):
            try:
                # Run the backend training from model_utils.py
                output_data = train_all_models(df, target=target_col, selected_models=selected_models)
                best_model_name = output_data["best_model"]
                best_model_obj = output_data["models"][best_model_name]
                
                st.toast("Training Complete!", icon="✅")
                
                # --- NEW: 4 TABS INSTEAD OF 3 ---
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Model Comparison", 
                    "🔍 Feature Importance", 
                    "📂 Prediction Results", 
                    "🧪 What-If Predictor (Scenario Testing)"
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

                # --- TAB 3: PREDICTION EXPLORER ---
                with tab3:
                    st.subheader("Test Data Predictions")
                    y_true = output_data["y_test"]
                    best_preds = output_data["predictions"][best_model_name]
                    pred_df = pd.DataFrame({"Actual Target": y_true, f"Predicted ({best_model_name})": best_preds})
                    pred_df["Correct Prediction?"] = pred_df["Actual Target"] == pred_df[f"Predicted ({best_model_name})"]
                    st.dataframe(pred_df)

                # --- TAB 4: WHAT-IF PREDICTOR (SCENARIO TESTING) ---
                with tab4:
                    st.subheader("🧪 Interactive Scenario Testing")
                    st.markdown(f"Adjust the environmental and biological parameters below. The app will use the **{best_model_name}** model to predict the outcome.")
                    st.markdown("*(Default values are loaded from the first row of your dataset)*")
                    
                    # 1. Get the first row as defaults
                    X_baseline = df.drop(columns=[target_col])
                    baseline_row = X_baseline.iloc[0].to_dict()
                    
                    # 2. Create a clean 3-column layout for all inputs
                    user_inputs = {}
                    input_cols = st.columns(3)
                    
                    for i, (col_name, default_val) in enumerate(baseline_row.items()):
                        with input_cols[i % 3]:  # Distribute evenly across 3 columns
                            # If it's a number, use a Number Input
                            if pd.api.types.is_numeric_dtype(df[col_name]):
                                # Handle NaNs in the default value
                                safe_default = 0.0 if pd.isna(default_val) else float(default_val)
                                user_inputs[col_name] = st.number_input(
                                    col_name, 
                                    value=safe_default,
                                    format="%.4f"
                                )
                            # If it's categorical/text, use a Select Box
                            else:
                                unique_vals = df[col_name].dropna().unique().tolist()
                                safe_default = "Unknown" if pd.isna(default_val) else default_val
                                if safe_default not in unique_vals:
                                    unique_vals.insert(0, safe_default)
                                user_inputs[col_name] = st.selectbox(
                                    col_name, 
                                    options=unique_vals, 
                                    index=unique_vals.index(safe_default)
                                )
                    
                    st.markdown("---")
                    
                    # 3. Prediction Button Logic
                    if st.button("🔮 Predict Outcome for this Scenario", type="primary", use_container_width=True):
                        # Convert user inputs to a 1-row DataFrame
                        user_df = pd.DataFrame([user_inputs])
                        
                        # Trick to perfectly replicate pd.get_dummies formatting:
                        # We temporarily attach the user row to the original dataframe
                        combined = pd.concat([X_baseline, user_df], ignore_index=True)
                        
                        # Apply identical preprocessing steps
                        cat_cols = combined.select_dtypes(exclude=np.number).columns
                        combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
                        
                        # Extract the newly encoded user row (it's the last row)
                        user_encoded = combined_encoded.iloc[[-1]]
                        
                        # Ensure we only keep features that passed the variance threshold during training
                        final_features = output_data["feature_names"]
                        
                        # Catch missing dummy columns (in case user selected a category not in training)
                        for f in final_features:
                            if f not in user_encoded.columns:
                                user_encoded[f] = 0
                                
                        user_final = user_encoded[final_features]
                        
                        # Apply the trained Scaler
                        user_scaled = output_data["scaler"].transform(user_final)
                        
                        # Make the Prediction
                        pred_value = best_model_obj.predict(user_scaled)[0]
                        
                        # Display nicely
                        if pred_value == 1:
                            st.success("### 🎉 Prediction: **High Lipid Productivity (HLP)**")
                            st.balloons()
                        else:
                            st.error("### 📉 Prediction: **Low Lipid Productivity (LLP)**")

            except Exception as e:
                st.error(f"An error occurred during training or prediction: {str(e)}")