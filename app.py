import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model_utils import train_all_models  # Importing from your existing script

# Configure page settings
st.set_page_config(page_title="Lipid Productivity Predictor", page_icon="🧪", layout="wide")

st.title("🧪 Microalgae Lipid Productivity Predictor & Model Comparison")
st.markdown("Upload your dataset, train multiple machine learning models concurrently to predict **High/Low Lipid Productivity (HLP vs LLP)**, and analyze the most influential variables.")

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
target_col = st.sidebar.text_input("Target Column Name", "OUTCOME High/Low LC, LP, BP")

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
    elif target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in the dataset! Please check the spelling.")
    else:
        with st.spinner("Training models... This might take a few seconds."):
            # Call your backend logic
            try:
                output_data = train_all_models(df, target=target_col, selected_models=selected_models)
                st.toast("Training Complete!", icon="✅")
                
                # Setup Tabs for clean UI
                tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "🔍 Feature Importance", "📂 Prediction Results"])
                
                # --- TAB 1: MODEL COMPARISON ---
                with tab1:
                    st.subheader("Performance Metrics")
                    results_df = pd.DataFrame(output_data["results"])
                    
                    # Highlight highest values
                    st.dataframe(
                        results_df.style.highlight_max(axis=0, subset=['accuracy', 'f1_score'], color='lightgreen'),
                        use_container_width=True
                    )
                    
                    st.markdown("---")
                    st.subheader("Graphical Comparison")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_acc = px.bar(results_df, x='name', y='accuracy', color='name', 
                                         title="Accuracy by Model", text_auto='.3f')
                        fig_acc.update_layout(showlegend=False)
                        st.plotly_chart(fig_acc, use_container_width=True)
                        
                    with col2:
                        fig_f1 = px.bar(results_df, x='name', y='f1_score', color='name', 
                                        title="F1-Score by Model", text_auto='.3f')
                        fig_f1.update_layout(showlegend=False)
                        st.plotly_chart(fig_f1, use_container_width=True)

                # --- TAB 2: FEATURE IMPORTANCE ---
                with tab2:
                    best_model_name = output_data["best_model"]
                    st.write(f"🏆 The best performing model overall is: **{best_model_name}**")
                    
                    best_model_obj = output_data["models"][best_model_name]
                    feature_names = output_data["feature_names"]
                    
                    importances = None
                    # Tree based models
                    if hasattr(best_model_obj, "feature_importances_"):
                        importances = best_model_obj.feature_importances_
                    # Linear models
                    elif hasattr(best_model_obj, "coef_"):
                        importances = np.abs(best_model_obj.coef_[0])
                    
                    if importances is not None:
                        st.markdown("### Top 15 Most Important Variables")
                        st.caption("These are the inputs that have the strongest influence on predicting High vs Low Lipid Productivity.")
                        
                        feat_df = pd.DataFrame({
                            "Feature": feature_names,
                            "Importance": importances
                        }).sort_values(by="Importance", ascending=False).head(15)
                        
                        fig_feat = px.bar(
                            feat_df, x="Importance", y="Feature", orientation='h',
                            color="Importance", color_continuous_scale="Viridis"
                        )
                        fig_feat.update_layout(
                            yaxis={'categoryorder':'total ascending'},
                            height=550,
                            margin=dict(t=30, b=0, l=0, r=0)
                            )
                        st.plotly_chart(fig_feat, use_container_width=True)
                    else:
                        st.info(f"Feature importance visualization is not natively supported for {best_model_name} (e.g., KNN/ANN). Train Random Forest or XGBoost to view feature weights.")

                # --- TAB 3: PREDICTION EXPLORER ---
                with tab3:
                    st.subheader("Test Data Predictions")
                    st.write("Compare the actual ground-truth values against what the best model predicted.")
                    
                    # Format test predictions nicely
                    y_true = output_data["y_test"]
                    best_preds = output_data["predictions"][best_model_name]
                    
                    pred_df = pd.DataFrame({
                        "Actual Target": y_true,
                        f"Predicted ({best_model_name})": best_preds
                    })
                    pred_df["Correct Prediction?"] = pred_df["Actual Target"] == pred_df[f"Predicted ({best_model_name})"]
                    
                    st.dataframe(pred_df)
                    
                    # Allow user to download results
                    csv = pred_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Prediction Results (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")
            
            except Exception as e:
                st.error(f"An error occurred during training: {str(e)}")