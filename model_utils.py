from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import VarianceThreshold

import pandas as pd
import numpy as np

# ================= TRAINING =================
def train_all_models(df, target, selected_models):

    # 1. FILTER FOR ONLY ONE METRIC (Lipid Productivity - LP)
    df = df.dropna(subset=[target]).copy()
    df = df[df[target].isin(['HLP', 'LLP'])] 
    
    # 2. BINARY ENCODING
    y = df[target].map({'HLP': 1, 'LLP': 0})
    X_full = df.drop(columns=[target])

    # 3. HANDLE MISSING VALUES & CATEGORICAL DATA
    num_cols = X_full.select_dtypes(include=np.number).columns
    cat_cols = X_full.select_dtypes(exclude=np.number).columns

    X_full[num_cols] = X_full[num_cols].fillna(X_full[num_cols].mean())
    X_full[cat_cols] = X_full[cat_cols].fillna('Unknown')

    X_encoded = pd.get_dummies(X_full, columns=cat_cols, drop_first=True)

    # 4. VARIANCE FILTER
    selector = VarianceThreshold(threshold=0.01)
    X_var = selector.fit_transform(X_encoded)

    selected_features = X_encoded.columns[selector.get_support()]
    X = pd.DataFrame(X_var, columns=selected_features)

    # 5. SPLIT 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. SCALING
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. CLASSIFICATION MODELS 
    all_models = {
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_split=2, min_samples_leaf=2, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "ANN": MLPClassifier(max_iter=500, random_state=42)
    }

    models = {k: all_models[k] for k in selected_models}

    results = []
    trained_models = {}

    # 8. TRAIN LOOP
    predictions = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)

        predictions[name] = pred

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        results.append({
            "name": name,
            "accuracy": acc, 
            "f1_score": f1   
        })

        trained_models[name] = model

    # 9. BEST MODEL
    best_model = max(results, key=lambda x: x["accuracy"])["name"]

    return {
        "models": trained_models,
        "results": results,
        "best_model": best_model,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "X": X ,# <-- Added this back so predict/optimize don't crash
        "y_test": y_test.reset_index(drop=True),
        "predictions": predictions
    }

# ================= PREDICTION =================
def predict_best(input_dict, data):
    model = data["models"][data["best_model"]]
    scaler = data["scaler"]
    X = data["X"]
    features = data["feature_names"]

    df = pd.DataFrame([input_dict])

    # Ensure correct column order
    df = df.reindex(columns=features)

    # Fill missing values
    for col in df.columns:
        if df[col].iloc[0] == "" or pd.isna(df[col].iloc[0]):
            df[col] = X[col].mean()
        else:
            df[col] = float(df[col])

    df_scaled = scaler.transform(df)
    
    # Returns the PROBABILITY of High Lipid Productivity (Class 1)
    if hasattr(model, "predict_proba"):
        return round(model.predict_proba(df_scaled)[0][1], 4)
    else:
        return int(model.predict(df_scaled)[0])

# ================= OPTIMIZATION =================
def optimize_inputs(data, mode="max", n_iter=1000):
    model = data["models"][data["best_model"]]
    scaler = data["scaler"]
    X = data["X"]
    features = data["feature_names"]

    if mode == "max":
        best_output = -np.inf
    else:
        best_output = np.inf

    best_input = None

    for _ in range(n_iter):
        sample = []
        for col in features:
            val = np.random.uniform(X[col].min(), X[col].max())
            sample.append(val)

        sample_array = np.array(sample).reshape(1, -1)
        sample_scaled = scaler.transform(sample_array)
        
        # Optimize based on the probability of getting "High LP"
        if hasattr(model, "predict_proba"):
            pred = model.predict_proba(sample_scaled)[0][1]
        else:
            pred = int(model.predict(sample_scaled)[0])

        if mode == "max":
            if pred > best_output:
                best_output = pred
                best_input = sample
        else:
            if pred < best_output:
                best_output = pred
                best_input = sample

    best_input_dict = dict(zip(features, best_input))
    return best_input_dict, round(best_output, 4)

# ================= LOCAL TESTING BLOCK =================
if __name__ == "__main__":
    import pandas as pd

    csv_filename = "1-s2.0-S0960148125015654-mmc2.xlsx - Sheet1.csv"
    
    try:
        df = pd.read_csv(csv_filename)
        df.columns = df.columns.str.strip() 
        target_col = "OUTCOME High/Low LC, LP, BP"
        test_models = ["Random Forest", "Logistic Regression", "XGBoost"]
        
        print("Training models...")
        output_data = train_all_models(df, target=target_col, selected_models=test_models)
        
        print(f"\n✅ Best Model: {output_data['best_model']}")
        for res in output_data['results']:
            print(f"  -> {res['name']}: Accuracy = {res['accuracy'] * 100:.2f}%")
            
    except Exception as e:
        print(f"❌ Error: {e}")