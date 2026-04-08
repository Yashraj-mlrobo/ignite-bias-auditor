import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

# Ensure models directory exists
os.makedirs("models/saved", exist_ok=True)

def preprocess_and_train(df, target_col, protected_cols, dataset_name):
    print(f"--- Training Model for {dataset_name.upper()} ---")
    
    # 1. Separate Features and Target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Handle Categorical Variables (One-Hot Encoding)
    # We must keep the protected columns intact for the audit later
    X_encoded = pd.get_dummies(X, drop_first=True)

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

    # 4. Scale Numerical Features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # 5. Train the Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 6. Evaluate Baseline Accuracy
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Baseline Accuracy: {accuracy:.4f}")

    # 7. Save the Model and Scaler
    joblib.dump(model, f"models/saved/{dataset_name}_model.pkl")
    joblib.dump(scaler, f"models/saved/{dataset_name}_scaler.pkl")
    
    # Save test sets for the Audit and Explain engines
    X_test_scaled.to_csv(f"data/{dataset_name}_X_test.csv", index=False)
    y_test.to_csv(f"data/{dataset_name}_y_test.csv", index=False)
    
    print(f"Model and artifacts saved for {dataset_name}!\n")
    return model, X_test_scaled, y_test

# NOTE FOR CAPTAIN: Run this script directly to generate the .pkl files.
# Example usage (assuming df_adult is loaded):
# preprocess_and_train(df_adult, target_col='target', protected_cols=['sex', 'race'], dataset_name='adult')