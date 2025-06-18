import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Preprocessing function for training data
def preprocess_data(df):
    if 'id' in df.columns:
        df = df.drop(columns=['id'])  # Drop 'id' column if present

    df.replace(to_replace=r'^\s*\?$', value=np.nan, regex=True, inplace=True)
    df = df.ffill()  # Forward-fill missing values

    label_encoders = {}
    categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
    
    for col in categorical_columns:
        df[col] = df[col].astype(str).str.strip()
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoders for future use

    numeric_cols = df.select_dtypes(include=['object']).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    imputer = SimpleImputer(strategy="mean")
    df[df.columns] = imputer.fit_transform(df)

    X = df.drop(columns=['classification'])
    y = df['classification']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save feature names and scaler for later use
    joblib.dump(X.columns.tolist(), "models/feature_names.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    return X_scaled, y, scaler

# Preprocessing function for Flask input
def preprocess_input(input_data):
    # Load feature names and scaler
    feature_names = joblib.load("models/feature_names.pkl")
    scaler = joblib.load("models/scaler.pkl")

    # Convert input data into a DataFrame
    if isinstance(input_data, dict):  
        input_df = pd.DataFrame([input_data])  # Convert dictionary to DataFrame
    elif isinstance(input_data, list) or isinstance(input_data, np.ndarray):  
        input_df = pd.DataFrame(input_data, columns=feature_names)  # Handle list/array case
    else:  
        raise ValueError("Invalid input format: Expected dict, list, or NumPy array")

    # Handle missing and extra features
    missing_features = set(feature_names) - set(input_df.columns)
    extra_features = set(input_df.columns) - set(feature_names)

    print("Missing Features:", missing_features)
    print("Extra Features:", extra_features)

    for feature in missing_features:
        input_df[feature] = 0  # Assign default value

    # Ensure column order is correct
    input_df = input_df[feature_names]

    # Ensure correct shape
    print("Shape before scaling:", input_df.shape)

    # Scale input
    scaled_input = scaler.transform(input_df)

    print("Shape after scaling:", scaled_input.shape)

    # ✅ FIX: Reshape to ensure (1, num_features) before returning
    return scaled_input.reshape(1, -1)  # Ensure it's 2D
