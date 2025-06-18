import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import pickle

# Load synthetic dataset
df = pd.read_csv("kidney_cleaned.csv")

# Clean 'class' column
df['class'] = df['class'].replace([np.inf, -np.inf], np.nan)
df['class'] = pd.to_numeric(df['class'], errors='coerce')
df = df.dropna(subset=['class'])
print("Class distribution after cleaning:")
print(df['class'].value_counts())

# Separate features and target
X = df.drop(columns=['class'])
y = df['class'].astype(int)

# Handle Missing Values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining set distribution (before SMOTE):")
print(y_train.value_counts())
print("Test set distribution:")
print(y_test.value_counts())

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("\nTraining set distribution (after SMOTE):")
print(y_train.value_counts())

# Normalize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Model
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
rf.fit(X_train_scaled, y_train)

# Predictions
y_pred_rf = rf.predict(X_test_scaled)

# Evaluate Model
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Classification Report:")
    print(classification_report(y_true, y_pred))

evaluate_model("Random Forest", y_test, y_pred_rf)

# Save the trained model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(rf, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Random Forest model saved as 'model.pkl' and scaler saved as 'scaler.pkl'.")

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()