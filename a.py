import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Total rows
n_total = 5000
n_ckd = 3000  # ~60% CKD
n_non_ckd = 2000  # ~40% non-CKD

# Function to generate synthetic data based on CKD/non-CKD stats
def generate_synthetic_data(n_samples, is_ckd):
    data = {}
    
    # Numerical features with means and std devs based on your data
    data['age'] = np.random.normal(54.73 if is_ckd else 46.38, 15, n_samples).clip(1, 100)
    data['bp'] = np.random.normal(82.98 if is_ckd else 73.33, 10, n_samples).clip(50, 120)
    data['sg'] = np.random.normal(1.013 if is_ckd else 1.022, 0.005, n_samples).clip(1.005, 1.030)
    data['al'] = np.random.normal(2.02 if is_ckd else 0.03, 1.5, n_samples).clip(0, 5).round()
    data['su'] = np.random.normal(0.74 if is_ckd else 0.03, 0.5, n_samples).clip(0, 5).round()
    data['bgr'] = np.random.normal(179.56 if is_ckd else 113.46, 50, n_samples).clip(70, 500)
    data['bu'] = np.random.normal(77.28 if is_ckd else 34.94, 20, n_samples).clip(10, 150)
    data['sc'] = np.random.normal(4.47 if is_ckd else 1.02, 2, n_samples).clip(0.5, 25)
    data['sod'] = np.random.normal(135.27 if is_ckd else 141.31, 5, n_samples).clip(110, 150)
    data['pot'] = np.random.normal(4.65 if is_ckd else 4.38, 0.5, n_samples).clip(2.5, 6.0)
    data['hemo'] = np.random.normal(10.62 if is_ckd else 15.05, 2, n_samples).clip(5, 18)
    data['pcv'] = np.random.normal(22.47 if is_ckd else 33.28, 5, n_samples).clip(10, 50).round()
    data['wbcc'] = np.random.normal(59.23 if is_ckd else 58.36, 15, n_samples).clip(20, 100).round()
    data['rbcc'] = np.random.normal(23.31 if is_ckd else 32.34, 5, n_samples).clip(10, 50).round()
    
    # Binary features with prevalence rates
    data['rbc'] = np.random.binomial(1, 0.84 if is_ckd else 1.00, n_samples)
    data['pc'] = np.random.binomial(1, 0.65 if is_ckd else 0.99, n_samples)
    data['pcc'] = np.random.binomial(1, 0.22 if is_ckd else 0.01, n_samples)
    data['ba'] = np.random.binomial(1, 0.12 if is_ckd else 0.01, n_samples)
    data['htn'] = np.random.binomial(1, 0.84 if is_ckd else 0.03, n_samples)
    data['dm'] = np.random.binomial(1, 0.76 if is_ckd else 0.03, n_samples)
    data['cad'] = np.random.binomial(1, 0.67 if is_ckd else 0.01, n_samples)
    data['appet'] = np.random.binomial(1, 0.49 if is_ckd else 0.00, n_samples)
    data['pe'] = np.random.binomial(1, 0.49 if is_ckd else 0.00, n_samples)
    data['ane'] = np.random.binomial(1, 0.51 if is_ckd else 0.00, n_samples)
    
    # Class label
    data['class'] = np.ones(n_samples) if is_ckd else np.zeros(n_samples)
    
    return pd.DataFrame(data)

# Generate CKD and non-CKD data
ckd_data = generate_synthetic_data(n_ckd, True)
non_ckd_data = generate_synthetic_data(n_non_ckd, False)

# Combine and shuffle
synthetic_data = pd.concat([ckd_data, non_ckd_data], ignore_index=True)
synthetic_data = synthetic_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Define column order to match your dataset
columns = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 
           'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 
           'dm', 'cad', 'appet', 'pe', 'ane', 'class']
synthetic_data = synthetic_data[columns]

# Save to CSV
synthetic_data.to_csv('kidney_synthetic_5000.csv', index=False)
print("Synthetic dataset with 5000 rows saved as 'kidney_synthetic_5000.csv'.")
print("Class distribution:")
print(synthetic_data['class'].value_counts())