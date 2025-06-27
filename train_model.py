import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

# === Load dataset ===
file_path = "indian_liver_patient.csv"  # CSV is in the same folder as this file
columns = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
    "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Proteins",
    "Albumin", "Albumin_and_Globulin_Ratio", "Dataset"
]

# Read the CSV file
try:
    df = pd.read_csv(file_path, names=columns, skiprows=1)
except FileNotFoundError:
    print("❌ Error: CSV file not found. Make sure it's in the same folder.")
    exit()

# === Data preprocessing ===
df.dropna(inplace=True)  # Drop rows with missing values
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})  # Encode gender
df["Dataset"] = df["Dataset"].map({1: 1, 2: 0})  # 1 = Cirrhosis, 0 = No Cirrhosis

# === Feature and label split ===
X = df.drop("Dataset", axis=1)
y = df["Dataset"]

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Scale the features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# === Train Random Forest model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# === Save the model and scaler ===
with open("rf_acc_68.pkl", "wb") as f:
    pickle.dump(model, f)

with open("normalizer.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully.")
