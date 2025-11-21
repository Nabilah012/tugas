import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ==========================
# 1. LOAD DATASET
# ==========================
df = pd.read_csv("dqlab_telco.csv")

# ==========================
# 2. DATA CLEANING
# ==========================

# Buang baris yang Churn-nya kosong
df = df.dropna(subset=["Churn"])

# Ubah kolom TotalCharges agar numerik
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Isi NaN TotalCharges
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Encode target: Yes = 1, No = 0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ==========================
# 3. PILIH FITUR PENTING
# ==========================
# Ini contoh sederhana: 3 fitur numerik
X = df[["tenure", "MonthlyCharges", "TotalCharges"]]
y = df["Churn"]

# ==========================
# 4. TRAIN-TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================
# 5. TRAIN MODEL
# ==========================
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# 6. EVALUASI
# ==========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi:", accuracy)

# ==========================
# 7. SAVE MODEL
# ==========================
joblib.dump(model, "model_churn.pkl")
print("Model saved as model_churn.pkl")
