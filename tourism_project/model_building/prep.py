from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "tourism_project" / "data" / "tourism.csv"
TARGET = "ProdTaken"
DROP_COLUMNS = ["Unnamed: 0", "CustomerID"]

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=DROP_COLUMNS, errors="ignore")
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' is not present")

X = df.drop(columns=TARGET)
y = df[TARGET].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

for name, frame in {
    "Xtrain.csv": X_train, "Xtest.csv": X_test,
    "ytrain.csv": y_train.to_frame(TARGET), "ytest.csv": y_test.to_frame(TARGET),
}.items():
    frame.to_csv(PROJECT_ROOT / name, index=False)

print(f"Saved train/test splits to {PROJECT_ROOT}")
print(f"Train rows: {len(X_train):,}; test rows: {len(X_test):,}")
