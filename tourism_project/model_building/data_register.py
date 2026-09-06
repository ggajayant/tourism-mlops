from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tourism.csv"
REQUIRED_COLUMNS = {"CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
                    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
                    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
                    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
                    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"}

if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
missing = REQUIRED_COLUMNS - set(df.columns)
if missing:
    raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

print(f"Dataset registered: {DATA_PATH}")
print(f"Rows: {len(df):,}; columns: {len(df.columns)}")
print("Target distribution:\n", df["ProdTaken"].value_counts(normalize=True).sort_index())
print("Missing values (non-zero only):\n", df.isna().sum().loc[lambda s: s.gt(0)])
