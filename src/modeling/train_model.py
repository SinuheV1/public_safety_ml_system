import pandas as pd
from pathlib import Path
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error,r2_score,mean_absolute_error

#artifacts directory
artifacts_dir=Path('artifacts/')
artifacts_dir.mkdir(parents=True, exist_ok=True)

df=pd.read_parquet('data/processed/model_dataset.parquet')

#sort on date
df=df.sort_values("incident_day").reset_index(drop=True)

#target column name
target_column='target_incident_count_next_day'

#feature columns
feature_columns=['day_of_week','month','day_of_month','is_weekend','incidents_lag_1',
                'incidents_lag_7','incidents_lag_14','incidents_rolling_mean_7',
                'incidents_rolling_mean_14','incidents_rolling_sum_7','incidents_rolling_sum_14',
                'district_expanding_mean_incidents']

#train on about 5 years data
#test on about 1 year of data

cutoff_date=pd.Timestamp('2025-01-01')

train_mask=df['incident_day'] < cutoff_date
test_mask=df['incident_day'] >= cutoff_date

train_df=df.loc[train_mask].copy()
test_df=df.loc[test_mask].copy()

X_train=train_df[feature_columns]
X_test=test_df[feature_columns]

y_train=train_df[target_column]
y_test=test_df[target_column]


print("Train rows:", len(train_df))
print("Test rows:", len(test_df))
print("Train min/max:", train_df["incident_day"].min(), train_df["incident_day"].max())
print("Test min/max:", test_df["incident_day"].min(), test_df["incident_day"].max())
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

#model fit
regressor=RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1,oob_score=True)
regressor.fit(X_train,y_train)
print(f'Out-of-Box score: {regressor.oob_score_}')

y_prediction=regressor.predict(X_test)

mae=mean_absolute_error(y_test,y_prediction)
rmse=root_mean_squared_error(y_test,y_prediction)
r2=r2_score(y_test,y_prediction)


model_path = artifacts_dir / "model_100_trees.pkl"
joblib.dump(regressor, model_path)

print(f"Model saved to: {model_path}")

metrics = {"train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "cutoff_date": str(cutoff_date),
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2)}

metrics_path = artifacts_dir / "metrics_100_trees.json"

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Metrics saved to: {metrics_path}")

print("\n=== Training Summary ===")
print(metrics)