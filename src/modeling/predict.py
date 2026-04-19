import pandas as pd
import joblib


df=pd.read_parquet("data/processed/model_dataset.parquet")

#sort df
df=df.sort_values(['reporting_district','incident_day'])
#get latest row
df_latest=df.groupby('reporting_district').tail(1).reset_index(drop=True).copy()
df_latest['prediction_for_day']=df_latest['incident_day'] + pd.Timedelta(days=1)

#load model
model=joblib.load('artifacts/model_100_trees.pkl')

#features
feature_columns=['day_of_week','month','day_of_month','is_weekend','incidents_lag_1',
                'incidents_lag_7','incidents_lag_14','incidents_rolling_mean_7',
                'incidents_rolling_mean_14','incidents_rolling_sum_7','incidents_rolling_sum_14',
                'district_expanding_mean_incidents']

X_latest=df_latest[feature_columns]

#predict incidents tomorrow
df_latest['predicted_incident_count_next_day'] = model.predict(X_latest)
df_latest=df_latest[['reporting_district','prediction_for_day','predicted_incident_count_next_day']]
#top 10 districts with highest predicted risk
top_districts = df_latest.sort_values('predicted_incident_count_next_day',ascending=False).head(10)

'''
print(X_latest.drop_duplicates().shape)
print(X_latest.shape)
print()
'''
print(top_districts)

'''
print()
print(df_latest["predicted_incident_count_next_day"].nunique())
'''

df_latest.to_parquet('artifacts/latest_predictions.parquet',index=False)
top_districts.to_parquet('artifacts/top_10_districts.parquet',index=False)

importances = pd.Series(model.feature_importances_,index=feature_columns).sort_values(ascending=False)

print(importances.head(10))
