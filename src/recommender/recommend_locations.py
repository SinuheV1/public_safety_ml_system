import pandas as pd

"""
Standalone script for generating top-N district recommendations
from latest prediction outputs.

Used for batch outputs (CSV/parquet) and reproducible reporting.
"""

top_n=10

df=pd.read_parquet('artifacts/latest_predictions.parquet')

required_cols={'reporting_district','prediction_for_day','predicted_incident_count_next_day'}

if not required_cols.issubset(df.columns):
    missing=required_cols - set(df.columns)
    raise ValueError(f'Missing required columns: {missing}')

top_districts=df.sort_values('predicted_incident_count_next_day', ascending=False).head(top_n).copy()

top_districts['rank']=range(1, len(top_districts) + 1)
top_districts['predicted_incident_count_next_day']=top_districts['predicted_incident_count_next_day'].round(2)
top_districts['recommendation_reason']='High predicted next-day incident volume'

top_districts=top_districts[['rank','reporting_district','prediction_for_day',
                            'predicted_incident_count_next_day','recommendation_reason']]

print(top_districts)

top_districts.to_parquet('artifacts/top_10_districts.parquet', index=False)
top_districts.to_csv('artifacts/top_10_districts.csv', index=False)