import pandas as pd
from pathlib import Path

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)


df=pd.read_parquet('data/processed/district_daily_counts.parquet')
#df.info()
df=df.sort_values(["reporting_district", "incident_day"])
df=df.reset_index(drop=True)

#create calendar features

#day of week
df['day_of_week']=df['incident_day'].dt.dayofweek
#month
df['month']=df['incident_day'].dt.month
#day
df['day_of_month']=df['incident_day'].dt.day
#is weekend
df['is_weekend']=df['day_of_week'].isin([5,6]).astype(int)

#print(df[["incident_day","day_of_week","month","is_weekend"]].head())

#create lag features
#1 day, 1 week, 2 weeks

for i in [1,7,14]:
    df[f'incidents_lag_{i}']=df.groupby('reporting_district')['incident_count'].shift(i)

#create rolling features
#rolling 7 day mean
df['incidents_rolling_mean_7']=df.groupby('reporting_district')['incident_count'].transform(lambda x: x.rolling(7).mean())

#rolling 14 day mean
df['incidents_rolling_mean_14']=df.groupby('reporting_district')['incident_count'].transform(lambda x: x.rolling(14).mean())

#rolling 7 day sum
df['incidents_rolling_sum_7']=df.groupby('reporting_district')['incident_count'].transform(lambda x: x.rolling(7).sum())

#rolling 14 day sum
df['incidents_rolling_sum_14']=df.groupby('reporting_district')['incident_count'].transform(lambda x: x.rolling(14).sum())

# district baseline feature: prior historical average incident count per district
# uses only past information by shifting 1 day before expanding mean
df['district_expanding_mean_incidents']=(df.groupby('reporting_district')['incident_count']
                                        .transform(lambda s: s.shift(1).expanding().mean()))

#set target incident column
df['target_incident_count_next_day']=df.groupby('reporting_district')['incident_count'].shift(-1)


#drop null rows
required_cols=['incidents_lag_1','incidents_lag_7','incidents_lag_14','incidents_rolling_mean_7',
            'incidents_rolling_mean_14','incidents_rolling_sum_7','incidents_rolling_sum_14',
            'district_expanding_mean_incidents','target_incident_count_next_day']

df=df.dropna(subset=required_cols)
df=df.sort_values(['reporting_district','incident_day']).reset_index(drop=True)


print(f'DF Shape: {df.shape}')
print(f"\n Min Date: {df['incident_day'].min()}")
print(f"\n Max Date: {df['incident_day'].max()}")
print(f"\n Count of Districts: {df['reporting_district'].nunique()}")

df.to_parquet(output_dir / 'model_dataset.parquet',index=False)
