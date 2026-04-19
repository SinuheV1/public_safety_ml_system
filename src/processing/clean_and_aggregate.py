import pandas as pd
from pathlib import Path


output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

df=pd.read_parquet('data/raw/sheriff_crimes.parquet')

#standardize column names
df.columns=df.columns.str.lower()

#convert date from unix ms to local timezone
df['incident_date']=pd.to_datetime(df['incident_date'], unit='ms', utc=True).dt.tz_convert('America/Los_Angeles')
df['incident_reported_date']=pd.to_datetime(df['incident_reported_date'], unit='ms', utc=True).dt.tz_convert('America/Los_Angeles')

#convert latitude and longitude to numeric
df['latitude']=pd.to_numeric(df['latitude'],errors='coerce')
df['longitude']=pd.to_numeric(df['longitude'],errors='coerce')

#drop rows missing incident date or reporting district
df=df.replace('', pd.NA).dropna(subset=['incident_date','reporting_district'])

#normalize date
df['incident_day'] = df['incident_date'].dt.floor('D')

df.to_parquet(output_dir / 'sheriff_crimes_cleaned.parquet',index=False)


#====create continous time series====

#aggregate incidents by reporting district and day
df_grouped=df.groupby(['reporting_district','incident_day']).size().reset_index(name="incident_count")

#get all districts
districts=df_grouped['reporting_district'].drop_duplicates()

#get full date range
all_dates=pd.date_range(start=df_grouped['incident_day'].min(),end=df_grouped['incident_day'].max(),freq='D')
full_date_range_index=pd.MultiIndex.from_product([districts,all_dates],names=['reporting_district','incident_day'])
full_set=full_date_range_index.to_frame(index=False)

#merge datasets
df_final=full_set.merge(df_grouped,on=['reporting_district','incident_day'],how='left')

#fill missing data
df_final['incident_count']=df_final['incident_count'].fillna(0)

#cast incident counts as int
df_final['incident_count']=df_final['incident_count'].astype(int)

#drop timezone from date
df_final['incident_day']=df_final['incident_day'].dt.tz_localize(None)

#sort data
df_final=df_final.sort_values(['reporting_district','incident_day'])

df_final.to_parquet(output_dir / 'district_daily_counts.parquet',index=False)

