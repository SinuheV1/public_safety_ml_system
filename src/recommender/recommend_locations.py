import pandas as pd

df=pd.read_parquet('artifacts/lastest_predictions.parquet')


assert {'reporting_district','prediction_for_day','predicted_incident_count_next_day'}.issubset(df.columns)

#take top 10 districts
top_districts=df.sort_values('predicted_incident_count_next_day',ascending=False).head(10).copy()

top_districts['rank']=range(1,len(top_districts)+1)

top_districts=top_districts[['rank','reporting_district','prediction_for_day','predicted_incident_count_next_day']]
top_districts['predicted_incident_count_next_day']=top_districts['predicted_incident_count_next_day'].round(2)

top_districts['recommendation_reason']='High predicted next day incident volume'

