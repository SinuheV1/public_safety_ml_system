import requests
import pandas as pd
from pathlib import Path
import time




url = 'https://services.arcgis.com/RmCCgQtiZLDCtblq/arcgis/rest/services/Sheriff%20Part%201%20and%202%20Crimes%20(Historical)/FeatureServer/0/query'
params = {
    "where": "INCIDENT_DATE >= DATE '2020-01-01'",
    "outFields": "OBJECTID,INCIDENT_DATE,INCIDENT_REPORTED_DATE,CATEGORY,STAT_DESC,PART_CATEGORY,REPORTING_DISTRICT,UNIT_NAME,ZIP,LATITUDE,LONGITUDE",
    "returnGeometry": "false",
    "f": "json",
    "resultRecordCount": 1000,
    "resultOffset": 0}


offset = 0
batch_size = 1000
all_records = []

while True:
    params["resultOffset"] = offset
    params["resultRecordCount"] = batch_size

    response=requests.get(url,params=params)
    response.raise_for_status()
    data = response.json()
    records = [feature["attributes"] for feature in data["features"]]
    
    #protect against crashes if api response bad
    features = data.get("features", [])
    records = [f["attributes"] for f in features]

    #break loop when no more records
    if not records:
        print("No more records. Stopping.")
        break
    
    #add all records 
    all_records.extend(records)
    
    #print progress logging
    print(f"Fetched {len(records)} rows | Total: {len(all_records)}")
    
    #move to next batch
    offset += batch_size
    
    #sleep 2 seconds for api request limit
    time.sleep(2)
df = pd.DataFrame(all_records)
print(df.shape)
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

df.to_parquet(output_dir / "sheriff_crimes.parquet", index=False)