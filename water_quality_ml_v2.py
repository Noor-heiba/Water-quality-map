import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import pickle
import json
import urllib.request
import math
import os

print('Loading national EPA data...')
pivot = pd.read_csv(os.path.expanduser('~') + '/Downloads/epa_national_pivot.csv')
print(f'Loaded {len(pivot)} locations across {pivot["state"].nunique()} states')

coast_points = [
    (47.6,-122.3),(46.2,-124.1),(42.8,-124.5),(37.8,-122.5),(34.0,-118.3),(32.7,-117.2),
    (25.8,-80.2),(29.9,-90.1),(29.7,-95.4),(30.4,-87.2),(30.3,-89.1),(27.8,-97.4),(26.1,-97.2),
    (40.7,-74.0),(42.4,-71.0),(38.9,-77.0),(36.8,-76.3),(34.2,-77.9),(32.8,-79.9),
    (30.3,-81.7),(25.7,-80.2),(44.8,-66.9),(43.1,-70.9),
    (43.0,-79.0),(42.9,-78.9),(41.7,-87.6),(44.0,-83.0),(46.5,-84.3),(47.9,-89.9),(44.9,-83.3),
]

def haversine(lat1,lon1,lat2,lon2):
    R=6371
    dlat=math.radians(lat2-lat1)
    dlon=math.radians(lon2-lon1)
    a=math.sin(dlat/2)**2+math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R*2*math.asin(math.sqrt(a))

def distance_to_coast(lat,lon):
    return round(min(haversine(lat,lon,c[0],c[1]) for c in coast_points),1)

def get_elevation(lat,lon):
    url=f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    req=urllib.request.Request(url,headers={'User-Agent':'WaterQualityApp/1.0'})
    try:
        with urllib.request.urlopen(req,timeout=10) as r:
            return json.loads(r.read())['results'][0]['elevation']
    except:
        return 0

def get_env_data(lat,lon):
    url=(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
         f"&daily=temperature_2m_max,precipitation_sum"
         f"&hourly=soil_temperature_0cm,soil_temperature_18cm,"
         f"soil_moisture_0_to_1cm,soil_moisture_3_to_9cm"
         f"&timezone=America%2FLos_Angeles&past_days=30")
    try:
        with urllib.request.urlopen(url,timeout=10) as r:
            d=json.loads(r.read())
        h=d['hourly'];dd=d['daily']
        return {
            'avg_temp_max':round(sum(x for x in dd['temperature_2m_max'] if x)/len(dd['temperature_2m_max']),1),
            'avg_precip':round(sum(x for x in dd['precipitation_sum'] if x)/len(dd['precipitation_sum']),2),
            'soil_temp_surface':round(sum(x for x in h['soil_temperature_0cm'] if x)/len(h['soil_temperature_0cm']),1),
            'soil_temp_deep':round(sum(x for x in h['soil_temperature_18cm'] if x)/len(h['soil_temperature_18cm']),1),
            'soil_moisture_surface':round(sum(x for x in h['soil_moisture_0_to_1cm'] if x)/len(h['soil_moisture_0_to_1cm']),3),
            'soil_moisture_deep':round(sum(x for x in h['soil_moisture_3_to_9cm'] if x)/len(h['soil_moisture_3_to_9cm']),3),
        }
    except:
        return None

env_cache = os.path.expanduser('~') + '/Downloads/env_1000locations.csv'
if os.path.exists(env_cache):
    print('Loading cached env data...')
    env_df = pd.read_csv(env_cache)
else:
    sample = pivot.sample(1000, random_state=42)
    print('Fetching OpenMeteo + elevation + coast for 1000 locations...')
    env_rows = []
    for i, row in sample.iterrows():
        env = get_env_data(row['lat'], row['lon'])
        if env:
            env['lat'] = row['lat']
            env['lon'] = row['lon']
            env['elevation'] = get_elevation(row['lat'], row['lon'])
            env['dist_to_coast'] = distance_to_coast(row['lat'], row['lon'])
            env_rows.append(env)
        if len(env_rows) % 100 == 0 and len(env_rows) > 0:
            print(f'  {len(env_rows)} done')
    env_df = pd.DataFrame(env_rows)
    env_df.to_csv(env_cache, index=False)
    print(f'Got {len(env_df)} locations')

sample = pivot.sample(1000, random_state=42)
df = sample.merge(env_df, on=['lat','lon'])
print(f'Dataset: {len(df)} locations')

features = ['avg_temp_max','avg_precip','soil_temp_surface','soil_temp_deep',
            'soil_moisture_surface','soil_moisture_deep','elevation','dist_to_coast']

targets = ['Turbidity','Temperature_water','pH','Specific_conductance',
           'Dissolved_oxygen_DO','Alkalinity_total','Total_dissolved_solids']

best_models = {}
for target in targets:
    if target not in df.columns: continue
    subset = df[features+[target]].dropna()
    if len(subset) < 15:
        print(f'{target}: not enough data')
        continue
    X = subset[features]
    y = subset[target]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100,random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100,random_state=42,verbosity=0)
    }
    print(f'\n--- {target} (n={len(subset)}) ---')
    best_r2=-999; best_name=None; best_model=None
    for name,model in models.items():
        model.fit(X_train,y_train)
        preds=model.predict(X_test)
        r2=round(r2_score(y_test,preds),3)
        mae=round(mean_absolute_error(y_test,preds),3)
        print(f'  {name}: R²={r2}, MAE={mae}')
        if r2>best_r2: best_r2=r2; best_name=name; best_model=model
    best_models[target]=best_model
    print(f'  Best: {best_name} (R²={best_r2})')

os.makedirs(os.path.expanduser('~')+'/Downloads/water_models/',exist_ok=True)
for target,model in best_models.items():
    with open(os.path.expanduser('~')+f'/Downloads/water_models/{target}.pkl','wb') as f:
        pickle.dump(model,f)
print(f'\nSaved {len(best_models)} models!')

def predict_water_quality(lat,lon):
    env=get_env_data(lat,lon)
    if not env: return None
    env['elevation']=get_elevation(lat,lon)
    env['dist_to_coast']=distance_to_coast(lat,lon)
    results={}
    for taet,model in best_models.items():
        X=pd.DataFrame([env])[features]
        results[target]=round(float(model.predict(X)[0]),2)
    return results

print('\nTest prediction for San Jose (37.33, -121.88):')
result=predict_water_quality(37.33,-121.88)
if result:
    for k,v in result.items(): print(f'  {k}: {v}')

