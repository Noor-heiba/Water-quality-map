import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import json
import urllib.request

# Load EPA data
epa = pd.read_csv('resultphyschem 5.csv',
                  usecols=['CharacteristicName', 'ResultMeasureValue',
                           'ActivityLocation/LatitudeMeasure',
                           'ActivityLocation/LongitudeMeasure'],
                  low_memory=False)

epa['ResultMeasureValue'] = pd.to_numeric(epa['ResultMeasureValue'], errors='coerce')
epa = epa.dropna()
epa = epa.rename(columns={
    'ActivityLocation/LatitudeMeasure': 'lat',
    'ActivityLocation/LongitudeMeasure': 'lon'
})

params = ['Turbidity', 'Temperature, water', 'pH', 'Dissolved oxygen (DO)', 'Specific conductance']
epa_filtered = epa[epa['CharacteristicName'].isin(params)]
epa_pivot = epa_filtered.groupby(['lat', 'lon', 'CharacteristicName'])['ResultMeasureValue'].mean().unstack()
epa_pivot.columns = [c.replace(', ', '_').replace(' ', '_') for c in epa_pivot.columns]
epa_pivot = epa_pivot.reset_index().dropna()
print(f'EPA locations: {len(epa_pivot)}')

def get_env_data(lat, lon):
    url = (f"https://api.open-meteo.com/v1/forecast?"
           f"latitude={lat}&longitude={lon}"
           f"&daily=temperature_2m_max,precipitation_sum"
           f"&hourly=soil_temperature_0cm,soil_temperature_18cm,"
           f"soil_moisture_0_to_1cm,soil_moisture_3_to_9cm"
           f"&timezone=America%2FLos_Angeles&past_days=30")
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            d = json.loads(r.read())
        h = d['hourly']
        dd = d['daily']
        return {
            'avg_temp_max': round(sum(x for x in dd['temperature_2m_max'] if x) / len(dd['temperature_2m_max']), 1),
            'avg_precip': round(sum(x for x in dd['precipitation_sum'] if x) / len(dd['precipitation_sum']), 2),
            'soil_temp_surface': round(sum(x for x in h['soil_temperature_0cm'] if x) / len(h['soil_temperature_0cm']), 1),
            'soil_temp_deep': round(sum(x for x in h['soil_temperature_18cm'] if x) / len(h['soil_temperature_18cm']), 1),
            'soil_moisture_surface': round(sum(x for x in h['soil_moisture_0_to_1cm'] if x) / len(h['soil_moisture_0_to_1cm']), 3),
            'soil_moisture_deep': round(sum(x for x in h['soil_moisture_3_to_9cm'] if x) / len(h['soil_moisture_3_to_9cm']), 3),
        }
    except:
        return None

print('Fetching OpenMeteo data...')
env_rows = []
for i, row in epa_pivot.head(50).iterrows():
    env = get_env_data(row['lat'], row['lon'])
    if env:
        env['lat'] = row['lat']
        env['lon'] = row['lon']
        env_rows.append(env)
    if len(env_rows) % 10 == 0:
        print(f'  {len(env_rows)} done')

env_df = pd.DataFrame(env_rows)
df = epa_pivot.merge(env_df, on=['lat', 'lon'])
print(f'Final dataset: {len(df)} locations')

features = ['avg_temp_max', 'avg_precip', 'soil_temp_surface',
            'soil_temp_deep', 'soil_moisture_surface', 'soil_moisture_deep']
targets = [c for c in df.columns if c not in ['lat', 'lon'] + features]

for target in targets:
    subset = df[features + [target]].dropna()
    if len(subset) < 10:
        continue
    X = subset[features]
    y = subset[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }
    print(f'\n--- {target} ---')
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = round(r2_score(y_test, preds), 3)
        mae = round(mean_absolute_error(y_test, preds), 3)
        print(f'{name}: R²={r2}, MAE={mae}')

