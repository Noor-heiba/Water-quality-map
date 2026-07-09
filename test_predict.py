import pickle, json, urllib.request, math, os
import pandas as pd

coast_points = [
    (47.6,-122.3),(46.2,-124.1),(42.8,-124.5),(37.8,-122.5),(34.0,-118.3),(32.7,-117.2),
    (25.8,-80.2),(29.9,-90.1),(29.7,-95.4),(30.4,-87.2),(27.8,-97.4),
    (40.7,-74.0),(42.4,-71.0),(38.9,-77.0),(41.7,-87.6),(44.0,-83.0),
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

features = ['avg_temp_max','avg_precip','soil_temp_surface','soil_temp_deep',
            'soil_moisture_surface','soil_moisture_deep','elevation','dist_to_coast']

model_dir = os.path.expanduser('~') + '/Downloads/water_models/'
models = {}
for f in os.listdir(model_dir):
    if f.endswith('.pkl'):
        with open(model_dir+f,'rb') as fp:
            models[f.replace('.pkl','')] = pickle.load(fp)

def predict(lat,lon):
    env = get_env_data(lat,lon)
    if not env: return None
    env['elevation'] = get_elevation(lat,lon)
    env['dist_to_coast'] = distance_to_coast(lat,lon)
    results = {}
    for target,model in models.items():
        try:
            X = pd.DataFrame([env])[features]
            results[target] = round(float(model.predict(X)[0]),2)
        except:
            pass
    return results

print('San Jose prediction:')
r = predict(37.33,-121.88)
for k,v in r.items(): print(f'  {k}: {v}')

print('\nDenver prediction:')
r = predict(39.73,-104.99)
for k,v in r.items(): print(f'  {k}: {v}')

print('\nNew York prediction:')
r = predict(40.71,-74.01)
for k,v in r.items(): print(f'  {k}: {v}')

