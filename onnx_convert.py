import pickle, os, xgboost as xgb
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as XGBFloat

model_dir = os.path.expanduser('~') + '/Downloads/water_models/'
onnx_dir = os.path.expanduser('~') + '/Downloads/water_onnx/'
os.makedirs(onnx_dir, exist_ok=True)
features = ['avg_temp_max','avg_precip','soil_temp_surface','soil_temp_deep','soil_moisture_surface','soil_moisture_deep']
targets = ['Turbidity','Temperature_water','pH','Specific_conductance','Dissolved_oxygen_DO','Alkalinity_total','Total_dissolved_solids']

for target in targets:
    pkl_path = model_dir + f'{target}.pkl'
    if not os.path.exists(pkl_path): continue
    with open(pkl_path,'rb') as f: model = pickle.load(f)
    try:
        if isinstance(model, xgb.XGBRegressor):
            onnx_model = convert_xgboost(model, initial_types=[('float_input', XGBFloat([None, 6]))])
        else:
            onnx_model = convert_sklearn(model, initial_types=[('float_input', FloatTensorType([None, 6]))])
        save_path = onnx_dir + f'{target}.onnx'
        with open(save_path,'wb') as f: f.write(onnx_model.SerializeToString())
        print(f'{target}: saved ({round(os.path.getsize(save_path)/1024,1)} KB)')
    except Exception as e:
        print(f'{target}: failed — {e}')
print('Done!')
