import pickle, os, xgboost as xgb
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

model_dir = os.path.expanduser('~') + '/Downloads/water_models/'
onnx_dir = os.path.expanduser('~') + '/Downloads/water_onnx/'

with open(model_dir + 'Dissolved_oxygen_DO.pkl', 'rb') as f:
    model = pickle.load(f)

# Rename features to f0-f5 for XGBoost compatibility
model.get_booster().feature_names = ['f0','f1','f2','f3','f4','f5']

onnx_model = convert_xgboost(model, initial_types=[('float_input', FloatTensorType([None, 6]))])
save_path = onnx_dir + 'Dissolved_oxygen_DO.onnx'
with open(save_path, 'wb') as f:
    f.write(onnx_model.SerializeToString())
print(f'Dissolved_oxygen_DO: saved ({round(os.path.getsize(save_path)/1024,1)} KB)')
