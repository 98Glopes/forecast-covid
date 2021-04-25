import pmdarima as pm
import pickle
import os

from data.data_provider import DataProvider
from decouple import config



def get_arima_model():
    return pm.ARIMA(
        order=(2, 1, 2),
        seasonal_order=(2, 0, 2, 12)
    )

def fit_model(model: pm.ARIMA):
    data = DataProvider().provide(True)
    return model.fit(data)

def write_model_on_disk(model):
    model_path = config('MODEL_PATH')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def read_model_from_disk():
    model_path = config('MODEL_PATH')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def check_model_exists():
    model_path = config('MODEL_PATH')
    return os.path.isfile(model_path)

def get_new_model():
    model = get_arima_model()
    model = fit_model(model)
    write_model_on_disk(model)
    return model

def load_model():
    if check_model_exists():
        model = read_model_from_disk()
    else: 
        model = get_new_model()
    return model

def predict(days=1):
    if days <= 0:
        raise ValueError("Steps must be greater than zero")
    
    model = load_model()
    forecast = model.predict(days)

    return forecast
    