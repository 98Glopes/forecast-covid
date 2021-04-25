import pmdarima as pm
import pickle
import os

from data.data_provider import DataProvider
from decouple import config


class CovidModel:
    """
    COVID Model
    Model trained with daily covid cases to predict the future.
    Serialize the ARIMA's model in disk so isn't necessary refit
    The model was hyper parametrized using the estimator auto_arima,
    for more details check "eda - covid19" directory, all jupyter
    notebooks used are available.
    """
    data_provider = DataProvider()
    use_remote_dataset = config('USE_REMOTE_DATASET')
    model_path = config('MODEL_PATH')
    model = None

    def __init__(self):
        self.model = self.load_model()

    def get_arima_model(self):
        return pm.ARIMA(
            order=(2, 1, 2),
            seasonal_order=(2, 0, 2, 12)
        )

    def fit_model(self, model: pm.ARIMA):
        data = self.data_provider.provide(self.use_remote_dataset)
        return model.fit(data)

    def write_model_on_disk(self, model):
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)

    def read_model_from_disk(self):
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    @property
    def check_model_exists(self):
        return os.path.isfile(self.model_path)

    def get_new_model(self):
        model = self.get_arima_model()
        model = self.fit_model(model)
        self.write_model_on_disk(model)
        return model

    def load_model(self):
        if self.check_model_exists:
            model = self.read_model_from_disk()
        else: 
            model = self.get_new_model()
        return model

    def predict(self, days=1):
        if days <= 0:
            raise ValueError("Steps must be greater than zero")
        
        forecast = self.model.predict(days)

        return forecast
