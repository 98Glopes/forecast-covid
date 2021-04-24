import pmdarima as pm

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

    @staticmethod
    def get_arima_model():
        return pm.ARIMA(
            order=(2, 1, 2),
            seasonal_order=(2, 0, 2, 12)
        )
