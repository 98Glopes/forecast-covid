import pytest
import os
from unittest.mock import MagicMock

import pmdarima as pm

from predict_covid.model import CovidModel


@pytest.fixture(scope='session')
def model():
    model = CovidModel()
    model.use_remote_dataset = False
    model.data_path = 'mock.pkl'
    yield model

    if os.path.isfile(model.model_path):
        os.remove(model.model_path)


def test_predict(model):
    days = 10
    model.load_model()
    
    forecast = model.predict(days=days)

    assert isinstance(forecast, dict)
    assert len(forecast.items()) == days


def test_predict_zero_and_negative_days(model):
    
    with pytest.raises(ValueError):
        model.predict(days=0)

    with pytest.raises(ValueError):
        model.predict(days=-999)

