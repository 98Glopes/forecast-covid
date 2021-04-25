from predict_covid.data_provider import DataProvider
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def data_provider():
    provider = DataProvider()
    return provider

def test_data_provider_from_disk(data_provider):

    data_provider.load_dataset_from_github = MagicMock(
        return_value=data_provider.load_dataset_from_disk()
    )
    use_remote_dataset = False
    
    expected_coluns = {'new_cases'}
    df = data_provider.provide(use_remote_dataset)

    assert isinstance(df, pd.DataFrame)
    assert expected_coluns.issubset(set(df.columns))
    data_provider.load_dataset_from_github.assert_not_called()


def test_data_provider_from_github(data_provider):

    data_provider.load_dataset_from_github = MagicMock(
        return_value=data_provider.load_dataset_from_disk()
    )
    use_remote_dataset = True
    
    expected_coluns = {'new_cases'}
    df = data_provider.provide(use_remote_dataset)

    assert isinstance(df, pd.DataFrame)
    assert expected_coluns.issubset(set(df.columns))
    data_provider.load_dataset_from_github.assert_called_once()


@patch('predict_covid.data_provider.pd.read_csv', side_effect=ValueError)
def test_exception_load_dataset_from_github(pandas_mock, data_provider):
    data_provider.load_dataset_from_disk = MagicMock()

    df = data_provider.load_dataset_from_github()

    pandas_mock.assert_called_once()
    data_provider.load_dataset_from_disk.assert_called_once()
