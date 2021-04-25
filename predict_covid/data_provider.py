import warnings

import pandas as pd
from decouple import config


class DataProvider:
    """
    Data Provider
    Get raw dataset from disk or GitHub and filter just the interest columns
    to fit Machine Learning model
    """
    interest_columns = ['new_cases', 'date']
    location = 'World'
    index_column = 'date'
    dataset_path = config('DATASET_PATH')
    dataset_github_url = config('DATASET_GITHUB_URL')

    def load_dataset_from_disk(self):
        return pd.read_csv(self.dataset_path)

    def load_dataset_from_github(self):
        try:
            df = pd.read_csv(self.dataset_github_url)
        except Exception:
            df = self.load_dataset_from_disk()
            warnings.warn("Wasn't possible load dataset from github. Disk dataset was loaded",
                          UserWarning)
        return df

    def load_dataset(self, use_remote_dataset):
        if use_remote_dataset:
            df = self.load_dataset_from_github()
        else:
            df = self.load_dataset_from_disk()
        return df

    def filter_interest_columns(self, dataset: pd.DataFrame):
        return dataset[self.interest_columns]

    def set_dataset_index(self, dataset: pd.DataFrame):
        return dataset.set_index(self.index_column, drop=True)

    def filter_dataset_location(self, dataset: pd.DataFrame):
        return dataset[dataset['location'] == self.location]

    def provide(self, use_remote_dataset):
        df = self.load_dataset(use_remote_dataset)
        df = self.filter_dataset_location(df)
        df = self.filter_interest_columns(df)
        df = self.set_dataset_index(df)
        return df

