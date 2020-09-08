import json
from collections import defaultdict

import pandas as pd
from responsibly.dataset import COMPASDataset, GermanDataset
from sklearn.preprocessing import LabelEncoder

'''class DatasetType(Enum):
    """
    Enumeration of the Datasets under consideration.
    Values represent the sensitive attributes for each of the dataset types
    """
    COMPAS = ['age', 'sex']
    GERMAN_CREDIT = ['credit']
'''

class Dataset:
    """
    The Dataset Interface for all fairness analysis
    """

    def __init__(self, config_file) :
        f = open(config_file, 'r')
        config = json.loads(f.read())
        self.type = config['dataset']
        _dataset = COMPASDataset() if self.type == 'COMPAS' else GermanDataset()
        self._sensitive_attributes = config["sensitive_attributes"]
        self._non_numeric_attributes = config["non_numeric_attributes"]

        self._n_attributes_dict = config["numeric_attributes"]
        self._numeric_attributes = list(self._n_attributes_dict.keys()) if self._n_attributes_dict else []
        self._date_attributes_dict = config["date_attributes"]
        self._date_attributes = list(self._date_attributes_dict.keys()) if self._date_attributes_dict else []

        self._target = config['ground_truth']
        self._predictions = config['predictions']

        self._data_x_readable = pd.DataFrame(_dataset.df
                                             [
                                                 self._sensitive_attributes+self._non_numeric_attributes+self._numeric_attributes+self._date_attributes])
        self._data_y_readable = pd.DataFrame(_dataset.df[self._target])
        self._encoder_dict_x = defaultdict(LabelEncoder)
        self._encoder_dict_y = defaultdict(LabelEncoder)
        self._data_x, self._data_y = self._preprocess_data()

    def _preprocess_data(self) :
        """
        Private method to process data by converting all text features to numeric
        :return: Tuple containing encoded features and targets
        """

        if self.type == 'COMPAS' :
            self._data_x_readable.fillna('', inplace=True)

        self._data_x_readable = self._data_x_readable.loc[:, ~self._data_x_readable.columns.duplicated()]

        # write a date attribute function

        if self._numeric_attributes :
            for attribute in self._numeric_attributes :
                self._data_x_readable[attribute+'_bins'] = pd.cut(x=self._data_x_readable[attribute],
                                                                  bins=self._n_attributes_dict[attribute])
                self._data_x_readable = self._data_x_readable.drop(columns=[attribute])
                self._data_x_readable = self._data_x_readable.rename(columns={attribute+'_bins' : attribute})

        return self._data_x_readable.apply(lambda x : self._encoder_dict_x[x.name].fit_transform(x)), \
               self._data_y_readable.apply(lambda y : self._encoder_dict_y[y.name].fit_transform(y))

    def get_sensitive_attributes(self) :
        return self._sensitive_attributes

    def get_data(self, readable=False) :
        """
        Method to get the data
        :param readable: Boolean flag to set if the returned data is human-readable
        :return: Tuple of data features and targets
        """
        if readable :
            return self._data_x_readable, self._data_y_readable
        return self._data_x, self._data_y

    # this as sort of useless. The init function ensures readable and unreadable forms are available as private vars,
    # accessible by get_data - skv2109
    def make_data_readable(self, x_df, y_df=None) :
        """
        Method to convert encoded data to their human readable form
        :param x_df: The encoded feature dataframe
        :param y_df: Optional dataframe with targets
        :return: Decoded features and targets
        """
        if self.type == 'COMPAS' :
            x_df.fillna('', inplace=True)
        x_readable = x_df.loc[:, ~self._data_x_readable.columns.duplicated()]
        x_readable = x_readable.apply(
            lambda x : self._encoder_dict_x[x.name].inverse_transform(x))
        if y_df is None:
            return x_readable
        y_readable = y_df.apply(lambda y: self._encoder_dict_y[y.name].inverse_transform(y))
        return x_readable, y_readable

    def encode_data(self, x_df, y_df=None):
        """
        Method to encode data to be processed by a computational model
        :param x_df: The human readable feature dataframe
        :param y_df: Optional dataframe with human readable targets
        :return: Encoded features (and targets)
        """
        # if self.type == 'COMPAS':
        # x_df.fillna('', inplace=True)
        x_encoded = x_df.loc[:, ~self._data_x_readable.columns.duplicated()]
        x_encoded = x_encoded.apply(
            lambda x : self._encoder_dict_x[x.name].transform(x))
        if y_df is None :
            return x_encoded
        y_encoded = y_df.apply(lambda y : self._encoder_dict_y[y.name].transform(y))
        return x_encoded, y_encoded


if __name__ == '__main__' :
    dataset = Dataset('config_compas.json')
    x, y = dataset.get_data(readable=True)
