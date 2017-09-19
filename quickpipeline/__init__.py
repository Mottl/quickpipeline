# -*- coding: utf-8 -*-

"""
quickpipeline module implements QuickPipeline class that do all the necessary
things to prepare data for machine learning tasks.

2017 (c) Dmitry Mottl
License: MIT 
"""

from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    Imputer, StandardScaler,
    LabelEncoder, OneHotEncoder, LabelBinarizer
)

class QuickPipeline():
    """
    QuickPipeline

    Performs the following tasks on input pandas dataframes:
    1. Fills empty data in a dataframe
    2. Converts categorical columns to one-hot columns or binary columns
    3. Moves and scales numerical columns to mean=1 and std=1

    Parameters
    ----------
    categorical_features : array-like
        A list of column names that must be one-hot encoded

    y_column_name : str
        A name of column that is considered as y and must be converted from string to integer

    impute : str (default='mean')
        A strategy of imputing missed values; passed to sklearn.preprocessing.Imputer. Default is 

    scale : bool (default=True)
        Moves and scales numerical columns to mean=1 and std=1

    copy : bool (default=True)
        Return a new dataframe instead of modification of input dataframe
    """
    def __init__(self, categorical_features=None, y_column_name=None, impute='mean', scale=True, copy=True):
        self.categorical_features = categorical_features
        self.y_column_name = y_column_name
        self.impute = impute
        self.scale = scale
        self.copy = copy
    
    def fit_transform(self, df, df2=None):
        '''
        Fit and transform pandas dataframes

        Parameters
        ----------

        df: pandas Dataframe, shape (n_samples, n_features(+1 if y used))
            Training dataframe with y column if needed (must be specified with
            y_column_name in constructor)

        df2: pandas Dataframe, shape (n_samples, n_features) (default=None)
            Testing dataframe
        '''
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError('df must be pandas DataFrames')

        if df2 is not None:
            if not isinstance(df2, pd.DataFrame):
                raise ValueError('df must be pandas DataFrames')
            df_columns = set(df.columns)
            df2_columns = set(df2.columns)
            if self.y_column_name is not None:
                if self.y_column_name in df_columns:
                    del df_columns[self.y_column_name]
                if self.y_column_name in df2_columns:
                    del df2_columns[self.y_column_name]
                if len(df_columns ^ df2_columns) != 0:
                    raise ValueError('df and df2 columns mismatch')
                    
        if self.y_column_name is not None and self.y_column_name not in df.columns:
            raise ValueError('y_column_name not found in df')
            
        if self.copy:
            df = df.copy()
            if df2 is not None:
                df2 = df2.copy

        # create a list of categorical features if not set
        if self.categorical_features is None:  # get categorical_features automatically
            self.categorical_features = list(filter(
                lambda c: c != self.y_column_name
                    and (
                        (df[c].dtype == object)
                        or (df2 is not None and df2[c].dtype == object)
                    ),
                df.columns
            ))
        # impute missing values in non-categorical features and normalize values:
        for c in df.columns:
            if (c in self.categorical_features) or (c == self.y_column_name):
                continue
            imputer = Imputer(strategy=self.impute)
            df[c] = imputer.fit_transform(df[c].values.reshape(-1,1))
            scaler = StandardScaler()
            scaler.fit_transform(df[c].values.reshape(-1,1))
            if df2 is not None:
                df2[c] = imputer.transform(df2[c].values.reshape(-1,1))
                df2[c] = scaler.transform(df2[c].values.reshape(-1,1))

        # create a dicts for encoders, key is a column name, value is an encoder
        self.__label_encoders = defaultdict(LabelEncoder)
        self.__label_binarizers = defaultdict(LabelBinarizer)
        self.__onehot_encoders = defaultdict(OneHotEncoder)

        for c in self.categorical_features:
            df[c] = df[c].fillna('~~~')  # fills with '~~~'
            if df2 is not None:
                df[c] = df[c].fillna('~~~')  # fills with '~~~'

            uniques = set(df[c].unique())
            if df2 is not None:
                uniques += set(df2[c].unique())
                
            if len(uniques) == 1:
                # remove columns that do not contains useful data
                del df[c]
                if df2 is not None:
                    del df2[c]
            elif len(uniques) == 2:
                # binarize
                df[c] = self.__label_binarizers[c].fit_transform(df[c])
                if df2 is not None:
                    df2[c] = self.__label_binarizers[c].transform(df2[c])
            else:
                # convert to labels
                
                # get all possible values from a given column and fit LabelEncoder
                categories = set(df[c].unique())
                if df2 is not None:
                    categories += set(df2[c].unique())
                categories = sorted(categories)

                labels = [c+'_'+cat if cat!='~~~' else '~~~' for cat in categories]  # column labels
                # construct a column of possible values
                possible_values = self.__label_encoders[c].fit_transform(categories).reshape(-1,1)
                
                transformed_series = self.__label_encoders[c].transform(df[c]).reshape(-1,1)
                if df2 is not None:
                    transformed_series2 = self.__label_encoders[c].transform(df2[c]).reshape(-1,1)

                # create a one-hot matrix dataframe for a given column
                self.__onehot_encoders[c].fit(possible_values)
                one_hot_matrix = self.__onehot_encoders[c].transform(transformed_series)
                one_hot_dataframe = pd.DataFrame(
                    data=one_hot_matrix.toarray(), # convert sparse matrix to 2dim array
                    index=df.index,
                    columns=labels,
                    dtype=np.int32
                )

                # remove `missing values` column form one_hot_dataframe
                if '~~~' in one_hot_dataframe.columns:
                    del one_hot_dataframe['~~~']

                # remove old column and add a one-hot matrix
                del df[c]

                # add one-hot columns to df
                for c1 in one_hot_dataframe.columns:
                    df[c1] = one_hot_dataframe[c1]

                if df2 is not None:
                    one_hot_matrix = self.__onehot_encoders[c].transform(transformed_series2)
                    one_hot_dataframe = pd.DataFrame(
                        data=one_hot_matrix.toarray(),
                        index=df2.index,
                        columns=labels,
                        dtype=np.int32
                    )

                    if '~~~' in one_hot_dataframe.columns:
                        del one_hot_dataframe['~~~']

                    del df2[c]
                    for c1 in one_hot_dataframe2.columns:
                        df2[c1] = one_hot_dataframe2[c1]

        if self.y_column_name is not None:
            if df[self.y_column_name].dtype == object:
                self.y_encoder = LabelEncoder()
                df[self.y_column_name] = self.y_encoder.fit_transform(df[self.y_column_name])

            # move y column to end
            y = df[self.y_column_name]
            del df[self.y_column_name]
            df[self.y_column_name] = y

        if df2 is not None:
            return df, df2
        else:
            return df

if __name__ == "__main__":
    s1 = pd.Series([1,2,3,np.nan,4,5], dtype=np.float16)
    s2 = pd.Series(["A","B",np.nan,"A","C","B"])
    y = pd.Series(["yes","yes","no","yes","no","no"])
    df = pd.DataFrame({"s1": s1, "s2": s2, "y": y})

    pipeline = QuickPipeline(y_column_name="y", copy=True)
    df_prepared = pipeline.fit_transform(df)
    print(df_prepared)
