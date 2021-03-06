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
from scipy.stats import skew

class QuickPipeline():
    """
    QuickPipeline

    Performs the following tasks on input pandas dataframes:
    1. Fills empty data in a dataframe;
    2. Converts categorical columns to one-hot columns or binary columns;
    3. Deskews, moves and scales numerical columns to mean=1 and std=1;
    4. Drops uncorrelated and unuseful columns.

    Parameters
    ----------
    categorical_features : array-like
        A list of column names that must be one-hot encoded.

    y_column_name : str
        A name of column that is considered as y and must be converted from
        string to integer.

    impute : str (default='mean')
        A strategy of imputing missed values; passed to
        sklearn.preprocessing.Imputer.

    scale : bool (default=True)
        Moves and scales numerical columns to mean=1 and std=1.

    max_missing : float (default=0.9)
        The maximum percentage of missing data in a column. Discards a column
        if a percentage exceeds this value.

    min_correlation : float (default=None)
        Absolute minimum correlation coefficient between feature and y column.
        Feature column droped if absolute correlation is lower than this value.

    deskew : float (default=0.2)
        Deskew features with an absolute skewness more than this parameter
        (see scipy.stats.skew). Set to None to disable deskewing.

    copy : bool (default=True)
        Return a new dataframe(s) instead of modification the input
        dataframe(s).
    """
    def __init__(self, categorical_features=None, y_column_name=None,
                 impute='mean', scale=True,
                 max_missing=0.9, min_correlation=None,
                 deskew=0.2, copy=True
                ):
        self.categorical_features = categorical_features
        self.y_column_name = y_column_name
        self.impute = impute
        self.scale = scale
        self.max_missing = max_missing
        self.min_correlation = min_correlation
        self.deskew = deskew
        self.copy = copy

        # hardcoded thresholds:
        self.min_unique_for_deskew = 50
    
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
                df_columns.discard(self.y_column_name)
                df2_columns.discard(self.y_column_name)
                if len(df_columns ^ df2_columns) != 0:
                    raise ValueError('df and df2 columns mismatch')
                    
        if self.y_column_name is not None and self.y_column_name not in df.columns:
            raise ValueError('y_column_name not found in df')
            
        if self.copy:
            df = df.copy()
            if df2 is not None:
                df2 = df2.copy()

        # convert pandas categorical columns to string
        for c in df.columns:
            if pd.api.types.is_categorical_dtype(df[c]):
                df[c] = df[c].astype(str)
        if df2 is not None:
            for c in df2.columns:
                if pd.api.types.is_categorical_dtype(df2[c]):
                    df2[c] = df2[c].astype(str)

        # remove feature if missing data percentage exceeds self.max_missing:
        for c in df.columns:
            if c == self.y_column_name:
                continue
            missing = float(df[c].isnull().sum())/df.shape[0]
            if df2 is not None:
                missing2 = float(df2[c].isnull().sum())/df2.shape[0]
            else:
                missing2 = 0

            if missing > self.max_missing or missing2 > self.max_missing:
                del df[c]
                if df2 is not None:
                    del df2[c]
                continue

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

        # find and correct skewed features
        self.deskewed_features = list()
        if self.deskew != 0.0 and self.deskew is not None:
            numeric_features = list(df.dtypes[df.dtypes != object].index)
            if self.y_column_name in numeric_features:
                del numeric_features[numeric_features.index(self.y_column_name)]

            skewness = df[numeric_features].apply(lambda s: skew(s.dropna().astype(np.float_)))
            skewed_positive = skewness[skewness>self.deskew].index
            skewed_negative = skewness[skewness<-self.deskew].index

            for c in skewed_positive:
                # skip if a number of unique values are too low
                if df[c].nunique() < self.min_unique_for_deskew:
                    continue

                if min(df[c])<=0:  # skip if negative values found
                    continue
                if (df2 is not None) and (min(df2[c])<=0):
                    continue
                df[c] = np.log(df[c])
                if df2 is not None:
                    df2[c] = np.log(df2[c])

                self.deskewed_features.append(c)

            #for c in skewed_negative:
            #    df[c] = np.exp(df[c])
            #    if df2 is not None:
            #        df2[c] = np.exp(df2[c])

        # impute missing values in numeric features and normalize values:
        for c in df.columns:
            if (c in self.categorical_features) or (c == self.y_column_name):
                continue

            imputer = Imputer(strategy=self.impute)
            df[c] = imputer.fit_transform(df[c].values.reshape(-1,1))
            scaler = StandardScaler()
            df[c] = scaler.fit_transform(df[c].values.reshape(-1,1))
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
                df2[c] = df2[c].fillna('~~~')  # fills with '~~~'

            uniques = set(df[c].unique())
            if df2 is not None:
                uniques |= set(df2[c].unique())
                
            if len(uniques) == 1:
                # remove columns that do not contain useful data
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
                    categories |= set(df2[c].unique())
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
                    dtype=np.int8
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
                        dtype=np.int8
                    )

                    if '~~~' in one_hot_dataframe.columns:
                        del one_hot_dataframe['~~~']

                    del df2[c]
                    for c1 in one_hot_dataframe.columns:
                        df2[c1] = one_hot_dataframe[c1]

        if (self.min_correlation is not None) and (self.y_column_name is not None):
            correlation = df.corr()[self.y_column_name]
            self.non_correlative = correlation[
                (correlation<self.min_correlation)
                & (correlation>-self.min_correlation)
            ].index.values
            df.drop(self.non_correlative, axis=1, inplace=True)
            if df2 is not None:
                df2.drop(self.non_correlative, axis=1, inplace=True)

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
