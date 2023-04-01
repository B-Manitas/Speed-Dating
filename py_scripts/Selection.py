# Metadata
__author__ = "Eytan Levy, Guillaume Surleau, Manitas Bahri"
__date__ = "03/2023"

# Librairies import
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler


class Selection:
    """
    This class contains the methods to select the features of the dataframe.
    """

    def get_samples_of_dataframe(self, dataframe: pd.DataFrame, n_sample: int = 500) -> pd.DataFrame:
        """
        Get a sample of the dataframe containing the same number of rows with match = 1 and match = 0.

        Args:
            dataframe (pd.DataFrame): The input dataframe.
            n_sample (int): The number of rows to sample. By default 500.

        Returns:
            pd.DataFrame: A sample of the dataframe containing the same number of rows with match = 1 and match = 0.
        """
        id_yes = dataframe[dataframe["match"] == 1].index[:n_sample]
        id_no = dataframe[dataframe["match"] == 0].index[:n_sample]

        return dataframe.iloc[list(id_yes) + list(id_no)]

    def get_groups_X_y(self, dataframe: pd.DataFrame):
        """
        Return the groups, features and targets of the `dataframe`.

        Args:
            dataframe (pd.DataFrame): The DataFrame.

        Returns:
            pd.Series, pd.DataFrame, pd.Series: The DataFrames groups, X and y.
        """
        groups = dataframe["id_group"]
        cols_x_to_drop = ["id_group", "match", "decision", "p_decision"]

        df_x = dataframe.drop(columns=cols_x_to_drop)
        df_y = dataframe["match"]

        return groups, df_x, df_y

    def get_X_y(self, dataframe: pd.DataFrame):
        """
        Return the features and targets of the `dataframe`.

        Args:
            dataframe (pd.DataFrame): The DataFrame.

        Returns:
            pd.DataFrame, pd.Series: The DataFrames X and y.
        """
        x_to_drop = ["match", "decision", "p_decision", "id_group"]
        x = dataframe.drop(columns=x_to_drop, errors="ignore")
        y = pd.DataFrame(dataframe["match"], index=dataframe.index)

        return x, y

    def split_by_groups(self, dataframe: pd.DataFrame, ratio_test: float = .2, seed: int = 0):
        """
        Return two random dataframe train and tests respecting the groups of the data.

        Args:
            dataframe (pd.DataFrame): The DataFrame to split.
            ratio_test (float, optional): The ratio of the size of the test set. Defaults to 0.2.
            seed (int, optional): The seed of the random. Defaults to 0.

        Returns:
            pd.DataFrame, pd.DataFrame: The train and test DataFrames.
        """
        # If the dataframe is empty, return two empty dataframes.
        if len(dataframe) == 0:
            return dataframe, dataframe

        # If the ratio is not between 0 and 1, return the dataframe and an empty dataframe.
        elif not(0 < ratio_test < 1):
            return dataframe, pd.DataFrame(data=[], columns=dataframe.columns)

        # Else, split the dataframe.
        else:
            splitter = GroupShuffleSplit(
                n_splits=2, test_size=ratio_test, random_state=seed)
            groups, _, _ = self.get_groups_X_y(dataframe)

            i_train, i_test = next(splitter.split(dataframe, groups=groups))

            train = dataframe.iloc[i_train]
            test = dataframe.iloc[i_test]

            return train, test

    def undersampling(self, x: pd.DataFrame, y: pd.DataFrame):
        """
        Return the undersampled data.

        Args:
            x (pd.DataFrame): The features.
            y (pd.DataFrame): The targets.

        Returns:
            pd.DataFrame, pd.DataFrame: The undersampled features and targets.
        """
        
        # Create the training data with the SMOTE algorithm.
        rus = RandomUnderSampler(random_state=0)
        x_sampled, y_sampled = rus.fit_resample(x, y) # type: ignore   

        # Convert the numpy arrays to dataframes.
        x_sampled = pd.DataFrame(x_sampled, columns=x.columns, index=x_sampled.index)
        y_sampled = pd.DataFrame(y_sampled, columns=["match"], index=y_sampled.index)

        return x_sampled, y_sampled
