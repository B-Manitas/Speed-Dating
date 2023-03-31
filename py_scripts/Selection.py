# Librairies import
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA


class Selection:
    """
    This class contains the methods to select the features of the dataframe.
    """

    def __init__(self) -> None:
        pass

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
        x = dataframe.drop(
            columns=["match", "decision", "p_decision", "id_group"])
        y = dataframe["match"].to_frame()

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
            return dataframe, pd.DataFrame(columns=dataframe.columns)

        # Else, split the dataframe.
        else:
            splitter = GroupShuffleSplit(
                n_splits=2, test_size=ratio_test, random_state=seed)
            groups, _, _ = self.get_groups_X_y(dataframe)

            i_train, i_test = next(splitter.split(dataframe, groups=groups))

            train = dataframe.iloc[i_train]
            test = dataframe.iloc[i_test]

            return train, test
