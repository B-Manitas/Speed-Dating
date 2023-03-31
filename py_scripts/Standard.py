# Metadata
__author__ = "Eytan Levy, Guillaume Surleau, Manitas Bahri"
__date__ = "07/2023"


# Librairies import
import pandas as pd
import numpy as np
import json
import re
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA

# Import custom modules
import py_scripts.utils as ut
import py_scripts.Utils as ut2


class Standard:
    """
    This class contains the functions used to standardize the data.

    Args:
        data_folder (str, optional): The path to the data folder. Defaults to "../data/".
    """

    def __init__(self, data_folder: str = "../data/") -> None:
        self.data_folder = data_folder

    def normalize_dataframe(self, dataframe: pd.DataFrame, columns: list) -> None:
        """
        Normalize the `columns` of a dataframe.

        Args:
            dataframe (pd.DataFrame): The dataframe.
            columns (list): The list of columns to normalize.

        Returns:
            pd.DataFrame: The dataframe with the `columns` normalized.
        """
        df_norm = dataframe[columns]
        df_norm = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())
        dataframe.update(df_norm)

    def normalize_range(self, value: int, min: int, max: int, born_max: int = 10) -> int:
        """
        Normalize a value by converting it to a value between 0 and a given maximum bound.

        Args:
            value (int): The value to normalize.
            min (int): The minimum value of the range of values.
            max (int): The maximum value of the range of values.
            born_max (int, optional): The upper bound of the normalized range. Defaults to 10.

        Returns:
            int: The normalized value.
        """
        if np.isnan(value):
            return value

        if value < min:
            return 0

        if value > max:
            return born_max

        return round(born_max * value / max)

    def normalize_range_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the columns of type "range" of a dataframe.

        Args:
            dataframe (pd.DataFrame): The dataframe.

        Returns:
            pd.DataFrame: The dataframe with the columns of type "range" normalized.
        """
        # Open the file which contains the types of the labels.
        with open(self.data_folder + "labels/types.json", "r") as f:
            d_type = json.load(f)

            # Iterates over each label of the dataframe.
            for key in dataframe.keys():

                # If the label is of type "range".
                if re.match(r"range \[[0-9?]+ \- [0-9?]+\]", d_type[key]):

                    # Get the minimum and maximum values of the range of this label.
                    min, max = re.findall(r"[0-9?]+", d_type[key])

                    # Display a warning if the minimum or maximum value is unknown.
                    if min == "?" or max == "?":
                        warning = f"Warning: '{key}' possède un min ou max inconnue -> [{min} - {max}]."
                        print(warning)

                    # Normalize the range.
                    else:
                        dataframe[key] = dataframe[key].map(
                            lambda x: self.normalize_range(x, int(min), int(max)))

        return dataframe

    def standardize_dataframe(self, dataframe: pd.DataFrame, columns: list) -> None:
        """
        Standardize the columns of `dataframe`.

        Args:
            dataframe (pd.DataFrame): The dataframe to updated.
            columns (list): List of columns to standardize.
        """
        df_stand = dataframe[columns]
        df_stand = (df_stand - df_stand.mean()) / df_stand.std()
        dataframe.update(df_stand)
