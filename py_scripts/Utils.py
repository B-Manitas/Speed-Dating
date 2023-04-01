# Metadata
__author__ = "Eytan Levy, Guillaume Surleau, Manitas Bahri"
__date__ = "03/2023"


# Librairies import
import json
import pandas as pd
import numpy as np


class Utils:
    """
    This class contains useful functions for the data science project.

    Attributes:
        data_folder (str): The path to the data folder.

    Methods:
        count_multiple_columns(dataframe: pd.DataFrame, column: str, blacklist: list = []) -> Tuple[List[int], List[str]]
        filter_dataframe_by_substr(dataframe: pd.DataFrame, substr: str = "") -> pd.DataFrame
        get_dict_onehot(series: pd.Series, columns: list) -> dict
        get_files_bow(series: str) -> str
        get_matching_keys(keys, sub_strs: list) -> list
        isnan(o: object) -> bool
        update_range_types(dataframe: pd.DataFrame) -> dict
        print_labels_info(*args: str) -> None
    """
    data_folder = "../data/"

    @classmethod
    def count_multiple_columns(cls, dataframe: pd.DataFrame, column: str, blacklist: list = []):
        """
        Count the number of occurences for each column that contains a substring `str` in its name.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data to process.
            column (str): The substring to search for in the column names.
            blacklist (list, optional): List of column names to exclude from the results. Defaults to [].

        Returns:
            Tuple[List[int], List[str]]: A tuple containing two lists:
                the first contains the values of the occurrences
                the second contains the names of the columns.
        """
        # Filter the DataFrame to include only the rows where the specified column contains the substring
        df = cls.filter_dataframe_by_substr(dataframe, column)

        # Transforme les colonnes de df en lignes.
        # Transform the columns of df into rows.
        col_to_line = df.melt(var_name='columns', value_name='index')

        # Compute the frequency table of the columns
        df = pd.crosstab(**col_to_line)  # type: ignore

        # Remove the columns specified in the "blacklist" list
        df.drop(columns=blacklist, inplace=True)

        # Sort the columns in alphabetical order
        data = df.iloc[1].sort_values(ascending=False)

        values = data.values
        labels = " ".join(data.keys()).replace(column, "").split()

        return values, labels

    @classmethod
    def count_ratio_match_features(cls, dataframe: pd.DataFrame, features: str) -> dict:
        """
        Count the ratio of matches for `features`.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data to process.
            features (str): The substring to search for in the column names.

        Returns:
            dict: A dictionary containing the ratio of matches for `features`.
        """
        careers = Utils.filter_dataframe_by_substr(dataframe, features).columns
        ratios = {}

        for career in careers:
            df_career = dataframe[dataframe[career] == 1]

            if len(df_career) == 0:
                ratio = 0

            else:
                ratio = np.sum(df_career['match'] == 1) * 100 / len(df_career)

            ratios[career] = ratio

        ratios = dict(
            sorted(ratios.items(), key=lambda item: item[1], reverse=True))
        return ratios

    @classmethod
    def count_match_proportions(cls, dataframe: pd.DataFrame, features: str) -> pd.DataFrame:
        """
        Count the proportion of matches for `features`.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data to process.
            features (str): The substring to search for in the column names.

        Returns:
            pd.DataFrame: A DataFrame containing the proportion of matches for `features`.
        """
        df = dataframe.groupby([features, "match"])
        df = df.size().reset_index(name="count")
        df_transform = df.groupby(features)["count"].transform("sum")

        df["proportion"] = df["count"] / df_transform

        return df

    @classmethod
    def filter_dataframe_by_substr(cls, dataframe: pd.DataFrame, substr: str = "") -> pd.DataFrame:
        """
        Get a subset of the dataframe containing only the columns whose names contain the substring `substr`.

        Args:
            dataframe (pd.DataFrame): The original DataFrame.
            substring (str): The keyword used to filter the DataFrame columns. By default the empty string "".

        Returns:
            pd.DataFrame: The subset of `dataframe` containing only the columns whose names contain `substr`
            sorted in alphabetical order.
        """
        columns = [column for column in dataframe.keys() if substr in column]
        columns.sort()

        return dataframe[columns]

    @classmethod
    def get_dict_onehot(cls, series: pd.Series, columns: list) -> dict:
        """
        Get a dictionary containing the values of the series encoded in one-hot.

        Args:
            series (pd.Series): The series to encode.
            columns (list): The list of columns.


        Returns:
            dict: A dictionary containing the values of the series encoded in one-hot.
        """
        return {f"{series.name}_{c}": [0] * len(series) for c in columns}

    @classmethod
    def get_files_bow(cls, series: str) -> str:
        """
        Get the name of the file containing the bag of words for a given series.

        Args:
            series (str): The name of the series.

        Returns:
            str: The name of the json file containing the bag of words.
        """
        # Dictionary associating the names of series to the names of bag of words files
        files = {"career": "work_field",
                 "work_field": "work_field", "from": "locations"}

        return files[series] + ".json"

    @classmethod
    def get_matching_keys(cls, keys, sub_strs: list) -> list:
        """
        Get the keys of `keys` that have a substring of the list `sub_strs`.

        Args:
            keys (list): list of strings
            sub_strs (list): list of strings to search in the keys

        Returns:
            list: The keys of `keys` that have a substring of the list `sub_strs`.
        """
        return [k for c in sub_strs for k in keys if c in k]

    @classmethod
    def isnan(cls, o: object) -> bool:
        """
        Get the keys of `keys` that have a substring of the list `sub_strs`.

        Args:
            o (object): The object to test.

        Returns:
            bool: True, if the object is NaN. False, otherwise.
        """
        return str(o).lower() == "nan"

    @classmethod
    def update_range_types(cls, dataframe: pd.DataFrame) -> dict:
        """
        Update the range of the labels of type "range".

        Args:
            dataset (pd.DataFrame): The input dataframe.

        Returns:
            dict: A dictionary containing the data types for each column.
        """
        # Open the file containing the types of labels.
        with open(cls.data_folder + "labels/types.json") as f:
            d_type = json.load(f)
            d_type_new = {}

            # Iterate over each key of the d_type dictionary
            for k in d_type.keys():

                # Check if the column is of type 'range'
                if "range" in d_type[k]:
                    k_max = dataframe[k].max()

                    # If the maximum value of the column is greater than 20,
                    # define the range of values on [0-100].
                    if k_max > 20:
                        d_type_new[k] = f"range [0 - {100}]"

                    # If the maximum value of the column is less than or equal to 12,
                    # define the range of values on [0-10].
                    elif k_max <= 12:
                        d_type_new[k] = f"range [0 - {10}]"

                    # Else, define the range of values on [0-?]
                    else:
                        d_type_new[k] = f"range [0 - ?]"

                # If the column is not of type 'range', copy the original data type.
                else:
                    d_type_new[k] = d_type[k]

        return d_type_new

    @classmethod
    def print_labels_info(cls, *args: str) -> None:
        """
        Display the information corresponding to each label passed as arguments.

        Args:
            *args (str): The names of the labels to display.
        """
        path_info = cls.data_folder + "labels/infos.json"
        path_type = cls.data_folder + "labels/types.json"
        unknown_labels = []

        # Open the files containing the necessary information.
        with open(path_info, "r") as f_info, open(path_type, "r") as f_type:
            types = json.load(f_type)
            infos = json.load(f_info)
            infos_keys = infos.keys()

            for k in args:
                # If the label does not exist, add it to the list of unknown labels.
                if k in infos_keys:
                    print(f"{k} ({types[k]}) -> {infos[k]}")

                else:
                    unknown_labels += [k]

            # Display the list of unknown labels
            if unknown_labels:
                print(f"label(s) inconnu(s): {unknown_labels}")

    @classmethod
    def summarize_null_data(cls, dataframe: pd.DataFrame, displaying: bool = False) -> pd.DataFrame:
        """
        Summarize the number of missing values (NaN) for each column of `dataframe`.

        Args:
            dataframe (pd.DataFrame): The DataFrame.
            displaying (bool, optional): If True, display the percentage of missing values by column. Default, False.

        Returns:
            pd.DataFrame: The DataFrame with the columns "label", "count" and "ratio".
        """
        # Create an empty DataFrame with the columns: "label", "count" and "percentage".
        df_null = pd.DataFrame(columns=["label", "count", "ratio"])
        lenght = len(dataframe)

        # Iterates over the columns of the dataframe.
        for label in dataframe:
            count = dataframe[label].isna().sum()
            ratio = round(100 * count / lenght)

            # Display the percentage.
            if displaying:
                print(label, f"{ratio:.0f}%")

            # Add the missing values data for this label.
            row_data = [{"label": label, "count": count, "ratio": ratio}]
            row = pd.DataFrame.from_records(row_data)
            df_null = pd.concat([df_null, row])

        return df_null.sort_values(by="count")
