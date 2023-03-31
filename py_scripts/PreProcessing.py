# Metadata
__author__ = "Eytan Levy, Guillaume Surleau, Manitas Bahri"
__date__ = "07/2023"


# Librairies import
import pandas as pd
import numpy as np
import json
import re
from sklearn.decomposition import PCA

# Import custom modules
from .Utils import Utils
from .Standard import Standard
from .Selection import Selection


class PreProcessing (Standard, Selection):
    """
    This class contains all the methods to preprocess the data.
    """

    def __init__(self, data_folder: str = "../data/") -> None:
        super().__init__(data_folder)

    def apply_pca(self, dataframe: pd.DataFrame, n_components=.95) -> pd.DataFrame:
        """
        Get the dataframes with PCA applied.

        Args:
            dataframe (pd.DataFrame): The DataFrame.
            n_components (int, float, optional): The number of components to keep. Defaults to .95.

        Returns:
            pd.DataFrame: The DataFrame with the new columns.
        """
        pca = PCA(n_components=n_components)

        # Apply PCA.
        df_transform = dataframe.drop(columns=["id_individual", "p_id"])
        pca_data = pca.fit_transform(df_transform)
        labels = [f"PC{i}" for i in range(1, len(pca_data[0]) + 1)]

        # Create a new dataframe with the PCA data.
        pca_df = pd.DataFrame(pca_data, columns=labels,
                              index=dataframe.index)
        pca_df["id_individual"] = dataframe["id_individual"]
        pca_df["p_id"] = dataframe["p_id"]

        return pca_df

    def enc_multhot_dataframe(self, dataframe: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Apply the multiple hot encoding method to a column of the dataframe.
        The column `columns` is deleted.

        Args:
            dataframe (pd.DataFrame): The dataframe.
            columns (list): The columns to apply the multiple hot encoding.

        Returns:
            pd.DataFrame: The dataframe with the columns encoded.
        """
        df_multhot = dataframe.copy()

        for col in columns:
            df_multhot_col = self.get_multhot_columns(df_multhot[col])
            df_multhot = pd.concat([df_multhot, df_multhot_col], axis=1)
            df_multhot.drop(columns=[col], axis=1, inplace=True)

        return df_multhot

    def fill_nan_series(self, series: pd.Series, by=None, round: bool = False) -> pd.Series:
        """
        Replace NaN values in `series` with a given value.

        Args:
            series (pd.Series): The series to fill.
            by (optional): The value to use to replace NaN. If not specified, the mean of the series will be used.
            round (bool, optional): If True, the values will be rounded. The default is False.

        Returns:
            pd.Series: The series with NaN values replaced.
        """
        filled_s = series.fillna(by if by else series.mean())

        return filled_s.round() if round else filled_s

    def fill_nan_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Replace NaN values in a `dataframe`.

        Args:
            dataframe (pd.DataFrame): The dataframe to fill.

        Returns:
            pd.DataFrame: The dataframe with NaN values replaced.
        """
        # Fill with zeros this features
        to_fill_by_zeros = {"p_id", "exp_relation", "career_c",
                            "interrest_correlation"}

        # Round this features
        to_round = {"frequency_go_out", "age", "p_age", }

        # Open the file of the types of the labels of the database.
        with open("../data/labels/types.json", "r") as f:
            dtype = json.load(f)

            for col in dataframe.columns:
                # Replace NaN values with zeros
                if (col in to_fill_by_zeros) or ("categorical" in dtype.get(col, "")):
                    dataframe[col] = self.fill_nan_series(dataframe[col], 0)

                # Replace NaN values with the mean of the column and round the values
                if (col in to_round) or ("range" in dtype.get(col, "")):
                    dataframe[col] = self.fill_nan_series(
                        dataframe[col], round=True)

                # Replace NaN values with the mean of the column
                else:
                    dataframe[col] = self.fill_nan_series(dataframe[col])

        return dataframe

    def get_multhot_columns(self, series: pd.Series) -> pd.DataFrame:
        """
        Return the dataframe which correspond to the multiple hot encoding method applied to the column `series`.

        Args:
            series (pd.Series): The column to apply the multiple hot encoding.

        Returns:
            pd.DataFrame: The dataframe with the data encoded with the method.
        """
        # Tokenize the column
        series = series.str.lower()
        series = series.str.replace(r"[^a-z ]", " ", regex=True)
        series = series.str.split(" ")
        series = series.map(lambda x: [e for e in x if len(e) > 2], "ignore")

        # Get the name of the file of the bag of words.
        bow_file = Utils.get_files_bow(str(series.name))

        # Open the file of the bag of words of the work fields.
        with open(self.data_folder + "bag_of_words/" + bow_file, "r") as f:
            bag_of_word, fields = json.load(f)
            dict_onehot = Utils.get_dict_onehot(series, fields)

            # Iterates over each row of the DataFrame
            for i in range(len(series)):
                series_i = series.iloc[i]

                # If the value is not null.
                if not Utils.isnan(series_i):
                    work_i = " ".join(series_i)

                    # Iterates over each word of the bag of words.
                    # If the word is in the work field, the value of the column is set to 1.
                    for word in bag_of_word:
                        if re.search(word, work_i):
                            key = f"{series.name}_{bag_of_word[word]}"
                            dict_onehot[key][i] = 1

        return pd.DataFrame(dict_onehot, index=series.index)

    def load_dataset(self) -> pd.DataFrame:
        """
        Load the speed dating dataset.

        Returns:
            pd.DataFrame: The speed dating dataset.
        """
        path = self.data_folder + "speed_dating_data.csv"

        dataset = pd.read_csv(path, encoding="ISO-8859-1")

        # Preprocess the dataframe.
        dataset = self.rename_dataframe_columns(dataset)
        dataset = self.remove_dataframe_columns(dataset)

        return dataset

    def load_preproc_dataset(self, ratio_test=.2, rescaled: bool = True):
        """
        Load the preprocessed speed dating dataset.

        Args:
            ratio_test (float, optional): The ratio of the size of the test set. Defaults to .2.
            rescaled (bool, optional): If True, rescale the data, otherwise not. Defaults to True. 
            Allows to visualize the original data.

        Returns:
            pd.DataFrame, pd.DataFrame, pd.DataFrame: train, test and valid DataFrame.
        """
        # Load the dataset.
        dataset = self.load_dataset()

        # Clean the dataset.
        dataset, _ = self.remove_cols_with_null_data(dataset, 30)
        dataset = self.normalize_range_columns(dataset)

        # Split the dataset into train, test and valid.
        train, test = self.split_by_groups(dataset, ratio_test)
        test, valid = self.split_by_groups(test, .5)

        # Apply the preprocessing.
        train = self.preprocessing_dataframe(train, rescaled)
        test = self.preprocessing_dataframe(test, rescaled)
        valid = self.preprocessing_dataframe(valid, rescaled)

        return train, test, valid

    def preprocessing_dataframe(self, dataframe: pd.DataFrame, rescaled: bool = True) -> pd.DataFrame:
        """
        Preprocess the dataframe.

        Args:
            dataframe (pd.DataFrame): The dataframe to preprocess.
            rescaled (bool, optional): If True, rescale the data, otherwise not. Defaults to True.

        Returns:
            pd.DataFrame: The preprocessed dataframe.
        """
        # Define columns to encode, normalize and standardize.
        blacklist = ["career_c", "id_work_field"]
        to_mulhot_encode = ["work_field", "career", "from"]

        norm_columns = ["world_pref", "imprelig", "sports", "rate_prob_like"]
        norm_substr = ["p_important_", "p_rate_", "interest_", "rate_self_"]
        stand_columns = ["p_age", "p_race", "age", "exp_relation", "number_match",
                         "frequency_go_out"]

        # Apply the preprocessing on the dataset.
        dataset = self.remove_cols_personnality_snd(dataframe)
        dataset = self.enc_multhot_dataframe(dataset, to_mulhot_encode)
        dataset = self.fill_nan_dataframe(dataset)
        dataset.drop(columns=blacklist, inplace=True, axis=1)
        dataset.drop(dataset[dataset.p_id.isna()].index, inplace=True)

        # Rescaling the dataset.
        if rescaled:
            norm_substr = Utils.get_matching_keys(dataset.columns, norm_substr)
            stand_columns += norm_columns + norm_substr
            self.standardize_dataframe(dataset, stand_columns)

        return dataset

    def remove_cols_with_null_data(self, df_removed: pd.DataFrame, gt: int = 0):
        """
        Remove all columns containing a percentage of missing data greater than or equal to the threshold `gt`.

        Args:
            df_removed (pd.DataFrame): The input DataFrame to clean.
            gt (int, optional): The threshold of missing data percentage. All columns having a percentage of missing data
            greater than or equal to this threshold will be removed. By default, the threshold is set to 0.

        Returns:
            pd.DataFrame, pd.DataFrame: A tuple containing the cleaned dataframe and a dataframe summarizing the missing
        """
        # Compute the number of missing values per label.
        df_null = Utils.summarize_null_data(df_removed)

        # Select the labels that exceed the threshold `gt`.
        labels_to_remove = list(df_null[df_null["ratio"] >= gt]["label"])

        # Remove the columns that exceed this threshold.
        df_removed = df_removed.drop(columns=labels_to_remove)

        return df_removed, df_null

    def remove_cols_personnality_snd(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Remove the columns of the DataFrame containing secondary personality data.
        Secondary personality data is data that ends with a format of type: `n_x` with `n` a number and `x` a number or a
        letter `s`.

        Args:
            dataframe (pd.DataFrame): The DataFrame to clean.

        Returns:
            pd.DataFrame: The DataFrame without the removed secondary personality columns.
        """
        cols_to_drop = []

        # Itere over the columns of the dataframe.
        for cols in dataframe.keys():
            # The secondary personality data ends with the following format "n_x".
            if re.search("[0-9]_[0-9s]$", cols):
                cols_to_drop += [cols]

        # Remove the columns of the DataFrame containing secondary personality data.
        df_removed = dataframe.drop(columns=cols_to_drop)

        return df_removed

    def rename_dataframe_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Rename the columns of the `dataframe` from a local file.

        Args:
            dataset (pd.DataFrame): The input DataFrame to rename.

        Returns:
            pd.DataFrame: The dataframe with the renamed columns.
        """
        # Open the file which contains the labels to rename.
        with open(self.data_folder + "labels/to_rename.json", "r") as f:
            labels = json.load(f)
            dataframe = dataframe.rename(columns=labels)

        return dataframe

    def remove_dataframe_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Remove the columns of the `dataframe` from a local file.

        Args:
            dataset (pd.DataFrame): The input DataFrame to rename.

        Returns:
            pd.DataFrame: The dataframe with the removed columns.
        """
        # Open the file which contains the labels to remove.
        with open(self.data_folder + "labels/to_drop.csv", "r") as f:
            labels = f.read().replace(" ", "").split(",")
            dataframe = dataframe.drop(columns=labels)

        return dataframe
