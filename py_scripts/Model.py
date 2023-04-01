# Metadata
__author__ = "Eytan Levy, Guillaume Surleau, Manitas Bahri"
__date__ = "03/2023"


# Librairies import
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import json

from .PreProcessing import PreProcessing


class Model:
    """
    Tools to train and test a model.
    """

    def __init__(self):
        self.n_pca = 74
        self.preproc = PreProcessing()
        self.form_path = "../data/form.json"

    def set_form_path(self, form_path):
        """
        Set the path of the form.

        Args:
            form_path: The path of the form.
        """
        self.form_path = form_path

    def set_model(self, model):
        """
        Set the model.

        Args:
            model: The model to set.
        """
        self.model = model

    def predict_user(self) -> None:
        """
        Predict the user's match.
        """
        print("Loading user data...")

        with open(self.form_path, "r") as f:
            form = json.load(f)
            df_form = pd.DataFrame.from_dict(form, orient="index").T

            print("Preprocessing user data...")
            df_form = self.preproc.preprocessing_dataframe(df_form)
            
            # pca_cols = [f"PC{i}" for i in range(1, self.n_pca + 1)]
            # pca_renm = { df_form.columns[i] : pca_cols[i] for i in range(len(df_form.columns)) }
            
            # df_form = df_form.rename(columns=pca_renm)

            # print("Predicting user match...")
            # match_proba = self.model.predict_proba(df_form)
            # match_proba = match_proba[0, 1]

        # print("\nThe model predicts a match with a probability of: ", match_proba)

    def curve_range(self, scores: np.ndarray):
        """
        Compute the mean and the standard deviation of the scores.

        Args:
            scores (np.ndarray): The scores of the model.

        Returns:
            scores_mean (np.ndarray): The mean of the scores.
            fill_lw (np.ndarray): The lower bound of the scores.
            fill_up (np.ndarray): The upper bound of the scores.
        """
        scores_mean = scores.mean(axis=1)
        scores_std = scores.std(axis=1)

        fill_up = scores_mean + scores_std
        fill_lw = scores_mean - scores_std

        return scores_mean, fill_lw, fill_up

    def plot_validation_curve(self, train_score: np.ndarray, test_score: np.ndarray, model_name: str, param_name: str, param_range) -> None:
        """
        Plot the validation curve of a model.

        Args:
            train_score (np.ndarray): The training score of the model.
            test_score (np.ndarray): The test score of the model.
            model_name (str): The name of the model.
            param_name (str): The name of the parameter.
            param_range (np.ndarray): The range of the parameter.
        """
        train_mean, train_fill_lo, train_fill_up = self.curve_range(
            train_score)
        test_mean, test_fill_lo, test_fill_up = self.curve_range(test_score)

        plt.plot(param_range, train_mean, label="train curve", marker="o")
        plt.plot(param_range, test_mean, label="test curve", marker="o")

        plt.fill_between(param_range, train_fill_lo, train_fill_up, alpha=0.2)
        plt.fill_between(param_range, test_fill_lo, test_fill_up, alpha=0.2)

        plt.title(f"Validation Curve with {model_name}")
        plt.xlabel(param_name)
        plt.ylabel("F1 Micro Score")
        plt.legend(loc="lower right")
        plt.ylim(0, 1.1)
