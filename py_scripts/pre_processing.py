"""
Le fichier `pre-processing` contient un ensemble de fonctions utilisé pour le pré-traitements des données.


Functions:
    enc_multhot_dataframe(dataframe: pd.DataFrame, columns: list) -> pd.DataFrame

    fill_nan_series(series: pd.Series, by=None) -> pd.Series
    
    fill_nan_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame

    get_multhot_columns(series: pd.Series) -> pd.DataFrame

    load_dataset() -> pd.DataFrame

    load_preproc_dataset(ratio_test: float = .2, rescaled: bool = True, split_y: bool = True)

    normalize_dataframe(dataframe: pd.DataFrame, columns: list)
    
    normalize_range(value: int, min: int, max: int, born_max: int = 10) -> int
    
    normalize_range_columns(dataframe: pd.DataFrame) -> pd.DataFrame

    preprocessing_dataframe(dataframe: pd.DataFrame, rescaled: bool = True) -> pd.DataFrame
    
    remove_cols_with_null_data(df_removed: pd.DataFrame, gt: int = 0) -> (pd.DataFrame, pd.DataFrame)

    remove_cols_personnality_snd(dataframe: pd.DataFrame) -> pd.DataFrame
    
    rename_dataframe_columns(dataframe: pd.DataFrame) -> pd.DataFrame
    
    remove_dataframe_columns(dataframe: pd.DataFrame) -> pd.DataFrame 

    split_X_y(dataframe: pd.DataFrame, index: list)
    
    summarize_null_data(dataframe: pd.DataFrame, displaying: bool = False) -> pd.DataFrame 
"""

# Metadata
__author__ = "Eytan Levy, Guillaume Surleau, Manitas Bahri"
__date__ = "07/2023"


# Librairies
import pandas as pd
import numpy as np
import json
import re

# Custom packages
import py_scripts.utils as ut

# Constantes
DATA_FOLDER = "../data/"


def enc_multhot_dataframe(dataframe: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Applique la méthode de multiple hot encoding à une colonne du dataframe. 
    La colonne `columns` est supprimée.

    Args:
        dataframe (pd.DataFrame): Le dataframe.
        column (list): La liste des colonnes à transformer

    Returns:
        pd.DataFrame: Le dataframe avec les colonnes encodées.

    """
    df_multhot = dataframe.copy()

    for col in columns:
        df_multhot_col = get_multhot_columns(df_multhot[col])
        df_multhot = pd.concat([df_multhot, df_multhot_col], axis=1)
        df_multhot.drop(columns=[col], axis=1, inplace=True)

    return df_multhot


def fill_nan_series(series: pd.Series, by=None) -> pd.Series:
    """
    Remplace les valeurs NaN dans une série par une valeur donnée.

    Args:
        series (pd.Series): La série à remplir.
        by (optional): La valeur à utiliser pour remplacer les NaN.
            Si non spécifiée, la moyenne de la série sera utilisée.

    Returns:
        pd.Series: La série avec les valeurs NaN remplacées.

    """
    return series.fillna(by if by else series.mean())


def fill_nan_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace les valeurs NaN dans `dataframe`.

    Args:
        dataframe (pd.DataFrame): Le DataFrame à remplir.

    Returns:
        pd.DataFrame: Le DataFrame avec les valeurs NaN remplacées.

    """
    # Colonnes à remplir avec des zéros
    to_fill_by_zeros = {"p_id", "exp_relation", "career_c",
                        "interrest_correlation", }

    # Ouvre le fichier des types des labels de la base de données.
    with open("../data/labels/types.json", "r") as f:
        dtype = json.load(f)

        for col in dataframe.columns:
            # Remplace les valeurs NaN avec des zéros si la colonne est dans la liste
            # ou si elle est de type catégoriel
            if (col in to_fill_by_zeros) or ("categorical" in dtype.get(col, "")):
                dataframe[col] = fill_nan_series(dataframe[col], 0)

            # Sinon, remplacement des valeurs NaN avec la moyenne de la colonne
            else:
                dataframe[col] = fill_nan_series(dataframe[col])

    return dataframe


def get_multhot_columns(series: pd.Series) -> pd.DataFrame:
    """
    Renvoie le dataframe qui correspond à l'encodage par la méthode du multiple hot encoding appliqué à la colonne séries.    

    Args:
        series: La colonne à appliquer le multiple hot encoding.

    Returns:
        pd.DataFrame : Le dataframe avec les données encodées avec la méthode.

    """
    # Transforme en minuscule
    series = series.str.lower()

    # Supprimes les caractères spéciaux
    series = series.str.replace(r"[^a-z ]", " ", regex=True)

    # Tokenize les domaines de travails
    series = series.str.split(" ")

    # Supprime les mots ayant moins de deux caractère.
    series = series.map(lambda x: [e for e in x if len(e) > 2], "ignore")

    # Récupère le nom du fichier du sac de mot.
    bow_file = ut.get_files_bow(str(series.name))

    # Ouvre le fichier bag of word des domaines de travails.
    with open(DATA_FOLDER + "bag_of_words/" + bow_file, "r") as f:
        bag_of_word, fields = json.load(f)
        dict_onehot = ut.get_dict_onehot(series, fields)

        # Parcours chaque ligne du DataFrame
        for i in range(len(series)):
            series_i = series.iloc[i]

            # Si la valeur est non nul.
            if not ut.isnan(series_i):
                work_i = " ".join(series_i)

                for word in bag_of_word:
                    if re.search(word, work_i):
                        key = f"{series.name}_{bag_of_word[word]}"
                        dict_onehot[key][0] = 1

    return pd.DataFrame(dict_onehot, index=series.index)


def load_dataset() -> pd.DataFrame:
    """
    Charge le jeu de données de speed dating.

    Returns:
        pd.DataFrame: Le DataFrame speed dating.
    """

    path = DATA_FOLDER + "speed_dating_data.csv"

    dataset = pd.read_csv(path, encoding="ISO-8859-1")

    # Pré-traite le dataframe.
    dataset = rename_dataframe_columns(dataset)
    dataset = remove_dataframe_columns(dataset)

    return dataset


def load_preproc_dataset(ratio_test: float = .2, rescaled: bool = True, split_y: bool = True):
    """
    Charge le jeu de données néttoyé et normalisé de speed dating.

    Args:
        ratio_test (float, optional): Le ratio de la taille du jeu de test. Par défauts, 0.2.
        rescaled (boolean, optional): Si True rescaled les données, sinon non. 
        Permets de visualiser les données originales. Par défauts, True.
        split_y (boolean, optional): Si True renvoie deux dataframe, le premier correcponds 
        aux inputs et le second aux targets. Par défauts, True.

    Returns:
        (pd.DataFrame, pd.DataFrame), (pd.DataFrame, pd.DataFrame): (x_train, y_train), (x_test, y_test).
        pd.DataFrame, pd.DataFrame: train, test.
    """
    # Charge le dataset.
    dataset = load_dataset()

    # Sépare le dataset
    groups = dataset["id_group"]
    df_x = dataset["match"]
    df_y = dataset.drop(
        columns=["id_group", "match", "decision", "p_decision"])

    # Clean le dataset.
    dataset, _ = remove_cols_with_null_data(dataset, 30)
    dataset = normalize_range_columns(dataset)

    # Split le dataset.
    spliter = GroupShuffleSplit(test_size=ratio_test)
    i_train, i_test = next(spliter.split(df_y, df_x, groups))

    if split_y:
        # Sépare les inputs et targets du dataframe.
        x_train, y_train = split_X_y(dataset, i_train)
        x_test, y_test = split_X_y(dataset, i_test)

        # Applique le preprocessing sur les dataset d'entrainement et de test séparément.
        x_train = preprocessing_dataframe(x_train, rescaled)
        x_test = preprocessing_dataframe(x_test, rescaled)

        return (x_train, y_train), (x_test, y_test)

    else:
        train = dataset.iloc[i_train]
        test = dataset.iloc[i_test]

        # Applique le preprocessing sur les dataset d'entrainement et de test séparément.
        train = preprocessing_dataframe(train, rescaled)
        test = preprocessing_dataframe(test, rescaled)

        return train, test


def normalize_dataframe(dataframe: pd.DataFrame, columns: list):
    """
    Cette fonction prend une trame de données et une liste de colonnes et renvoie une trame de données
    avec les colonnes normalisées

    Args:
      dataframe (pd.DataFrame): Le dataframe.
      columns (list): La liste des colonnes à normaliser.
    """
    df_norm = dataframe[columns]
    df_norm = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())
    dataframe.update(df_norm)


def normalize_range(value: int, min: int, max: int, born_max: int = 10) -> int:
    """
    Normalise une valeur en la convertissant en une valeur comprise entre 0 et un maximum borné donné.

    Args:
        value (int): La valeur à normaliser.
        min (int): La valeur minimale de la plage de valeurs.
        max (int): La valeur maximale de la plage de valeurs.
        born_max (int, optional): La borne supérieure de la plage normalisée. Par défaut, 10.

    Returns:
        int: La valeur normalisée.
    """

    if np.isnan(value):
        return value

    if value < min:
        return 0

    if value > max:
        return born_max

    return round(born_max * value / max)


def normalize_range_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les valeurs d'une colonne de type "range" du `dataframe` vers un intervalle standard de [0-10].

    Args:
        dataset (pd.DataFrame): Le DataFrame d'entrée.

    Returns:
        pd.DataFrame: Le DataFrame transformé avec les valeurs normalisées.
    """

    # Ouvre le fichier qui contient les types des labels
    with open(DATA_FOLDER + "labels/types.json", "r") as f:
        d_type = json.load(f)

        # Pour chaque label du dataframe
        for key in dataframe.keys():

            # Si le label est de type "range".
            if re.match(r"range \[[0-9?]+ \- [0-9?]+\]", d_type[key]):

                # Récupère les valeurs minimale et maximale de l'intervalle.
                min, max = re.findall(r"[0-9?]+", d_type[key])

                # Renvoie un warning, si la valeur minimale ou maximale est inconnue
                if min == "?" or max == "?":
                    warning = f"Warning: '{key}' possède un min ou max inconnue -> [{min} - {max}]."
                    print(warning)

                # Sinon, normalize l'intervalle.
                else:
                    dataframe[key] = dataframe[key].map(
                        lambda x: normalize_range(x, int(min), int(max)))

    return dataframe


def preprocessing_dataframe(dataframe: pd.DataFrame, rescaled: bool = True) -> pd.DataFrame:
    """
    Renvoie le dataframe néttoyer avec les méthodes de pré-proccesing.

    Args:
        dataframe (pd.DataFrame): Le dataframe à nettoyer.
        rescaled (boolean, optional): Si True rescaled les données, sinon non. 
            Permets de visualiser les données originales. Par défaut, True.

    Returns:
        Le dataframe néttoyé.
    """
    # Déclaration des variables nécéssaires.
    blacklist = ["career_c", "id_work_field"]
    to_mulhot_encode = ["work_field", "career", "from"]

    norm_columns = ["world_pref", "imprelig", "sports", "rate_prob_like"]
    norm_substr = ["p_important_", "p_rate_", "interest_", "rate_self_"]
    stand_columns = ["p_age", "p_race", "age", "exp_relation", "number_match",
                     "frequency_go_out"]

    # Applique le préprocessing sur le dataset.
    dataset = remove_cols_personnality_snd(dataframe)
    dataset = enc_multhot_dataframe(dataset, to_mulhot_encode)
    dataset = fill_nan_dataframe(dataset)
    dataset.drop(columns=blacklist, inplace=True, axis=1)
    dataset.drop(dataset[dataset.p_id.isna()].index, inplace=True)

    # Re-scaling du dataset.
    if rescaled:
        norm_substr = ut.get_matching_keys(dataset.columns, norm_substr)
        normalize_dataframe(dataset, norm_columns + norm_substr)
        standardize_dataframe(dataset, stand_columns)

    return dataset


def remove_cols_with_null_data(df_removed: pd.DataFrame, gt: int = 0):
    """
    Supprime toutes les colonnes contenant un pourcentage de données manquantes supérieur ou égal au seuil `gt`.

    Args:
        dataset (pd.DataFrame): Le DataFrame d'entrée à nettoyer.
        gt (int, optional): Le seuil de pourcentage de données manquantes. Toutes les colonnes ayant un pourcentage de données 
        manquantes supérieur ou égal à ce seuil seront supprimées. Par défaut, le seuil est fixé à 0.

    Returns:
        pd.DataFrame, pd.DataFrame: Un tuple contenant le dataframe nettoyé et un dataframe résumant les données 
        manquantes pour chaque colonne.
    """

    # Cacule le nombre de valeurs manquantes par label.
    df_null = summarize_null_data(df_removed)

    # Séléctionne les labels qui dépasse le seuil `gt`.
    labels_to_remove = list(df_null[df_null["ratio"] >= gt]["label"])

    # Supprime les colonnes qui dépasse ce seuil.
    df_removed = df_removed.drop(columns=labels_to_remove)

    return df_removed, df_null


def remove_cols_personnality_snd(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes du DataFrame contenant des données de personnalité secondaire.
    Les données de personnalité secondaire sont celle qui se terminne par un format de type: 
    `n_x` avec `n` un chiffre et `x` un chffre ou une lettre `s`.

    Args:
        dataframe (pd.DataFrame): Le DataFrame à nettoyer.

    Returns:
        pd.DataFrame: Le DataFrame sans les colonnes de personnalité supprimées.

    """
    cols_to_drop = []

    # Parcours les colonnes du dataframe.
    for cols in dataframe.keys():
        # Les données de personnalit" secondaire se termine par le format suivant "n_x".
        if re.search("[0-9]_[0-9s]$", cols):
            cols_to_drop += [cols]

    # Supprime les colonnes du DataFrame contenant des données de personnalité secondaire.
    df_removed = dataframe.drop(columns=cols_to_drop)

    return df_removed


def rename_dataframe_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Renomme les colonnes du `dataframe` à partir d'un fichier local.

    Args:
        dataset (pd.DataFrame): Le DataFrame d'entrée à renommer.

    Returns:
        pd.DataFrame: Le dataframe avec les colonnes renommées.
    """

    # Ouvre le fichier qui contient des labels à renommer.
    with open(DATA_FOLDER + "labels/to_rename.json", "r") as f:
        labels = json.load(f)
        dataframe = dataframe.rename(columns=labels)

    return dataframe


def remove_dataframe_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes inutilisées, listées dans un fichier local, du `dataframe`.

    Args:
        dataset (pd.DataFrame): Le DataFrame d'entrée à nettoyer.

    Returns:
        pd.DataFrame: Un dataframe avec les colonnes inutilisées supprimées.
    """

    # Ouvre le fichier qui contient des labels à renommer.
    with open(DATA_FOLDER + "labels/to_drop.csv", "r") as f:
        labels = f.read().replace(" ", "").split(",")
        dataframe = dataframe.drop(columns=labels)

    return dataframe


def split_X_y(dataframe: pd.DataFrame, index: list):
    """
    Sépare les valeurs d'input et d'output du `dataframe`.

    Args:
        dataframe (pd.DataFrame): Le DataFrame à séparer.
        index (list): La listes des index à garder du `dataframe`. 

    Returns:
        pd.DataFrame, pd.DataFrame : 
            Le premier dataframe correspond au input. 
            Le second dataframe correspond au output.
    """
    x = dataframe.iloc[index]
    x = x.drop(columns=["match", "decision", "p_decision"])

    y = dataframe.iloc[index]["match"].to_frame()
    return x, y


def standardize_dataframe(dataframe: pd.DataFrame, columns: list):
    """
    Cette fonction standardise les colonnes d'un dataframe.

    Args:
        dataframe (pd.DataFrame): Le dataframe.
        columns (list): Liste des colonnes à standardiser.
    """
    df_stand = dataframe[columns]
    df_stand = (df_stand - df_stand.mean()) / df_stand.std()
    dataframe.update(df_stand)


def summarize_null_data(dataframe: pd.DataFrame, displaying: bool = False) -> pd.DataFrame:
    """
    Compte le nombre de valeurs manquantes (NaN) pour chaque colonne de `dataframe`. 

    Args:
        dataframe (pd.DataFrame): Le DataFrame pandas.
        displaying (bool, optional): Si True, affiche le pourcentage de valeurs manquantes par colonne. Par défault, False.

    Returns:
        pd.DataFrame: Le dataframe pandas avec les colonnes "label", "count" et "ratio".
    """
    # Créer un DataFrame vide avec les colonnes: "label", "count" et "percentage".
    df_null = pd.DataFrame(columns=["label", "count", "ratio"])
    lenght = len(dataframe)

    # Pour chaque label du DataFrame.
    for label in dataframe:
        count = dataframe[label].isna().sum()
        ratio = round(100 * count / lenght)

        # Affiche la proportion.
        if displaying:
            print(label, f"{ratio:.0f}%")

        # Ajoute les données des valeurs manquantes pour ce label.
        row_data = [{"label": label, "count": count, "ratio": ratio}]
        row = pd.DataFrame.from_records(row_data)
        df_null = pd.concat([df_null, row])

    return df_null.sort_values(by="count")


def standardize_dataframe(dataframe: pd.DataFrame, columns: list):
    """
    Cette fonction standardise les colonnes d'un dataframe.

    Args:
      dataframe (pd.DataFrame): Le dataframe.
      columns (list): Liste des colonnes à standardiser.
    """
    df_stand = dataframe[columns]
    df_stand = (df_stand - df_stand.mean()) / df_stand.std()
    dataframe.update(df_stand)
