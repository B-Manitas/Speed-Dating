"""
Le fichier `utils` contient des fonctions utiles pour le projet de data science.


Functions:
    filter_dataframe_by_substr(dataframe: pd.DataFrame, substr: str = "") -> pd.DataFrame
    
    get_dict_onehot(series: pd.Series, columns: list) -> dict:
    
    get_files_bow(series: str) -> str:

    isnan(o: object) -> bool

    update_range_types(dataframe: pd.DataFrame) -> dict

    print_labels_info(*args: str) -> None
"""

# Metadata
__author__ = "Eytan Levy, Guillaume Surleau, Manitas Bahri"
__date__ = "07/2023"


# Librairies
import json
import pandas as pd

# Constantes
DATA_FOLDER = "../data/"


def count_multiple_columns(dataframe: pd.DataFrame, column: str, blacklist: list = []):
    """
    Retourne le nombres d'occurences pour chaque colonne qui contient une sous-chaîne `str` dans son nom.

    Args:
        dataframe (pd.DataFrame): Le DataFrame contenant les données à traiter.
        column (str): La sous-chaîne de caractères à rechercher dans les noms des colonnes.
        blacklist (list, optional): Liste des noms de colonnes à exclure des résultats. Par défaut [].

    Returns:
        Tuple[List[int], List[str]]: Un tuple contenant deux listes : 
            la première contient les valeurs des occurences
            la seconde contient les noms des colonnes.
    """
    # Filtrer le DataFrame pour inclure seulement les lignes où la colonne spécifiée contient la sous-chaîne
    df = filter_dataframe_by_substr(dataframe, column)

    # Transforme les colonnes de df en lignes.
    col_to_line = df.melt(var_name='columns', value_name='index')

    # Calcule la table de fréquence des colonnes
    df = pd.crosstab(**col_to_line)  # type: ignore

    # Supprime les colonnes spécifiées dans la liste "blacklist"
    df.drop(columns=blacklist, inplace=True)

    data = df.iloc[1].sort_values(ascending=False)

    values = data.values
    labels = " ".join(data.keys()).replace(column, "").split()

    return values, labels


def filter_dataframe_by_substr(dataframe: pd.DataFrame, substr: str = "") -> pd.DataFrame:
    """
    Retourne le sous-ensemble du dataframe dont les noms des colonnes contienent le mot `substr`. 

    Args:
        dataframe (pd.DataFrame): Le DataFrame d'origine.
        substring (str): Le mot-clé utilisé pour filtrer les colonnes du DataFrame. Par défaut le mot vide "".

    Returns:
        pd.DataFrame: Le sous-ensemble de `dataframe` qui contient uniquement les colonnes dont le nom contient `substr`
                      triées dans l'ordre alphabétique.

    """
    columns = [column for column in dataframe.keys() if substr in column]
    columns.sort()

    return dataframe[columns]


def get_dict_onehot(series: pd.Series, columns: list) -> dict:
    """
    Retourne un dictionnaire contenant les valeurs de la série encodées en one-hot.

    Args:
        series (pd.Series): La série à encoder.
        columns (list): La liste des colonnes.

    Returns:
        dict: Un dictionnaire contenant les valeurs de la série encodées en one-hot.

    """
    return {f"{series.name}_{c}": [0] * len(series) for c in columns}


def get_files_bow(series: str) -> str:
    """
    Retourne le nom du fichier contenant le sac de mots pour une série donnée.

    Args:
        series (str): Le nom de la série.

    Returns:
        str: Le nom du fichier contenant le sac de mots.

    """
    # Dictionnaire associant les noms de séries aux noms de fichiers de sacs de mots
    files = {"career": "work_field",
             "work_field": "work_field", "from": "locations"}

    # Retourne le nom du fichier qui correspond à la série.
    return files[series] + ".json"


def get_matching_keys(keys, sub_strs: list) -> list:
    """
    Renvoie les clefs de `keys` qui ont une sous chaine de caractère de la liste `sub_strs`.

    Args:
      keys (list): liste de chaînes
      sub_strs (list): liste des chaînes à rechercher dans les clés
    """
    return [k for c in sub_strs for k in keys if c in k]


def isnan(o: object) -> bool:
    """
    Renvoie True si l'objet `o` est NaN. Sinon False.

    Args:
        o (object): L'objet à tester.

    Returns:
        bool: True, si l'objet est NaN. False, sinon.
    """
    return str(o).lower() == "nan"


def update_range_types(dataframe: pd.DataFrame) -> dict:
    """
    Mets à jours la plage des labels de types "range".

    Args:
        dataset (pd.DataFrame): Le dataframe d'entrée.

    Returns:
        dict: Un dictionnaire contenant les types de données pour chaque colonne.
    """

    # Ouvrir le fichier qui contient les types des labels.
    with open(DATA_FOLDER + "labels/types.json") as f:
        d_type = json.load(f)
        d_type_new = {}

        # Parcourir chaque clé du dictionnaire d_type
        for k in d_type.keys():

            # Vérifier si la colonne est de type 'range'
            if "range" in d_type[k]:
                k_max = dataframe[k].max()

                # Si la valeur maximale de la colonne est supérieure à 20,
                # défini la plage de valeurs sur [0-100].
                if k_max > 20:
                    d_type_new[k] = f"range [0 - {100}]"

                # Si la valeur maximale de la colonne est inférieure ou égale à 12,
                # défini la plage de valeurs sur [0-10].
                elif k_max <= 12:
                    d_type_new[k] = f"range [0 - {10}]"

                # Sinon, défini la plage de valeurs sur [0-?]
                else:
                    d_type_new[k] = f"range [0 - ?]"

            # Si la colonne n'est pas de type 'range', copier le type de données d'origine.
            else:
                d_type_new[k] = d_type[k]

    return d_type_new


def print_labels_info(*args: str) -> None:
    """
    Affiche les informations correspondantes pour chaque label passé en arguments.

    Args:
        *args (str): Les noms des labels à afficher.
    """
    path_info = DATA_FOLDER + "labels/infos.json"
    path_type = DATA_FOLDER + "labels/types.json"
    unknown_labels = []

    # Ouvre les fichiers contenant les informations nécessaires.
    with open(path_info, "r") as f_info, open(path_type, "r") as f_type:
        types = json.load(f_type)
        infos = json.load(f_info)
        infos_keys = infos.keys()

        for k in args:

            # Si le label existe, affiche les informations.
            if k in infos_keys:
                print(f"{k} ({types[k]}) -> {infos[k]}")

            else:
                unknown_labels += [k]

        # Affiche la liste des labels inconnus
        if unknown_labels:
            print(f"label(s) inconnu(s): {unknown_labels}")
