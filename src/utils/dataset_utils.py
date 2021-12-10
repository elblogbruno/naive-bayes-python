import pandas as pd
import numpy as np


def load_dataset(path: str, include_header = False) -> pd.core.frame.DataFrame:
    """
    Funció que donada la ruta del fitxer, retorna el dataset carregat en un objecte pandas
    :param path: String amb la ruta al fitxer
    :return: DataFrame amb les dades
    """
    try:
        if include_header:
            dataset = pd.read_csv(path, delimiter=';')
        else:
            dataset = pd.read_csv(path, header=None, delimiter=';')

    except:
        print("::-> ERROR : llegeix_taulell_doc()...")
    else:
        return dataset


def clean_dataset(dataset: pd.core.frame.DataFrame, type='default') -> pd.core.frame.DataFrame:
    """
    Funció que processa els valors "NULLS" d'un dataset aplicant l'estratègia o tractament especificat
    :param dataset: DataFrame amb la informació que es vol filtrar
    :param type: Indica el tipus de tractament dels valors "?" equivalents a "NaN"
    :return: DataFrame aplicant el metode especificat a totes les línies amb valors "?"
    """
    if type == 'default':
        # Deletes all rows with missing values
        return dataset[(dataset == '?').sum(axis=1) == 0]
    elif type == 'backfill':
        # Applies pandas method of backfilling
        dataset = dataset.replace(to_replace='?', value=np.NaN)
        return dataset.fillna(method='backfill', axis=1)
    elif type == 'mean':
        # Replaces missing values with the mean of the column
        dataset = dataset.replace(to_replace='?', value=np.NaN)
        return dataset.fillna(dataset.mean())
    else:
        print("::-> ERROR : clean_dataset - " + str(type) + " is not a valid option...")


def print_nulls(dataset: pd.core.frame.DataFrame, show_rows=True):
    """
    Donat un dataset, mostra totes aquelles línies que contenen files amb valors nulls
    :param dataset: DataFrame que es vol estudiar
    :param show_rows: Boleà que indica si es desitja mostrar les línies amb valors nulls. Per defecte sí es mostraran
    """
    counter = 0
    if show_rows is True:
        for index, row in enumerate(dataset.values):
            if '?' in row:
                print(str(index) + " " + str(row))
                counter += 1
    else:
        for row in dataset.values:
            if '?' in row:
                counter += 1
    print("\n---------------------")
    print(" Total NULLS = " + str(counter))
    print("---------------------\n")