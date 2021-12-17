import numpy as np
from numpy import ndarray

from src.algorithm.token import Token


def get_vocabulary(x_train: np.ndarray, y_train: np.ndarray, target_values: np.ndarray):
    """
    Returns a dictionary of tokens from a given list of strings
    """
    dic = dict()

    number_of_classes = dict()
    for row_class in target_values:
        number_of_classes[row_class] = 0

    for row_index, row in enumerate(x_train):
        for token in row.split():
            row_class = y_train[row_index]

            number_of_classes[row_class] += 1
            if token not in dic.keys():
                dic[token] = [0, 0]
                dic[token][row_class] = 1
            else:
                dic[token][row_class] += 1

    return dic, number_of_classes


def get_probability_table(total_elements, vocabulary, number_of_classes, target_values, lp_smoothing=0, verbose=0):
    """
    Returns a dictionary of probabilities for a given list of strings
    """
    single_probs = np.array([], dtype=np.float)

    for class_index in target_values:
        class_probability = number_of_classes[class_index] / total_elements
        n = len(vocabulary)  # Number of documents in the class

        if verbose > 0:
            print("Number of tokens in class {}: {}".format(class_index, n))
            print("Number of tokens in vocabulary {} {}".format(len(vocabulary), total_elements))

        # Calculate the probability of each token in the vocabulary
        for i, word in enumerate(vocabulary):
            nk = vocabulary[word][class_index]  # Number of times the word appears in the class

            if lp_smoothing > 0:
                prob_condicionada = (nk + lp_smoothing) / (n + lp_smoothing * n)
            else:
                prob_condicionada = nk / n  # numobre de vegades que apareix la parauala en el text amb aquella
                # classe concreta  dividit entre  el numero de documents de la classe

            p_total = prob_condicionada * class_probability  # P(word|class) * P(class)

            vocabulary[word][class_index] = p_total

            if verbose > 1:
                print(i)

        return vocabulary
