import numpy as np
from numpy import ndarray

from src.algorithm.token import Token


def get_vocabulary(x_train: np.ndarray, y_train: np.ndarray, target_values: np.ndarray, max_word_frequency: int):
    """
    Returns a dictionary of tokens from a given list of strings
    """
    vocabulary = dict()

    number_of_classes = dict()
    number_of_total_count_classes = dict()

    for row_class in target_values:
        number_of_total_count_classes[row_class] = 0
        number_of_classes[row_class] = 0

    for row_index, row in enumerate(x_train):
        row_class = y_train[row_index]

        for token in row.split():
            number_of_total_count_classes[row_class] += 1

            if token not in vocabulary.keys():
                vocabulary[token] = [0] * len(target_values)
                vocabulary[token][row_class] = 1

                number_of_classes[row_class] += 1
            else:
                if max_word_frequency != -1 and vocabulary[token][row_class] >= max_word_frequency:
                    continue
                vocabulary[token][row_class] += 1

    return vocabulary, number_of_classes, number_of_total_count_classes


def get_probability_table(total_elements, vocabulary, duplicated_class_count, class_count, target_values, lp_smoothing=0, verbose=0):
    """
    Returns a dictionary of probabilities for a given list of strings
    """
    for class_index in target_values:
        class_probability = duplicated_class_count[class_index] / total_elements # P(C)

        n = duplicated_class_count[class_index]  # Number of documents in the class

        if verbose > 0:
            print("Number of tokens in class {}: {}".format(class_index, class_count[class_index]))
            print("Number of tokens in vocabulary {} {}".format(len(vocabulary), total_elements))

        # Calculate the probability of each token in the vocabulary
        for word in vocabulary:
            nk = vocabulary[word][class_index]  # Number of times the word appears in the class

            # Laplace smoothin
            prob_condicionada = (nk + lp_smoothing) / (n + len(vocabulary)*lp_smoothing)

            if verbose > 1:
                print("Probability of word {} in class {}: {}".format(word, class_index, prob_condicionada))

            p_total = prob_condicionada * class_probability # P(word|class) * P(class)

            vocabulary[word][class_index] = p_total

    return vocabulary
