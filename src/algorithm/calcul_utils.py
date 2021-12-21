import numpy as np


def get_vocabulary(x_train: np.ndarray, y_train: np.ndarray, target_values: np.ndarray, max_word_frequency: int,
                   stop_words: list, filter_words: list):
    """
    Returns a dictionary of tokens from a given list of strings
    """
    vocabulary = dict()
    vocabulary_delete = dict()

    for row_index, row in enumerate(x_train):
        row_class = y_train[row_index]

        for token in row.split():

            word_delete = vocabulary_delete.get(token)
            if len(vocabulary_delete) > 1 and word_delete:
                continue

            if filter_words:
                for filter_word in filter_words:
                    if filter_word in token:
                        continue

            if stop_words and token in stop_words:
                continue

            check_voc = vocabulary.get(token)

            if check_voc is None:
                vocabulary[token] = [0] * len(target_values)
                vocabulary[token][row_class] = 1
            else:
                if max_word_frequency != -1 and vocabulary[token][row_class] >= max_word_frequency:
                    # If the word appears more than the maximum frequency, it is not added to the vocabulary
                    del vocabulary[token]
                    vocabulary_delete[token] = 1
                    continue
                vocabulary[token][row_class] += 1

    return vocabulary


def get_probability_table(total_elements, vocabulary, class_count, target_values,
                          lp_smoothing=0, verbose=0):
    """
    Returns a dictionary of probabilities for a given list of strings
    """
    for class_index in target_values:
        number_of_class_elements = class_count[class_index]  # Number of documents in the class

        if verbose > 0:
            print("Number of tokens in class {}: {}".format(class_index, class_count[class_index]))
            print("Number of tokens in vocabulary {} {}".format(len(vocabulary), total_elements))

        # Calculate the probability of each token in the vocabulary
        for word in vocabulary:
            word_class_appears = vocabulary[word][class_index]  # Number of times the word appears in the class

            # If lp _smoothing is set to 0, the probability is calculated as the number of times the word appears in
            # the class
            conditional_probability = (word_class_appears + lp_smoothing) / (
                    number_of_class_elements + lp_smoothing * len(vocabulary))

            if verbose > 1:
                print("Probability of word {} in class {}: {}".format(word, class_index, conditional_probability))

            if conditional_probability == 0:
                vocabulary[word][class_index] = 0
                continue

            vocabulary[word][class_index] = np.log(conditional_probability)

    return vocabulary
