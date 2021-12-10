from src.algorithm.calcul_utils import *


class NaiveBayes():
    def __init__(self, dataset, target_column):
        self.dataset = dataset
        self.target_column = target_column
        self.target_values = self.dataset[target_column].unique()

    def fit(self, X, y):
        dict_tokens = get_dictionary_of_tokens(X)
        single_probs, conditional_probabilities = get_probability_table(self.dataset, dict_tokens, self.target_values)
        self.single_probs = single_probs
        self.conditional_probability = conditional_probabilities

        print("single_probs: ", single_probs)
        print("conditional_probabilities: ", conditional_probabilities)

    def predict(self, X):
        pass

    def score(self, X, y):
        pass
