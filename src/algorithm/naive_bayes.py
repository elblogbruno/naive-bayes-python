from sklearn.base import BaseEstimator, ClassifierMixin

from src.algorithm.calcul_utils import *
import numpy as np
import time


class NaiveBayes(BaseEstimator, ClassifierMixin):

    def __init__(self, verbose=0, lp_smoothing=1, max_word_frequency = -1):
        print("Naive Bayes initialized with verbose: {} and lp_smoothing: {}".format(verbose, lp_smoothing))
        self.lp_smoothing = lp_smoothing
        self.verbose = verbose
        self.max_word_frequency = max_word_frequency

    def fit(self, X_train, y_train):
        if len(X_train) < 1:
            raise Exception("X_train is empty")

        if len(y_train) < 1:
            raise Exception("y_train is empty")

        if len(X_train) != len(y_train):
            raise Exception("X_train and y_train should have same length")

        self.X_train = X_train.values.flatten()
        self.y_train = y_train
        self.target_values = np.unique(y_train)

        if self.verbose > 0:
            print("Target values: ", self.target_values)
            print("Training data: ", len(self.X_train))
            start_time = time.time()

        total_vocabulary, class_count, duplicated_class_count = get_vocabulary(self.X_train, y_train, self.target_values, self.max_word_frequency)

        if self.verbose > 0:
            end_time = time.time()
            print("Dict tokens obtained in {} seconds".format(end_time - start_time))

        start = time.time()

        unique, counts = np.unique(self.y_train, return_counts=True)

        self.total_probs = get_probability_table(len(y_train), total_vocabulary, duplicated_class_count, counts,  self.target_values, self.lp_smoothing, self.verbose)
        end = time.time()

        if self.verbose > 0:
            print('Time taken to get_probability_table: {}'.format(end - start))
            print("Probabilities calculated")

        if self.verbose > 1:
            print("total_probs: ", self.total_probs)

    def __single_prediction(self, X_row, prob_word, target_values=None):
        """"
        :param X_row:
        :param prob_word:
        :param target_values:

        :return: 0 or 1
        """

        for word in X_row.split():
            if word in self.total_probs.keys():
                for class_index in target_values:
                    if self.verbose > 1:
                        print("Calculating probability for word: {} and class: {}".format(word, class_index))

                    if self.total_probs[word][class_index] == 0:
                        prob_word[class_index] = 0
                    else:
                        prob_word[class_index] += np.log(self.total_probs[word][class_index])

        if self.verbose > 1:
            print("Probability for each class: ", prob_word)

        return max(prob_word, key=prob_word.get)

    def predict(self, X_test):
        if self.target_values is None:
            raise Exception("Target values not set")

        if len(X_test) < 1:
            raise Exception("X_test is empty")

        pred = []

        prob_word = {}

        for class_index in self.target_values:
            prob_word[class_index] = 0.0

        X_test = X_test.values.flatten()

        for row in X_test:
            if self.verbose > 1:
                print("Predicting value for row: {}".format(row))

            x_class = self.__single_prediction(row, prob_word, self.target_values)
            pred.append(x_class)

        return pred

    def accuracy_score(self, y_test, y_pred):
        """
        Calculates accuracy score for the given test data and predicted values
        :param y_test:
        :param y_pred:
        :return:
        """
        if len(y_test) != len(y_pred):
            raise Exception("y_test and y_pred should have same length")

        correct = 0
        for index, row in enumerate(y_test):
            if self.verbose > 1:
                print("Testing Row: {} - {}", index, row)

            if y_pred[index] == row:
                correct += 1

        return correct / len(y_test)

    def f1_score(self, y_train, y_pred):
        """
        Calculates the f1 score F1 = 2 * (precision * recall) / (precision + recall)
        :param y_train:
        :param y_pred:
        :return:
        """
        if len(y_train) != len(y_pred):
            raise Exception("y_train and y_pred should have same length")

        precision = self.precision_score(y_train, y_pred)
        recall = self.recall_score(y_train, y_pred)

        F1 = 2 * (precision * recall) / (precision + recall)

        return F1

    def recall_score(self, y_test, y_pred):
        """
        Calculates the recall score tp / (tp + fn) where tp is the number of true positives and fn is the number of false negatives
        :param y_train:
        :param y_pred:
        :return:
        """
        if len(y_test) != len(y_pred):
            raise Exception("y_train and y_pred should have same length")

        fn = 0.0
        tp = 0.0

        for (index, val) in enumerate(y_test):
            if y_test[index] == 1 and y_pred[index] == 1:  # true positive (tp) case - both are 1 in the test and
                # predicted
                tp += 1
            elif y_test[index] == 1 and y_pred[index] == 0:  # false negative (fn) case - test is 1 but predicted 0
                fn += 1

        return tp / (tp + fn)

    # https://github.com/varunshenoy/simple-metrics
    def precision_score(self, y_test , y_pred):
        """
        Calculates precision score tp / (tp + fp) where tp is the number of true positives and fp is the number of false positives.
        :param y_test:
        :param y_pred:
        :return:
        """
        if len(y_test) != len(y_pred):
            raise Exception("y_test and y_pred should have same length")

        fp = 0.0
        tp = 0.0

        for (index, val) in enumerate(y_test):
            if y_test[index] == 1 and y_pred[index] == 1: # true positive (tp) case - both are 1 in the test and
                # predicted
                tp += 1
            elif y_test[index] == 0 and y_pred[index] == 1: # false positive as predicted positive but actual negative
                fp += 1

        return tp / (tp + fp)
