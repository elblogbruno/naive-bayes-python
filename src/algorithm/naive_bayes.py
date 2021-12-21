from sklearn.base import BaseEstimator, ClassifierMixin

from algorithm.calcul_utils import *
import numpy as np
import time


class NaiveBayes(BaseEstimator, ClassifierMixin):

    def __init__(self, verbose=0, lp_smoothing=1, max_word_frequency=-1, stop_words=None, filter_words=None):

        stop_word_use = stop_words is not None
        filter_word_use = filter_words is not None

        print("Naive Bayes initialized with verbose: {} - lp_smoothing: {} - max_word_frequency {} - removing "
              "stop_words {} -  removing filter words {}".format(verbose, lp_smoothing, max_word_frequency,
                                                                 stop_word_use, filter_word_use))

        self.lp_smoothing = lp_smoothing
        self.verbose = verbose
        self.max_word_frequency = max_word_frequency
        self.stop_words = stop_words
        self.filter_words = filter_words

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"lp_smoothing": self.lp_smoothing, "max_word_frequency": self.max_word_frequency}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

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

        total_vocabulary = get_vocabulary(self.X_train, y_train, self.target_values, self.max_word_frequency,
                                          self.stop_words, self.filter_words)
        self.total_elements = len(self.y_train)

        print("Total vocabulary: ", len(total_vocabulary))

        if self.verbose > 0:
            end_time = time.time()
            print("Dict tokens obtained in {} seconds".format(end_time - start_time))

        start = time.time()

        unique, self.class_count = np.unique(self.y_train, return_counts=True)

        self.total_probs = get_probability_table(self.total_elements, total_vocabulary, self.class_count,
                                                 self.target_values, self.lp_smoothing, self.verbose)
        end = time.time()

        if self.verbose > 0:
            print('Time taken to get_probability_table: {}'.format(end - start))
            print("Probabilities calculated")

        if self.verbose > 1:
            print("total_probs: ", self.total_probs)

    def __single_prediction(self, X_row, prob_word, class_prob, target_values=None):
        """"
        :param X_row:
        :param prob_word:
        :param target_values:

        :return: 0 or 1
        """

        for word in X_row.split():
            word_prob = self.total_probs.get(word)
            if word_prob:
                for class_index in target_values:
                    word_conditional_prob = self.total_probs[word][class_index]
                    prob_word[class_index] += word_conditional_prob

        for class_index in target_values:
            prob_word[class_index] *= class_prob[class_index]

        return max(prob_word, key=prob_word.get)

    def predict(self, X_test):
        if self.target_values is None:
            raise Exception("Target values not set")

        if len(X_test) < 1:
            raise Exception("X_test is empty")

        pred = []

        prob_word = {}
        class_prob = {}

        for class_index in self.target_values:
            prob_word[class_index] = 0.0
            class_prob[class_index] = self.class_count[class_index] / self.total_elements  # We calculate the class
            # probability once.  P(C)

        X_test = X_test.values.flatten()

        for row in X_test:
            x_class = self.__single_prediction(row, prob_word, class_prob, self.target_values)
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
    def precision_score(self, y_test, y_pred):
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
            if y_test[index] == 1 and y_pred[index] == 1:  # true positive (tp) case - both are 1 in the test and
                # predicted
                tp += 1
            elif y_test[index] == 0 and y_pred[index] == 1:  # false positive as predicted positive but actual negative
                fp += 1

        return tp / (tp + fp)
