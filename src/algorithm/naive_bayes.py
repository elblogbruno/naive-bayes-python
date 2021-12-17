from src.algorithm.calcul_utils import *
import numpy as np
import time


class NaiveBayes():
    def __init__(self, verbose=0, lp_smoothing=1):
        self.lp_smoothing = lp_smoothing
        self.verbose = verbose

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
            start_time = time.time()

        total_vocabulary, class_count = get_vocabulary(self.X_train, y_train,  self.target_values)

        if self.verbose > 0:
            end_time = time.time()
            print("Dict tokens obtained in {} seconds".format(end_time - start_time))

        start = time.time()
        total_probs = get_probability_table(len(X_train), total_vocabulary, class_count,  self.target_values, self.lp_smoothing, self.verbose)
        end = time.time()

        if self.verbose > 0:
            print('Time taken to get_probability_table: {}'.format(end - start))
            print("Probabilities calculated")

        self.total_probs = total_probs

        if self.verbose > 1:
            print("total_probs: ", total_probs)

    def __single_prediction(self, X_row, prob_word, target_values=None):
        for word in X_row.split():
            for class_index in target_values:
                if word in self.total_probs.keys():
                    prob_word[class_index] += self.total_probs[word][class_index]

        max = 0
        max_index = 0

        for class_index in prob_word.keys():
            if prob_word[class_index] > max:
                max = prob_word[class_index]
                max_index = class_index

        return max_index

    def predict(self, X_test):
        if self.target_values is None:
            raise Exception("Target values not set")

        if len(X_test) < 1:
            raise Exception("X_test is empty")

        pred = []

        prob_word = {}

        for class_index in self.target_values:
            prob_word[class_index] = 0

        for i in range(len(X_test)):
            X_row = X_test.iloc[i].values.flatten()[0]
            if self.verbose > 1:
                print("Predicting value for row: {} {}".format(i, X_row))
            pred.append(self.__single_prediction(X_row, prob_word, self.target_values))

        return pred

    def score(self, y_train, y_pred):
        correct_predict = 0
        wrong_predict = 0
        for index, row in enumerate(y_train):  # for each row in the dataset
            result = y_pred[index]  # predict the row
            if self.verbose > 1:
                print("Scoring Row: {} - {}", index, row)
            if result == row:  # predicted value and expected value is same or not
                correct_predict += 1  # increase correct count
            else:
                wrong_predict += 1  # increase incorrect count
        accuracy = correct_predict / (correct_predict + wrong_predict)  # calculating accuracy
        return accuracy
