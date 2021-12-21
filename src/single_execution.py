from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

from algorithm.naive_bayes import NaiveBayes
import time


def single_execution_train_test(x, y, test_size, random_state, verbose, lp_smoothing, max_word_frequency, stop_words, filter_words,
                     use_custom_metrics=False):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    naive_bayes = NaiveBayes(verbose=verbose, lp_smoothing=lp_smoothing, max_word_frequency=max_word_frequency,
                             stop_words=stop_words, filter_words=filter_words)

    start = time.time()
    naive_bayes.fit(x_train, y_train)
    end = time.time()

    total_train_time = end - start
    print('Time taken to train: {}'.format(total_train_time))

    start = time.time()
    y_pred = naive_bayes.predict(x_test)
    end = time.time()

    total_predict_time = end - start
    print('Time taken to predict: {}'.format(total_predict_time))

    start = time.time()
    if use_custom_metrics:
        _accuracy_score = naive_bayes.accuracy_score(y_test, y_pred)
        _recall_score = naive_bayes.recall_score(y_test, y_pred)
        _precision_score = naive_bayes.precision_score(y_test, y_pred)
        _f1_score = naive_bayes.f1_score(y_test, y_pred)

        print('Accuracy Score (My Implementation): {}'.format(_accuracy_score))
        print('Precision Score (My Implementation): {}'.format(_precision_score))
        print('Recall Score (My Implementation): {}'.format(_recall_score))
        print('f1 Score (My Implementation): {}'.format(_f1_score))
    else:
        sk_accuracy_score = accuracy_score(y_test, y_pred)

        sk_recall_score = recall_score(y_test, y_pred)

        sk_precision_score = precision_score(y_test, y_pred)

        sk_f1_score = f1_score(y_test, y_pred)

        print('Accuracy Score (Sklearn):    {}'.format(sk_accuracy_score))
        print('Precision Score (Sklearn):  {}'.format(sk_precision_score))
        print('Recall Score (Sklearn):   {}'.format(sk_recall_score))
        print('f1 Score (Sklearn):  {}'.format(sk_f1_score))

    end = time.time()

    total_score_time = end - start
    print('Time taken to score: {}'.format(total_score_time))




    total_time = total_train_time + total_predict_time + total_score_time
    print('Total Time: {}'.format(total_time))
