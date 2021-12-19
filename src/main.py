from utils.dataset_utils import *
from algorithm.naive_bayes import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import time

path = '../data/FinalStemmedSentimentAnalysisDataset.csv'
dataset_train = load_dataset(path, include_header=True)
target_column = 'sentimentLabel'

print(dataset_train.isnull().sum())
dataset_train = dataset_train.dropna()
print(dataset_train.isnull().sum())

verbose = 0
lp_smoothing = 0

x = dataset_train.drop([target_column, 'tweetId', 'tweetDate'], 1)
y = dataset_train[target_column].values.ravel()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

naive_bayes = NaiveBayes(verbose=verbose, lp_smoothing=lp_smoothing, max_word_frequency=-1)

start = time.time()
naive_bayes.fit(x_train, y_train)
end = time.time()

print('Time taken to train: {}'.format(end - start))

start = time.time()
y_pred = naive_bayes.predict(x_test)
end = time.time()

print('Time taken to predict: {}'.format(end - start))

start = time.time()
_accuracy_score = naive_bayes.accuracy_score(y_test, y_pred)
sk_accuracy_score = accuracy_score(y_test, y_pred)

_recall_score = naive_bayes.recall_score(y_test, y_pred)
sk_recall_score = recall_score(y_test, y_pred)

_precision_score = naive_bayes.precision_score(y_test, y_pred)
sk_precision_score = precision_score(y_test, y_pred)


_f1_score = naive_bayes.f1_score(y_test, y_pred)
sk_f1_score = f1_score(y_test, y_pred)
end = time.time()


print('Time taken to score: {}'.format(end - start))
print('Accuracy Score:  My Implementation {} <-> sklearn {}'.format(_accuracy_score, sk_accuracy_score))
print('Precision Score: My Implementation {} <-> sklearn {}'.format(_precision_score, sk_precision_score))
print('Recall Score: My Implementation {} <-> sklearn {}'.format(_recall_score, sk_recall_score))
print('f1 Score: My Implementation {} <-> sklearn {}'.format(_f1_score, sk_f1_score))


# # cross validation
# from sklearn.model_selection import cross_val_score
#
# # do cross validation
# scores = cross_val_score(naive_bayes, x, y, cv=10, scoring='accuracy')
# print('Cross Validation Scores: {}'.format(scores))
# print('Cross Validation Mean: {}'.format(scores.mean()))
# print('Cross Validation Standard Deviation: {}'.format(scores.std()))
