from utils.dataset_utils import *
from algorithm.naive_bayes import *
from sklearn.model_selection import train_test_split
import time

path = '../data/FinalStemmedSentimentAnalysisDataset.csv'
dataset_train = load_dataset(path, include_header=True)
target_column = 'sentimentLabel'

print(dataset_train.isnull().sum())
dataset_train = dataset_train.dropna()
print(dataset_train.isnull().sum())

verbose = 1
x = dataset_train.drop([target_column, 'tweetId', 'tweetDate'], 1)
y = dataset_train[target_column].values.ravel()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

naive_bayes = NaiveBayes(verbose=verbose)

start = time.time()
naive_bayes.fit(x_train, y_train)
end = time.time()

print('Time taken to train: {}'.format(end - start))

start = time.time()
y_pred = naive_bayes.predict(x_test)
end = time.time()

print('Time taken to predict: {}'.format(end - start))

start = time.time()
score = naive_bayes.score(y_test, y_pred)
end = time.time()

print('Time taken to score: {}'.format(end - start))
print('Score: ' + str(score))