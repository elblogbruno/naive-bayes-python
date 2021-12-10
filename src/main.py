from utils.dataset_utils import *
from algorithm.naive_bayes import *

path = '../data/FinalStemmedSentimentAnalysisDataset.csv'
dataset_train = load_dataset(path, include_header=True)
target_column = 'sentimentLabel'

x = dataset_train['tweetText']  # values converts it into a numpy array
y = dataset_train[target_column].values.ravel()

naive_bayes = NaiveBayes(dataset_train, target_column)
naive_bayes.fit(x, y)
score = naive_bayes.score(x, y)