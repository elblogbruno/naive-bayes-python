from utils.dataset_utils import *


def preprocessing():
    path = '../data/FinalStemmedSentimentAnalysisDataset.csv'
    dataset_train = load_dataset(path, include_header=True)
    target_column = 'sentimentLabel'

    print(dataset_train.isnull().sum())
    dataset_train = dataset_train.dropna()
    print(dataset_train.isnull().sum())

    x = dataset_train.drop([target_column, 'tweetId', 'tweetDate'], 1)
    y = dataset_train[target_column].values.ravel()

    return x, y
