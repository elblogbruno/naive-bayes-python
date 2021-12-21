from algorithm.naive_bayes import *

# gridsearchcv
from sklearn.model_selection import GridSearchCV, cross_val_score


def cross_val_score_and_grid(x, y, verbose, lp_smoothing, max_word_frequency, stop_words, filter_words, make_cross_validation=False, make_grid_search=False):
    if make_grid_search:
        print("Performing GridSearchCV... Will take some time...")

        # Create the parameter grid based on the parameters my algorithm accepts
        parameter = {'lp_smoothing': [0, 0.5, 1], 'max_word_frequency': [-1, 100, 1000, 10000],
                     'stop_words': [None, stop_words], 'filter_words': [None, filter_words]}

        naive_bayes = NaiveBayes(verbose=0) # verbose=0 means no output to console during grid search process

        grid = GridSearchCV(naive_bayes, parameter, cv=5, scoring='accuracy', verbose=2) # verbose=2 for more info

        grid.fit(x, y) # Fit the model

        print('Best Parameters: {}'.format(grid.best_params_))
        print('Best Score: {}'.format(grid.best_score_))

    if make_cross_validation:
        print('Running cross_val_score...')

        naive_bayes = NaiveBayes(verbose=verbose, lp_smoothing=lp_smoothing, max_word_frequency=max_word_frequency, stop_words=stop_words, filter_words=filter_words)
        scores = cross_val_score(naive_bayes, x, y, cv=5, scoring='accuracy', verbose=1) # verbose=1 for more info

        print('Cross Validation Scores: {}'.format(scores))
        print('Cross Validation Mean: {}'.format(scores.mean()))
        print('Cross Validation Standard Deviation: {}'.format(scores.std()))
