from nltk.corpus import stopwords
import argparse

from grid_and_cross_val_test import cross_val_score_and_grid
from preprocessing import preprocessing
from single_execution import single_execution_train_test

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--verbose", type=int, default=0, help="verbosity level of the program") # 0, 1, 2
ap.add_argument("-s", "--smoothing", type=float, default=1, help="Laplace smoothing parameter (default: 1)")
ap.add_argument("-m", "--max_word_frequency", type=int, default=-1, help="Maximum word frequency (default: -1 means no limit)")
ap.add_argument("-r", "--remove_stop_words",  help="Remove stopwords", action='store_true')
ap.add_argument("-f", "--remove_filter_words",  help="Remove filter words", action='store_true')
ap.add_argument("-dr", "--dont_remove_stop_words",  help="Does not remove stopwords", action='store_false')
ap.add_argument("-df", "--dont_remove_filter_words", help="Does not remove filter words", action='store_false')
ap.add_argument("-t", "--test_size", type=float, default=0.2, help="Test size (default: 0.2)")
ap.add_argument("-rs", "--random_state", type=int, default=42, help="Random state (default: 42)")
ap.add_argument("-c", "--cross_validation", type=bool, default=False, help="Perform Cross validation (default: False)")
ap.add_argument("-g", "--grid_search", type=bool, default=False, help="Perform Grid search (CPU Intensive) (default: False)")
args = vars(ap.parse_args())

verbose = args['verbose'] # verbose = 0 means minimal debug output. verbose = 1 means more debug output.
lp_smoothing = args['smoothing'] # Laplace smoothing parameter for the naive Bayes classifier.
max_word_frequency = args['max_word_frequency'] # Maximum word frequency for the naive Bayes classifier.
remove_stop_words = args['remove_stop_words'] # Set to True to remove stopwords from the dictionary.
remove_filter_words = args['remove_filter_words'] # Set to True to remove filter words from the dictionary
test_size = args['test_size'] # Test size for the train/test split.
random_state = args['random_state'] # Random state for the train/test split.
make_cross_validation = args['cross_validation'] # Set to True to perform cross validation.
make_grid_search = args['grid_search'] # Set to True to perform grid search.

x, y = preprocessing()


if remove_filter_words:
    filter_words = set(['https', 'http', 'co', 'com', 'amp', '@', '#', '$', '%', '^', '&', '*'])
else:
    filter_words = None

if remove_stop_words:
    stop_words = set(stopwords.words('english'))
else:
    stop_words = None

if make_cross_validation:
    print("Performing cross validation...")
    cross_val_score_and_grid(x, y, verbose=verbose, lp_smoothing=lp_smoothing, max_word_frequency=max_word_frequency, stop_words=stop_words, filter_words=filter_words, make_cross_validation=make_cross_validation, make_grid_search=make_grid_search)
else:
    print("Performing single execution...")
    single_execution_train_test(x, y, test_size, random_state=random_state, verbose=verbose, lp_smoothing=lp_smoothing, max_word_frequency=max_word_frequency, stop_words=stop_words, filter_words=filter_words)

