from src.algorithm.token import Token


def get_dictionary_of_tokens(x):
    """
    Returns a dictionary of tokens from a given list of strings
    """
    tokens = []
    for string in x:
        if type(string) == float:
            string = str(string)

        for token in string.split():
            tokens.append(str(token))
    return tokens


def get_probability_table(dataset, dictionary, target_values):
    """
    Returns a dictionary of probabilities for a given list of strings
    """
    probs = []

    single_probs = []

    for value in target_values:
        docsj = dataset[dataset == value]

        pvij = len(docsj) / len(dataset)

        single_probs.append(pvij)

        textj = docsj.values.tolist()
        textj = [item for sublist in textj for item in sublist]

        n = len(textj)

        for word in dictionary:
            nk = textj.count(word)
            pvk = (nk + 1) / (n + len(dictionary))
            t = Token(word, pvk, value)
            probs.append(t)

        return single_probs, probs
