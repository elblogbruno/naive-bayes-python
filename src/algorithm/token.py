class Token:
    def __init__(self, word, conditional_probability, target_value):
        self.word = word
        self.target_value = target_value
        self.conditional_probability = conditional_probability