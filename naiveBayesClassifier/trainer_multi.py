from naiveBayesClassifier.train_data_multi import TrainDataMulti

class TrainerMulti(object):

    """docstring for Trainer"""
    def __init__(self, tokenizer):
        super(TrainerMulti, self).__init__()
        self.tokenizer = tokenizer
        self.trained_data = TrainDataMulti(1)
        self.data = self.trained_data

    def train(self, content, class_name):
        """
        enhances trained data using the given text and class
        """
        self.data.increase_class(class_name)

        tokens = self.tokenizer.tokenize(content)

        for token in tokens:
            token = self.tokenizer.remove_stop_words(token)
            token = self.tokenizer.remove_punctuation(token)
            self.data.increase_token(token, class_name)
            self.data.increase_token_in_class(class_name)

