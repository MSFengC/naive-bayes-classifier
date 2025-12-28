from naiveBayesClassifier.trainedData import TrainedData

class Trainer(object):

    """docstring for Trainer"""
    def __init__(self, tokenizer):
        super(Trainer, self).__init__()
        self.tokenizer = tokenizer
        self.trained_data = TrainedData()
        self.data = self.trained_data

    def train(self, title, text, className):
        """
        enhances trained data using the given text and class
        """
        self.data.increaseClass(className)

        tokens_text = self.tokenizer.tokenize(text)
        tokens_title = self.tokenizer.tokenize(title)

        for token_txt in tokens_text:
            token_txt = self.tokenizer.remove_stop_words(token_txt)
            token_txt = self.tokenizer.remove_punctuation(token_txt)
            self.data.increaseToken('text', token_txt, className)

        for token_title in tokens_title:
            token_title = self.tokenizer.remove_stop_words(token_title)
            token_title = self.tokenizer.remove_punctuation(token_title)
            self.data.increaseToken('title', token_title, className)
