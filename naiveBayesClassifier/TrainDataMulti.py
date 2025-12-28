class TrainDataMulti:
    def __init__(self):
        self.count_classes = {}
        self.token_frequencies = {}
        self.tokens_classes = {}

    def increaseClass(self, className):
        self.count_classes[className] = self.count_classes.get(className, 0) + 1

    def increaseToken(self, token, className):
        if token not in self.token_frequencies:
            self.token_frequencies[token] = {}

        self.token_frequencies[token][className] = self.token_frequencies[token].get(className, 0) + 1

    def increaseTokenInClass(self, className):
        self.tokens_classes[className] = self.tokens_classes.get(className, 0) + 1

    def getClasses(self):
        return self.tokens_classes.keys()

    def getCountClasses(self):
        return len(self.count_classes.keys())