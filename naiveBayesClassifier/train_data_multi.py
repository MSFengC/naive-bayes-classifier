class TrainDataMulti:
    def __init__(self, alpha):
        self.count_classes = {}
        self.token_frequencies = {}
        self.tokens_classes = {}
        self.alpha = alpha

    def increase_class(self, class_name):
        self.count_classes[class_name] = self.count_classes.get(class_name, 0) + 1

    def increase_token(self, token, class_name):
        if token not in self.token_frequencies:
            self.token_frequencies[token] = {}

        self.token_frequencies[token][class_name] = self.token_frequencies[token].get(class_name, 0) + 1

    # \sum_w count(w, c)
    def increase_token_in_class(self, class_name):
        self.tokens_classes[class_name] = self.tokens_classes.get(class_name, 0) + 1

    def get_classes(self):
        return self.count_classes.keys()

    # K
    def get_count_classes(self):
        return len(self.count_classes.keys())

    def get_count_doc(self):
        return sum(self.count_classes.values())

    # V
    def get_v(self):
        return len(self.token_frequencies.keys())

    def get_n_c(self, class_name):
        return self.count_classes[class_name]

    def get_token_count_in_class(self, token, class_name):
        # if token in self.token_frequencies.keys() and class_name in self.token_frequencies[token].keys():
        #     return self.token_frequencies[token][class_name]
        # else:
        #     return 0.0
        return float(self.token_frequencies.get(token, {}).get(class_name, 0.0))

    def get_all_token_in_class(self, class_name):
        return self.tokens_classes.get(class_name, 0.0)
