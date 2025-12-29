import math
import operator
from collections import Counter


class ClassifierMulti(object):
    def __init__(self, train_data, tokenizer):
        super(ClassifierMulti, self).__init__()
        self.data = train_data
        self.tokenizer = tokenizer

    def classify(self, content_list):
        result = []
        rank = []

        classes = self.data.get_classes()
        for content in content_list:
            posterior_prob = {}
            tokens = list(self.tokenizer.tokenize(content['title'] + ' ' + content['text']))
            cnt = Counter(t for t in tokens if t)

            for class_name in classes:
                score = math.log(self.get_prior(class_name))
                for w, n in cnt.items():
                    p = self.get_token_prob(w, class_name)
                    score += n*math.log(p)
                # token_prob = [self.get_token_prob(token, class_name) for token in tokens]
                # token_set_prob = sum(math.log(p) for p in token_prob if p and p > 0)

                posterior_prob[class_name] = score
            rank = sorted(posterior_prob.items(),
                          key=operator.itemgetter(1),
                          reverse=True)
            result.append({"true": content["category"], "rank": rank})

        return result

    def get_prior(self, class_name):
        n = self.data.get_count_doc()
        n_c = self.data.get_n_c(class_name)
        k = self.data.get_count_classes()

        return (n_c + self.data.alpha) / (n + (k * self.data.alpha))

    def get_token_prob(self, token, class_name):
        token_count_in_class = self.data.get_token_count_in_class(token, class_name) or 0.0
        all_token_in_class = self.data.get_all_token_in_class(class_name)
        v = self.data.get_v()

        return (token_count_in_class + self.data.alpha) / (all_token_in_class + (self.data.alpha * v))
