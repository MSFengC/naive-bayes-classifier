from __future__ import division

import math
import operator
from functools import reduce

from naiveBayesClassifier.ExceptionNotSeen import NotSeen


class Classifier(object):
    """docstring for Classifier"""
    def __init__(self, trainedData, tokenizer):
        super(Classifier, self).__init__()
        self.data = trainedData
        self.tokenizer = tokenizer
        self.defaultProb = 0.000000001

    # ali ata bak
    def classify(self, args):
        
        documentCount = self.data.getDocCount()
        classes = self.data.getClasses()

        # only unique tokens
        if not args['title'] or args['title'].strip() != "":
            tokens_title = list(set(self.tokenizer.tokenize(args['title'])))
        if not args['text'] or args['text'].strip() != "":
            tokens_text = list(set(self.tokenizer.tokenize(args['text'])))
        
        probsOfClasses = {}

        for className in classes:
            
            # we are calculating the probablity of seeing each token 
            # in the text of this class
            # P(Token_1|Class_i)
            tokens_title_probs = [self.getTokenProb('title', token, className) for token in tokens_title]
            tokens_text_probs = [self.getTokenProb('text', token, className) for token in tokens_text]
            
            # calculating the probability of seeing the set of tokens
            # in the text of this class
            # P(Token_1|Class_i) * P(Token_2|Class_i) * ... * P(Token_n|Class_i)
            try:
                # tokenSet_title_prob = reduce(lambda a,b: a*b, (i for i in tokens_title_probs if i))
                tokenSet_title_prob = sum(math.log(p) for p in tokens_title_probs if p and p > 0)
            except:
                tokenSet_title_prob = 0
            try:
                # tokenSet_text_prob = reduce(lambda a, b: a * b, (i for i in tokens_text_probs if i))
                tokenSet_text_prob = sum(math.log(p) for p in tokens_text_probs if p and p > 0)

            except:
                tokenSet_text_prob = 0
            
            # probsOfClasses[className] = tokenSet_title_prob * tokenSet_text_prob * self.getPrior(className)
            probsOfClasses[className] = tokenSet_title_prob + tokenSet_text_prob + math.log(self.getPrior(className))
        
        return sorted(probsOfClasses.items(), 
            key=operator.itemgetter(1), 
            reverse=True)


    def getPrior(self, className):
        theta = 1.0
        classes = self.data.getClasses()

        return self.data.getClassDocCount(className) /  (self.data.getDocCount() + (len(classes)*theta))

    def getTokenProb(self, source, token, className):
        theta = 1.0
        #p(token|Class_i)
        classDocumentCount = self.data.getClassDocCount(className)

        # if the token is not seen in the training set, so not indexed,
        # then we return None not to include it into calculations.
        try:
            tokenFrequency = self.data.getFrequency(source, token, className)
        except NotSeen as e:
            return None

        # this means the token is not seen in this class but others.
        if tokenFrequency is None:
            # return self.defaultProb
            tokenFrequency = 0.0

        probablity =  (tokenFrequency + theta) / (classDocumentCount + (self.data.getParameterCount(source)*theta))
        return probablity
