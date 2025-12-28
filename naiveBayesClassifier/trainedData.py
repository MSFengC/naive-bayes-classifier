import sys
from naiveBayesClassifier.ExceptionNotSeen import NotSeen


class TrainedData(object):
    def __init__(self):
        self.docCountOfClasses = {}
        self.frequencies = {'title':{},'text':{}}

    def increaseClass(self, className, byAmount = 1):
        self.docCountOfClasses[className] = self.docCountOfClasses.get(className, 0) + 1

    def increaseToken(self, source, token, className, byAmount = 1):
        if not token in self.frequencies[source]:
                self.frequencies[source][token] = {}

        self.frequencies[source][token][className] = self.frequencies[source][token].get(className, 0) + 1


    def decreaseToken(self, token, className, byAmount=1):
        if token not in self.frequencies:
            raise NotSeen(token)
        foundToken = self.frequencies[token]
        if className not in self.frequencies:
            sys.stderr.write("Warning: token %s has no entry for class %s. Not decreasing.\n" % (token, className))
            return
        if foundToken[className] < byAmount:
            raise ArithmeticError("Could not decrease %s/%s count (%i) by %i, "
                                  "as that would result in a negative number." % (
                                      token, className, foundToken[className], byAmount))
        foundToken[className] -= byAmount

    def getDocCount(self):
        """
        returns all documents count
        """
        return sum(self.docCountOfClasses.values())

    def getClasses(self):
        """
        returns the names of the available classes as list
        """
        return self.docCountOfClasses.keys()

    def getClassDocCount(self, className):
        """
        returns document count of the class. 
        If class is not available, it returns None
        """
        return self.docCountOfClasses.get(className, None)

    def getParameterCount(self, source):

        return len(self.frequencies[source].keys())

    def getFrequency(self, source, token, className):
        if token in self.frequencies[source]:
            foundToken = self.frequencies[source][token]
            return foundToken.get(className)
        else:
            # raise NotSeen(token)
            return None
