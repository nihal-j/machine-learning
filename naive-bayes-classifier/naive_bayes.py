import numpy as np
import string

'''
For bag of words, changes need to be made in 4 places
(noted in comments)
'''

class Model:
    '''
        Model for sentiment analysis using Naive Bayes classification.
    '''
    def __init__(self):

        self.params = {}
        self.feature_count = 0
        self.countY = 0
        self.countN = 0
        self.dictionary = {}



    def preprocess_sample(self, s):
        '''
            `s` is a string only. Returns preprocessed list of strings.
        '''
        s = s.casefold()
        s = s.translate(str.maketrans('', '', string.punctuation))
        s = s.split()
        # for d in delimiters:
        #    string = string.replace(d, '')
        return s



    def construct_dictionary(self, strings):
        '''
            `strings[i]` is the preprocessed list of strings for ith sample.
        '''
        idx = 0
        for s in strings:
            for w in s:
                if w not in self.dictionary:
                    self.dictionary[w] = idx
                    idx += 1
        # setting a feature for unknown words
        # self.dictionary['UNK'] = idx



    def vectorize(self, s):
        '''
            Convert string `s` to a feature vector represented by terms of dictionary.
        '''
        vector = np.zeros(len(self.dictionary))
        for w in s:
            if w not in self.dictionary:
                # vector[self.dictionary['UNK']] = 1
                pass
            else:
                # use the next one line for bag of words
                # vector[self.dictionary[w]] += 1
                vector[self.dictionary[w]] = 1
        return vector



    def learn_parameters(self, vectors, labels):
        '''
            Estimate the likelihood parameters from `vectors`. `vectors[i]` is the vector
            representation of ith sample.
        '''
        # defining the parameters of the model
        # likelihoods will be P(Wi|C) for C=0 and C=1
        feature_count = len(self.dictionary)
        likelihoods = np.zeros((2, feature_count))

        # count of number of samples from each class
        countY = np.sum(labels)
        countN = len(labels) - countY

        denomY = countY
        denomN = countN
        # use the next two lines for bag of words
        # denomY = feature_count
        # denomN = feature_count

        for i, vector in enumerate(vectors):
            # adding to frequencies of each word in the corresponding class
            likelihoods[labels[i]] = likelihoods[labels[i]] + vector
            # use the next lines for bag of words
            # if labels[i]:
            #     denomY += np.sum(vector)
            # else:
            #     denomN += np.sum(vector)

        # prior probabilites of each class
        probY = countY/len(labels)
        probN = countN/len(labels)
        # converting frequencies to probabilities
        # remove the 2 for bag of words
        likelihoods[0] = (likelihoods[0] + 1)/(denomN + 2)
        likelihoods[1] = (likelihoods[1] + 1)/(denomY + 2)

        self.params = {
            'likelihoods': likelihoods,
            'probY': probY,
            'probN': probN
        }

        

    def predict(self, s):
        '''
            Make predictions for `s` using estimates of posterior probabilities
            from calculated likelihoods.
        '''
        s = self.preprocess_sample(s)
        vector = self.vectorize(s)
        
        likelihoods = self.params['likelihoods']
        probY = self.params['probY']
        probN = self.params['probN']

        estY = probY
        estN = probN
        for i in range(len(vector)):
            if vector[i]:
                estY *= likelihoods[1][i]
                estN *= likelihoods[0][i]
        
        if estY >= estN:
            return 1
        return 0


    def fit(self, data):
        '''
            `data` is a list of tuples. `data[i]` = (string[i], label[i]).
        '''
        samples = []
        labels = []
        for i in range(len(data)):
            samples.append(self.preprocess_sample(data[i][0]))
            labels.append(data[i][1])

        # from this point samples[i] is a preprocessed list for ith sample
        # and labels[i] is its correct label

        # constructing dictionary
        self.construct_dictionary(samples)

        # vectorizing every sample
        vectors = np.array([None for i in range(len(samples))])
        for i, sample in enumerate(samples):
            vectors[i] = self.vectorize(sample)

        self.learn_parameters(vectors, labels)