import numpy as np
import pickle
from naive_bayes import Model
import random
np.random.seed(1)
random.seed(1)

def calculate_accuracy(preds, t):

    return 100 - (np.mean(np.abs(preds - t))*100)
    

def calculate_fscore(preds, t):
    
    epsilon = 1e-20

    tp = np.sum(t*preds) + epsilon
    pp = np.sum(preds) + epsilon
    ap = np.sum(t) + epsilon

    precision = tp/pp
    recall = tp/ap
    
    fscore = (2*precision*recall)/(precision + recall + epsilon)
    return fscore


if __name__ == '__main__':

    # Driver function
    data_file = open('data/data.pickle', 'rb')
    data = pickle.load(data_file)
    random.shuffle(data)

    k = 5
    test_size = len(data)//k
    validation_scores = np.zeros(k)
    f_scores = np.zeros(k)

    for i in range(k):

        train = data[:i*test_size]
        test = data[i*test_size:(i+1)*test_size]
        train.extend(data[(i+1)*test_size:])
        model = Model()
        model.fit(train)
        preds = []
        t = []
        for sample, label in test:
            t.append(label)
            preds.append(model.predict(sample))
        acc = calculate_accuracy(np.array(preds), np.array(t))
        f = calculate_fscore(np.array(preds), np.array(t))
        validation_scores[i] = acc
        f_scores[i] = f
        print('On validation set', i, ': ')
        print('Accuracy: ', acc)
        print('Fscore: ', f)
        print()

    print('Validation score: ', np.mean(validation_scores))


    # preds = []
    # t = []
    # model = Model()
    # model.fit(data)
    # for sample, label in data:
    #         t.append(label)
    #         preds.append(model.predict(sample))
    # print(calculate_accuracy(np.array(preds), np.array(t)))