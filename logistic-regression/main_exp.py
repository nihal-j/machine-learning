import numpy as np
from logistic_regression import Model
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

np.random.seed(0)
np.set_printoptions(precision=4)

def normalize(data, method='min-max'):
    
    '''
        Normalize `data` using `method`. Computes statistics assuming each sample is a row.

        Arguments:
            data:       data to be normalized
            method:     if 'min-max' then normalization is used, else standardization is used

        Returns:
            Normalized data   
    '''
    
    if (method == 'min-max'):
        numerator = data - np.min(data, axis=0)
        denominator = np.max(data, axis=0) - np.min(data, axis=0)
        return numerator/denominator
    
    if (method == 'standardization'):
        numerator = data - np.mean(data, axis=0)
        denominator = np.std(data, axis=0)
        return numerator/denominator
    

def segregate_target(data):

    '''
        Segregates `data` into (X, t) tuple where `X` has each example as a column and `t`
        is the corresponding class label.
    '''
    
    X = data[:, :-1]
    t = data[:, -1:]
    
    return X, t


def train_test_validation_split(X, t, test_ratio=0.15):

    '''
        Make use of sklearn's `train_test_split` to split `X` into train, test and validation sets.
        'X' has samples stacked as rows.
        Size of test set formed = Size of validation set formed = numberOfSamples * `test_ratio`.
    '''

    m = X.shape[0]
    test_size = (int)(m*test_ratio)
    val_size = test_size
    train_size = m - (test_size + val_size)
    
    # X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=test_ratio, random_state=42)
    # X_valid, X_test, t_valid, t_test = train_test_split(X_test, t_test, test_size=0.5, random_state=42)

    perm = np.random.permutation(m)
    X = X[perm, :]
    t = t[perm, :]
    X_train = X[:train_size, :]
    t_train = t[:train_size, :]
    X_valid = X[train_size:train_size+val_size, :]
    t_valid = t[train_size:train_size+val_size, :]
    X_test = X[train_size+val_size:, :]
    t_test = t[train_size+val_size:, :]
    
    data = {
        'X_train': X_train,
        't_train': t_train,
        'X_valid': X_valid,
        't_valid': t_valid,
        'X_test': X_test,
        't_test': t_test
    }
    
    return data


def load_data(path):

    '''
        Load .npy data specified at `path`.
    '''
    
    data = np.load(path)
    return data


def calculate_accuracy(y, t):

    return 100 - (np.mean(np.abs(y - t))*100)
    

def calculate_fscore(y, t):
    
    epsilon = 1e-20

    tp = np.sum(t*y) + epsilon
    pp = np.sum(y) + epsilon
    ap = np.sum(t) + epsilon

    precision = tp/pp
    recall = tp/ap
    
    fscore = (2*precision*recall)/(precision + recall + epsilon)
    return fscore


if __name__ == '__main__':

    path = 'data/data.npy'
    data = load_data(path)

    # from here each sample is a row
    X, t = segregate_target(data)
    X = normalize(X)
    data = train_test_validation_split(X, t, test_ratio=0.15)

    # stacking each sample as a column
    X_train = data['X_train'].T
    t_train = data['t_train'].reshape(1,-1)
    X_test = data['X_test'].T
    t_test = data['t_test'].reshape(1,-1)
    X_val = data['X_valid'].T
    t_val = data['t_valid'].reshape(1, -1)
        
    # regs = np.array([])
    inits = [True, False]
    vals = np.array([])
    trains = np.array([])
    # for i in range(11):
    #     lambs = np.append(lambs, 1/10**i)

    # from here each sample is a column
    for init in inits:

        model = Model(learning_rate=0.01, regularization='l2', lamb=0.01, random_init=init, maxIters=10000)
        preds, costs = model.fit(X_train, t_train)
        print('Initialization: ', init)
        print('Train size: ', X_train.shape[1])
        print('Training accuracy is: ', calculate_accuracy(preds, t_train))
        print('Training fscore is: ', calculate_fscore(preds, t_train))
        print()
        trains = np.append(trains, calculate_accuracy(preds, t_train))

        preds = model.predict(X_val)
        print('Validation size: ', X_val.shape[1])
        print('Validation accuracy is: ', calculate_accuracy(preds, t_val))
        print('Validation fscore is: ', calculate_fscore(preds, t_val))
        print(model.w)
        print('==============================================================')
        vals = np.append(vals, calculate_accuracy(preds, t_val))

    # preds = model.predict(X_test)
    # print('Test size: ', X_test.shape[1])
    # print('Testing accuracy is: ', calculate_accuracy(preds, t_test))
    # print('Testing fscore is: ', calculate_fscore(preds, t_test))
    # print()

    # plt.xscale(value='log')
    # plt.xlabel('Regularization')
    # plt.ylabel('Accuracy')
    # plt.suptitle('Regularization = L2')
    # plt.plot(regs, trains)
    # plt.plot(regs, vals)
    # plt.show()

'''
Experiments
-------

1.

np.random.seed(0)
data = train_test_validation_split(X, t, test_ratio=0.15)
model = Model(learning_rate=0.01, regularization='l2', lamb=0.01, random_init=True, maxIters=10000)

Train size:  962
Training accuracy is:  98.64864864864865
Fscore is  0.9843937575030012

Test size:  205
Testing accuracy is:  100.0
Testing fscore is:  1.0

Train size:  205
Validation accuracy is:  100.0
Validation fscore is:  1.0

[[-30.43507442 -30.42814545 -32.584463     0.32593369]]

=============================================================================================================

2.
Best learning rate = 0.01 (highest training accuracy + validation accuracy)
Train size:  962
Training accuracy is:  98.96049896049897
Training fscore is:  0.9878934624697336

Validation size:  205
Validation accuracy is:  100.0
Validation fscore is:  1.0

=============================================================================================================

3.
Lambda:  0.01 (L2)
Lambda:  0.001 (L1)
=============================================================================================================

4.
model = Model(learning_rate=0.1, regularization=reg, lamb=0.001, random_init=True, maxIters=10000)
Lambda:  l1
Train size:  962
Training accuracy is:  98.96049896049897
Training fscore is:  0.9878934624697336

Validation size:  205
Validation accuracy is:  100.0
Validation fscore is:  1.0
[[-80.5903 -82.9257 -89.9327  -4.9593]]
----------------------------------------
Lambda:  l2
Train size:  962
Training accuracy is:  98.64864864864865
Training fscore is:  0.9843937575030012

Validation size:  205
Validation accuracy is:  100.0
Validation fscore is:  1.0
[[-58.7993 -60.8648 -65.4458  -2.1618]]

=============================================================================================================

5.
Initializations: No observable differences in accuracies
=============================================================================================================

6.
No regularization:
model = Model(learning_rate=0.001, regularization=None, lamb=0.01, random_init=True, maxIters=20000)
Train size:  962
Training accuracy is:  97.81704781704782
Training fscore is:  0.9750297265160524

Validation size:  205
Validation accuracy is:  100.0
Validation fscore is:  1.0

Test size:  205
Testing accuracy is:  100.0
Testing fscore is:  1.0

[[-25.7920536  -24.11829519 -26.37576435   1.36289472]]

'''