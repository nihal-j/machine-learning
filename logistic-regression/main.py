import numpy as np
from logistic_regression import Model
import matplotlib.pyplot as plt
np.random.seed(0)

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
        numerator = data - np.min(data, axis=1).reshape(-1, 1)
        denominator = (np.max(data, axis=1) - np.min(data, axis=1)).reshape(-1, 1)
        return numerator/denominator
    
    if (method == 'standardization'):
        numerator = data - np.mean(data, axis=1).reshape(-1, 1)
        denominator = np.std(data, axis=1).reshape(-1, 1)
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
    # X = normalize(X, method='standardization')
    data = train_test_validation_split(X, t, test_ratio=0.15)

    # stacking each sample as a column
    X_train = data['X_train'].T
    X_train = normalize(X_train, method='standardization')
    t_train = data['t_train'].reshape(1,-1)
    X_test = data['X_test'].T
    X_test = normalize(X_test, method='standardization')
    t_test = data['t_test'].reshape(1,-1)
    X_val = data['X_valid'].T
    X_val = normalize(X_val, method='standardization')
    t_val = data['t_valid'].reshape(1, -1)
        
    # from here each sample is a column
    model = Model(learning_rate=0.01, regularization='l2', lamb=1, random_init='gaussian', maxIters=1000)
    preds, costs = model.fit(X_train, t_train)
    print('Train size: ', X_train.shape[1])
    print('Training accuracy is: ', calculate_accuracy(preds, t_train))
    print('Training fscore is: ', calculate_fscore(preds, t_train))
    print()

    preds = model.predict(X_val)
    print('Validation size: ', X_val.shape[1])
    print('Validation accuracy is: ', calculate_accuracy(preds, t_val))
    print('Validation fscore is: ', calculate_fscore(preds, t_val))
    print()

    preds = model.predict(X_test)
    print('Test size: ', X_test.shape[1])
    print('Testing accuracy is: ', calculate_accuracy(preds, t_test))
    print('Testing fscore is: ', calculate_fscore(preds, t_test))
    print()

    print(model.w)
    print(model.b)
    # plt.xscale(value='log')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    # plt.plot(regs, trains)
    plt.plot(costs)
    plt.show()