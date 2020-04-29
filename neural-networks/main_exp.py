'''
    File for experimentation.
'''

from neural_network import Model
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

np.random.seed(42)

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
    
    # X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=test_ratio, random_state=0)
    # X_valid, X_test, t_valid, t_test = train_test_split(X_test, t_test, test_size=0.5, random_state=0)
    
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
    X = normalize(X, method='standardization')
    # X = normalize(X)
    data = train_test_validation_split(X, t, test_ratio=0.15)

    # stacking each sample as a column
    X_train = data['X_train'].T
    t_train = data['t_train'].reshape(1,-1)
    X_test = data['X_test'].T
    t_test = data['t_test'].reshape(1,-1)
    X_val = data['X_valid'].T
    t_val = data['t_valid'].reshape(1, -1)

    n_ = [10, 18, 1]
    activations_ = [None, 'relu', 'sigmoid']
    eta_ = 0.0001

    model = Model(n_, activations_, learning_rate_=eta_, max_iters_=5000)
    costs, preds = model.fit(X_train, t_train)
    print(preds)
    print(t_train)
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

    # plt.xscale(value='log')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    # plt.suptitle('Learning Rate=0.0001, Number of iterations=3000')
    # plt.plot(regs, trains)
    plt.plot(costs)
    plt.show()

    """
    np.random.seed(3)
    n_ = [10, 7, 6, 1]
    activations_ = [None, 'relu', 'relu', 'sigmoid']
    eta_ = 0.0001
    model = Model(n_, activations_, learning_rate_=eta_, max_iters_=3000)

    Exploding_gradients:
    n_ = [10, 25, 1]
    activations_ = [None, 'relu','sigmoid']
    eta_ = 0.001
    model = Model(n_, activations_, learning_rate_=eta_, max_iters_=10000)
    """