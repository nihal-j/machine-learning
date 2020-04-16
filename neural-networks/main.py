from neural_network import Model
from sklearn.model_selection import train_test_split
import numpy as np

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

def train_test_validation_split(X, t, test_ratio=0.33):

    '''
        Make use of sklearn's `train_test_split` to split `X` into train, test and validation sets.
    '''
    
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=test_ratio, random_state=42)
    X_valid, X_test, t_valid, t_test = train_test_split(X_test, t_test, test_size=0.5, random_state=42)
    
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


if __name__ == '__main__':

    path = 'data/data.npy'
    data = load_data(path)

    # from here each sample is a row
    X, t = segregate_target(data)
    data = train_test_validation_split(X, t, test_ratio=0.20)

    data['X_train'] = normalize(data['X_train'])
    data['X_valid'] = normalize(data['X_valid'])
    data['X_test'] = normalize(data['X_test'])

    # from here each sample is a column
    X_train = data['X_train'].T
    t_train = data['t_train'].reshape(1,-1)
    X_test = data['X_test'].T
    t_test = data['t_test'].reshape(1,-1)
    X_val = data['X_valid'].T
    t_val = data['t_valid'].reshape(1, -1)

    L_ = 3
    n_ = [10, 20, 12, 1]
    activation_ = 'relu'
    eta_ = 0.0001
    model = Model(L_, n_, activation_='relu', learning_rate_=eta_, max_iters_=5000)

    print(model.fit(X_train, t_train))