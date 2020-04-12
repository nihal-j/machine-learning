import numpy as np
from sklearn.model_selection import train_test_split
from logistic_regression import Model

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
    data = train_test_validation_split(X, t, test_ratio=0.33)

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

    model = Model(learning_rate=1.5, regularization='l2', lamb=0.000001, random_init=False, maxIters=1000)
    costs = model.fit(X_train, t_train)
    preds = model.predict(X_test)
    test_accuracy = model.calculate_accuracy(preds, t_test)
    print('Testing accuracy is: ', test_accuracy)
    print('Fscore is: ', model.calculate_fscore(preds, t_test))
    preds = model.predict(X_val)
    val_accuracy = model.calculate_accuracy(preds, t_val)
    print('Validation accuracy is: ', val_accuracy)
    print('Fscore is: ', model.calculate_fscore(preds, t_val))

    # X_train = np.array([[1.2, 1.4, 1.1], [-2.3, -1.6, -1.5]])
    # t_train = np.array([[1, 0, 1]])
    # X_test = np.array([[1.3], [-1.8]])
    # t_test = np.array([[0]])

    # model = Model(maxIters=2)
    # model.fit(X_train, t_train)