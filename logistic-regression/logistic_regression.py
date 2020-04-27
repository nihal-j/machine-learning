import numpy as np

np.random.seed(0)

class Model:

    '''
        Parameter vector `w` is a column vector of size (M, 1) where M is the number of features.

        Any data matrix `X` is a matrix of size (M, N) where M is the number of features and N is the
        number of samples, i.e, each sample is stacked into `X` as a column.

        These shapes are maintained throughout the class.
    '''
    
    def __init__(self, learning_rate=0.1, lamb=0, regularization='l2', random_init=True, maxIters=10000):

        '''
            Arguments:
                learning_rate:
                lamb:               Regularization parameter, lambda
                regularization:     'l1' or 'l2' or None
                random_init:        True if random (Gaussian) initialization of parameters is desired
                maxIters:
        '''
        
        self.learning_rate = learning_rate
        self.lamb = lamb
        self.regularization = regularization
        self.random_init = random_init
        self.maxIters = maxIters
        self.w = None
        self.b = None
        
    
    def initalize_parameters(self, dims):

        '''
            Initializes parameters `w` and `b` depending on the choice given by `random_init`.
            Shape of `w` is (1, dims) => it is a row vector.
            `b` is initialized as 0 always.
        '''
        
        if self.random_init:
            self.w = np.random.randn(dims).reshape(1, -1)
        else:
            self.w = np.zeros((1, dims))
        self.b = 0
    
    def sigmoid(self, x):

        '''
            Calculate the logisitic sigmoid of `x`.
            Note: Too large or too low values are trimmed to ensure that overflows or divisions by zero do not occur.
        '''
        
        M = x.shape[1]
        x[x > 15] = 15
        x[x < -15] = -15
            
        return (1 / (1 + np.exp(-x)))
    
    def predict(self, X):

        '''
            Convert probability values in `X` to predictions belonging to {0, 1} classes using a threshold of 0.5.
        '''
        
        M = X.shape[1]
        preds = np.zeros((1, M))
        
        probabilities = self.sigmoid(np.dot(self.w, X) + self.b)
        
        for i in range(M):
            preds[0][i] = 0 if probabilities[0][i] <= 0.5 else 1
            
        return preds
        
    
    def fit(self, X_train, t_train):

        '''
            Use gradient descent to optimize parameters for logistic regression.
        '''
        
        # number of features
        N = X_train.shape[0]
        # number of samples
        M = X_train.shape[1]
        
        self.initalize_parameters(N)
        costs = np.array([])
        
        for iteration in range(self.maxIters):
            
            # forward calculation
            y = self.sigmoid(np.dot(self.w, X_train) + self.b)
            
            # cost calculation
            if self.regularization == 'l2':
                reg_term = self.lamb*np.dot(self.w, self.w.T)/2
            elif self.regularization == 'l1':
                reg_term = self.lamb*np.sum(np.abs(self.w))/2
            elif self.regularization == None:
                reg_term = 0
                
            cost = -np.sum((t_train*np.log(y)) + ((1 - t_train)*np.log((1 - y)))) + reg_term
            costs = np.append(costs, cost)
            
            # gradient calculation
            if self.regularization == 'l2':
                reg_term = self.lamb*self.w
            elif self.regularization == 'l1':
                tmp = np.copy(self.w)
                tmp[tmp == 0] = 1e-5
                reg_term = self.lamb*(tmp/np.abs(tmp))
                
            dw = np.dot((y - t_train), X_train.T) + reg_term
            db = np.sum(y - t_train)
            
            # backward update
            self.w = self.w - self.learning_rate*dw
            self.b = self.b - self.learning_rate*db
            
        predictions = self.predict(X_train)
        return predictions, costs