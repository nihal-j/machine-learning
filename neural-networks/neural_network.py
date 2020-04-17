import numpy as np
np.set_printoptions(precision=4, linewidth=100, suppress=True)

class Model:
    
    def __init__(self, L_, n_, w_={}, b_={}, activation_='relu', learning_rate_=0.0001, max_iters_=1000):
        
        '''
            A Model is a deep neural network whose configuration is contained in `n`. Its paramters are
            defined in `w` and biases in `b`. The activation functions for hidden layers can be specified as
            'relu' or 'sigmoid'. The output layer always uses 'sigmoid' activation.
            
            Arguments:
                L_:             number of layers in neural network (excluding input layer)
                n_:             neural network configuration, specifically `n_[i]` is the number of nodes in layer i
                w_:             weight parameters for each layer, specifically `w_[i]` is the weight matrix of layer i
                b_:             biases for each layer, specifically `b_[i]` is the bias vector for layer i
                activation_:    activation for each hidden layer
                learning_rate_: learning rate for updation of weights
                max_iters_:     maximum number of iterations during training
        '''
        
        self.L = L_
        self.n = n_
        self.w = w_
        self.b = b_
        self.activation = activation_
        self.learning_rate = learning_rate_
        self.max_iters = max_iters_

        self.initialize_parameters()
        
    
    def initialize_parameters(self):
    
        # the ith hidden layer should be of shape (n[i - 1], n[i])
        for i in range(1, self.L + 1):
            self.w[i] = np.random.randn(self.n[i - 1], self.n[i])
            self.b[i] = np.random.randn(self.n[i], 1)
            
            
    def sigmoid(self, a):

        x = np.copy(a)
        x[x < -15] = -15
        x[x > 15] = 15

        return 1/(1 + np.exp(-x))
    

    def relu(self, a):

        x = np.copy(a)
        x[x < 0] = 0

        return x


    def activation_function(self, a):

        if self.activation == 'relu':
            return self.relu(a)

        if self.activation == 'sigmoid':
            return self.sigmoid(a)
        

    def derivative(self, a):

        if self.activation == 'relu':
            x = np.copy(a)
            x[x > 0] = 1
            x[x < 0] = 0
            return x

        if self.activation == 'sigmoid':
            return self.sigmoid(a)*self.sigmoid(1 - a)
        
    
    def forward_propagate(self, X):

        z = {}
        a = {}
        z[0] = X
        M = X.shape[1]
        for i in range(1, self.L):
            a[i] = np.matmul(self.w[i].T, z[i - 1]).reshape(self.n[i], M) + self.b[i]
            z[i] = self.activation_function(a[i])
        a[self.L] = np.matmul(self.w[self.L].T, z[self.L - 1]).reshape(self.n[self.L], M) + self.b[self.L]
        z[self.L] = self.sigmoid(a[self.L])

        return a, z
    
    
    def back_propagate(self, a, z, t):

        delta = {}
        dw = {}
        db = {}

        delta[self.L] = z[self.L] - t
        dw[self.L] = np.matmul(z[self.L - 1], delta[self.L].T)
        db[self.L] = np.sum(delta[self.L], axis=1, keepdims=True)

        for i in range(self.L - 1, 0, -1):

            delta[i] = self.derivative(a[i])*np.matmul(self.w[i + 1], delta[i + 1])
            dw[i] = np.matmul(z[i - 1], delta[i].T)
            db[i] = np.sum(delta[i], axis=1, keepdims=True)

        return dw, db
    
    
    def update(self, dw, db):

        for i in range(1, self.L + 1):
            
            self.w[i] = self.w[i] - self.learning_rate*dw[i]
            self.b[i] = self.b[i] - self.learning_rate*db[i]
            
    
    def predict(self, X):
    
        a, z = self.forward_propagate(X)
        preds = np.copy(z[self.L])
        preds[preds > 0.5] = 1
        preds[preds < 0.5] = 0

        return preds


    def calculate_accuracy(self, y, t):
    
        return 100 - (np.mean(np.abs(y - t))*100)

    
    def fit(self, X, t):
        
        for i in range(self.max_iters):
            
            a, z = self.forward_propagate(X)
            dw, db = self.back_propagate(a, z, t)
            self.update(dw, db)
                        
        preds = self.predict(X)
        training_accuracy = self.calculate_accuracy(preds, t)
        
        return training_accuracy