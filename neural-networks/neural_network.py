import numpy as np

class Model:
    
    def __init__(self, n_, activations_, w_={}, b_={}, learning_rate_=0.0001, max_iters_=1000):
        
        '''
            A Model is a deep neural network whose configuration is contained in `n`. Its paramters are
            defined in `w` and biases in `b`. The activation functions for hidden layers can be specified as
            'relu' or 'sigmoid'. The output layer always uses 'sigmoid' activation.
            
            Arguments:
                n_:             neural network configuration, specifically `n_[i]` is the number of nodes in layer i
                w_:             weight parameters for each layer, specifically `w_[i]` is the weight matrix of layer i
                b_:             biases for each layer, specifically `b_[i]` is the bias vector for layer i
                activations_:   activations for each hidden layer. `activations[0]` SHOULD be None
                learning_rate_: learning rate for updation of weights
                max_iters_:     maximum number of iterations during training
        '''
        
        self.n = n_
        self.L = len(self.n) - 1
        self.activations = activations_
        self.w = w_
        self.b = b_
        self.learning_rate = learning_rate_
        self.max_iters = max_iters_
        self.initialize_parameters()
        
    
    def initialize_parameters(self):
    
        # the ith hidden layer should be of shape (n[i], n[i - 1])
        for i in range(1, self.L + 1):
            self.w[i] = np.random.randn(self.n[i], self.n[i - 1])/np.sqrt(self.n[i])
            # self.w[i] = np.random.randn(self.n[i], self.n[i - 1])*0.01
            # self.b[i] = np.random.randn(self.n[i], 1)
            self.b[i] = np.zeros((self.n[i], 1))
            
            
    def sigmoid(self, a):

        x = np.copy(a)
        x[x < -15] = -15
        x[x > 15] = 15
        return 1/(1 + np.exp(-x))
    

    def relu(self, a):

        x = np.copy(a)
        x[x < 0] = 0.0
        return x


    def tanh(self, a):

        x = 2*np.copy(a)
        x[x < -15] = -15
        x[x > 15] = 15
        return (np.exp(x) - 1)/(np.exp(x) + 1)


    def activation_function(self, a, layer):

        '''
            Computes the activation for ith layer. i >= 1.
            i = 0 corresponds to input layer and has no activation
        '''

        if self.activations[layer] == 'relu':
            return self.relu(a)

        if self.activations[layer] == 'sigmoid':
            return self.sigmoid(a)

        if self.activations[layer] == 'tanh':
            return self.tanh(a)
        

    def derivative(self, a, layer):

        if self.activations[layer] == 'relu':
            x = np.copy(a)
            x[x > 0] = 1.0
            x[x <= 0] = 0.0
            return x

        if self.activations[layer] == 'sigmoid':
            return self.sigmoid(a)*self.sigmoid(1 - a)

        if self.activations[layer] == 'tanh':
            return 1 - (self.tanh(a)**2)
        
    
    def forward_propagate(self, X):

        z = {}
        a = {}
        z[0] = X
        M = X.shape[1]
        for i in range(1, self.L + 1):
            a[i] = np.matmul(self.w[i], z[i - 1]) + self.b[i]
            z[i] = self.activation_function(a[i], i)
        return a, z
    
    
    def back_propagate(self, a, z, t):

        delta = {}
        dw = {}
        db = {}

        delta[self.L] = z[self.L] - t
        dw[self.L] = np.matmul(delta[self.L], z[self.L - 1].T)
        db[self.L] = np.sum(delta[self.L], axis=1, keepdims=True)

        for i in range(self.L - 1, 0, -1):

            delta[i] = self.derivative(a[i], i)*np.matmul(self.w[i + 1].T, delta[i + 1])
            dw[i] = np.matmul(delta[i], z[i - 1].T)
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
        preds[preds <= 0.5] = 0
        return preds

    def compute_cost(self, y, t):

        y = np.copy(y)
        y[y == 1] = 1 - 1e-20
        y[y == 0] = 1e-20
        return -np.sum(t*np.log(y) + (1 - t)*(np.log(1 - y)))

    
    def fit(self, X, t):
        
        print_costs = True
        print_step = self.max_iters/20
        learning_rate_updated = False
        costs = np.zeros(self.max_iters)
        for i in range(self.max_iters):
            
            a, z = self.forward_propagate(X)
            costs[i] = self.compute_cost(z[self.L], t)

            if print_costs and i % print_step == 0:
                print('Cost after iteration ', i, ': ', costs[i])

            # if i > 0 and (costs[i] - costs[i - 1])/costs[i - 1] < 0.01 and learning_rate_updated == False:
            if i > self.max_iters/2 and learning_rate_updated == False:
                self.learning_rate *= 0.1
                learning_rate_updated = True

            dw, db = self.back_propagate(a, z, t)
            self.update(dw, db)
                        
        preds = self.predict(X)
        
        return costs, preds