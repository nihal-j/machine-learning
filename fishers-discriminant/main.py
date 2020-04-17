from fishers_discriminant import Model
import numpy as np
import matplotlib.pyplot as plt
import plotter

def load_data(path):

    '''
        Load .npy data specified at `path`.
    '''
    
    data = np.load(path)
    return data

def train_test_split(X, test_ratio=0.33):

    '''
        Split rows into two sets. The test set is separated from its class values into a
        separate variable `t`.
    '''
    
    np.random.shuffle(X)
    tot = X.shape[0]
    test_count = int(test_ratio*tot)
    test = X[:test_count, :-1]
    t = X[:test_count, -1]
    train = X[test_count:, :]
    return train, test, t

def calculate_accuracy(y, t):
        
        return 100 - (np.mean(np.abs(y - t))*100)
    
def calculate_fscore(y, t):
    
    intersection = np.sum(y*t)
    precision = intersection / np.sum(y)
    recall = intersection / np.sum(t)
    fscore = (2*precision*recall)/(precision + recall)
    return fscore

if __name__ == '__main__':

    model = Model()
    data = load_data('data/data2D.npy')
    train, test, t = train_test_split(data, 0.20)
    _2D = 1
    # print(train.shape, test.shape)

    # Model takes as input examples stacked as rows
    X0, X1, tx, ty = model.discriminate(data)
    y = model.predict(test)
    print(calculate_accuracy(y, t))
    print(calculate_fscore(y, t))

    plotter.plot_normal(X0, '#8B0000')
    plotter.plot_normal(X1, '#00008B')
    plotter.plot_line(X0, '#8B0000')
    plotter.plot_line(X1, '#00008B')
    plotter.plot_point(tx, ty)
    plt.show()

    plotter.plot_transformed(X0, model.get_w(), '#8B0000')
    plotter.plot_transformed(X1, model.get_w(), '#00008B')
    if _2D:
        x0, x1 = model.segregate_classes(train)
        plotter.plot(x0*1000, '#F08080')
        plotter.plot(x1*1000, '#ADD8E6')
    plt.show()