import csv
import numpy as np


def load_data(path):
    
    """
    
        Loads the CSV file located at `path` into a numpy array.
        
        Args:
            path:    str path to the CSV file to be loaded
        
        Returns:
            data:    numpy array with the data arranged in 2D
        
    """

    data = np.array([])
    with open(path) as file:
        reader = csv.reader(file, delimiter=',')
        rowCnt = 0
        for row in reader:
            rowCnt += 1
            for value in row:
                data = np.append(data, value)
    data = np.reshape(data, (rowCnt, -1))
    data = data.astype(np.float)
    
    return data



def store(path, data):
    
    """
    
        Stores `data` into persistent memory located at `path`.
        
        Args:
            path:    str path to destination
            data:    numpy array to be stored
    
    """
    
    np.save(path, data)



if __name__ == '__main__':
    
    data = load_data('data/a1_d1.csv')
    print(data.shape)
    store('data2D.npy', data)
    data = load_data('data/a1_d2.csv')
    print(data.shape)
    store('data3D.npy', data)