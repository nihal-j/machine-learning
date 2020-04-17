# Data information

# 1. variance of Wavelet Transformed image (continuous)
# 2. skewness of Wavelet Transformed image (continuous)
# 3. curtosis of Wavelet Transformed image (continuous)
# 4. entropy of image (continuous)
# 5. class (integer)

# Number of instances: 1372

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
    
    data = load_data('data/data_banknote_authentication.csv')
    print(data.shape)
    store('data.npy', data)