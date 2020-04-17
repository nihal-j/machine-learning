import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_normal(data, c):

    mu, std = norm.fit(data)
    plt.hist(data, bins=25, density=True, alpha=0.6, color=c)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)

def plot_point(x, y):

    plt.scatter(x, y, s=200, color='cyan')

def plot_line(data, c):

    cnt = len(data)
    plt.scatter(data, np.zeros(cnt)-0.004, color=c)

def plot_transformed(data, w, c):

    plt.scatter(data*w[0], data*w[1], color=c, alpha=0.5)

def plot(data, c):

    # data stacked as columns
    plt.scatter(data[0,:], data[1,:], color=c, alpha=0.3)