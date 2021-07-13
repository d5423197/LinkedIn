import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
class k_means:
    def __init__(self, k):
        self.k = k 
        
    def init_centroids(self, X):
        return X[np.random.choice(len(X), self.k)]
        
    def plot(self, X, centers):
        'we will use this function to keep updating the graph'
        plt.scatter(X[:,0], X[:,1], c = y)
        plt.scatter(centers[:,0], centers[:, 1], c = 'r')
        plt.annotate('centers', xy = (centers[0, 0], centers[0, 1]), 
                     arrowprops = dict(facecolor='black', shrink=0.05), xytext = (centers[0, 0] + 2, centers[0, 1] + 4))
        plt.show()
        
    def cluster(self, X, centers):
        return np.argmin(np.sqrt(((X[:, np.newaxis] - centers)**2).sum(axis = 2)), axis = 1)
        
    def recenter(self, X, centers, label):
        result = []
        'label starts from 0 to k-1'
        for i in range(self.k):
            'points belong to this cluster'
            points = X[label == i]
            new_center = points.mean(axis = 0)
            result.append(new_center)
        return np.stack(result, axis = 0)
        
    def train(self, X, max_iteration):
        centers = self.init_centroids(X)
        self.plot(X, centers)
        for i in range(max_iteration):
            label = self.cluster(X, centers)
            centers = self.recenter(X, centers, label)
        self.plot(X, centers)
        return centers

c = [[0, 0], [10, 10], [5, 5]]
X, y = make_blobs(n_samples = 50, centers = c, n_features = 2, random_state = 0)
k_means(3).train(X, 10000)

        