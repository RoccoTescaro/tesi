__package__ = 'tools'

import numpy as np
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import cv2

def kDistances(samples, k, nJobs=1, f_dist = np.linalg.norm):

    if f_dist == np.linalg.norm:    

        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=nJobs).fit(samples)
        
        distances, _ = nbrs.kneighbors(samples, n_neighbors=k+1, return_distance=True)
        
        return distances[:, -1]
    
    def kWorker(i):
        distances = []
        for j in range(len(samples)):
            distances.append(f_dist(samples[i]-samples[j]))
        distances.sort()
        return distances[k]
    
    kth_distances = Parallel(n_jobs=nJobs)(delayed(kWorker)(i) for i in range(samples.shape[0]))

    return kth_distances

    #nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(samples)
    #
    #def kWorker(i):
    #    distances, _ = nbrs.kneighbors(samples[i].reshape(1, -1), n_neighbors=k+1)
    #    return distances[0][-1]
    #
    #kth_distances = Parallel(n_jobs=nJobs)(delayed(kWorker)(i) for i in range(samples.shape[0]))
    #
    #return kth_distances

    #Unparallelized version with complexity O(n^2)

    #dist = []

    #for i in range(len(samples)):
    #    sample_dist = []

    #    for j in range(len(samples)):
    #        sample_dist.append(f_dist(samples[i]-samples[j]))

    #    sample_dist.sort()
    #    dist.append(sample_dist[k])

    #return dist
    
def realismScore(PSamples, QSample, hyperparam, PkDistances = [], nJobs = 1, f_dist = np.linalg.norm):

    if len(PkDistances) == 0:
        PkDistances = kDistances(PSamples, hyperparam['k'], nJobs, f_dist)

    max_dist = 0
   
    for i in range(len(PSamples)):

        nom = f_dist(PSamples[i]-PkDistances[i])
        den = f_dist(PSamples[i]-QSample)

        value = nom/den

        if value > max_dist:
            max_dist = value

    return max_dist

def getRThreshold(samples, hyperparam, kDistances_ = [], nJobs=1, f_dist = np.linalg.norm):

    if len(kDistances_) == 0:
        kDistances_ = kDistances(samples, hyperparam['k'], nJobs, f_dist)

    return np.mean(kDistances_)*hyperparam['a']

def uniformData(sampleDim, numSamples, a = 0, b = 1):
    data = np.random.uniform(a, b, (numSamples, sampleDim))
    return np.array([tuple(d) for d in data])

def normalData(sampleDim, numSamples, shift = 0): 
    data = np.random.multivariate_normal(shift*np.ones(sampleDim), np.eye(sampleDim), numSamples)
    return np.array([tuple(d) for d in data])

def drawManifold(samples, hyperparam, kDistances = [], sampleColor = None, sampleDistanceColor = None, fileName = 'manifold.png'):
    if len(samples[0]) != 2:
        print('Samples must be 2D')
        return

    if len(kDistances) == 0:
        kDistances = kDistances(samples, hyperparam['k'])
    
    plt.scatter(samples[:, 0], samples[:, 1], c=sampleColor)

    for i in range(len(samples)):
        circle = plt.Circle((samples[i][0], samples[i][1]), kDistances[i], color=sampleDistanceColor, fill=True, alpha=0.1)
        plt.gcf().gca().add_artist(circle)
    
    plt.savefig(fileName)

def compute_histogram(image_path, bins=256):
    image = cv2.imread(image_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256]).flatten()
    hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256]).flatten()
    return np.concatenate((hist_r, hist_g, hist_b))
    #return cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()




