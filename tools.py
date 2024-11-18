__package__ = 'tools'

import numpy as np
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import pdist, cdist
from scipy.stats import gaussian_kde

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

def hue_histogram(image_path, bins=256):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()
    return hist

def saturation_histogram(image_path, bins=256):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [1], None, [bins], [0, 256]).flatten()
    return hist

def value_histogram(image_path, bins=256):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [2], None, [bins], [0, 256]).flatten()
    return hist

def grayscale_histogram(image_path, bins=256):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()
    return hist

def hsv_histogram(image_path, bins=256):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()
    hist_s = cv2.calcHist([image], [1], None, [bins], [0, 256]).flatten()
    hist_v = cv2.calcHist([image], [2], None, [bins], [0, 256]).flatten()
    return np.concatenate((hist_h, hist_s, hist_v))

def rgb_histogram(image_path, bins=256):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256]).flatten()
    hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256]).flatten()
    return np.concatenate((hist_r, hist_g, hist_b))

def plotKDE(psamples, qsamples, filename, dist='euclidean', method='scott', n_points=500):
    #print(psamples.shape)
    #print(qsamples.shape)
    PSamplesIntraDistances = pdist(psamples, metric=dist)
    QSamplesIntraDistances = pdist(qsamples, metric=dist)
    InterDistances = cdist(psamples, qsamples, metric=dist).flatten()

    #take points from the minimum - one standard deviation to the maximum + one standard deviation
    min_value = min(np.min(PSamplesIntraDistances), np.min(QSamplesIntraDistances), np.min(InterDistances))
    max_value = max(np.max(PSamplesIntraDistances), np.max(QSamplesIntraDistances), np.max(InterDistances))
    std_value = max(np.std(PSamplesIntraDistances), np.std(QSamplesIntraDistances), np.std(InterDistances))

    x = np.linspace(min_value - std_value, max_value + std_value, n_points)

    pintra_kde = kde(PSamplesIntraDistances, x, method)
    qintra_kde = kde(QSamplesIntraDistances, x, method)
    inter_kde = kde(InterDistances, x, method)

    overlap_area_pq = np.trapz(np.minimum(pintra_kde, qintra_kde), x)
    overlap_area_pi = np.trapz(np.minimum(pintra_kde, inter_kde), x)
    overlap_area_qi = np.trapz(np.minimum(qintra_kde, inter_kde), x)

    print(f'Overlap area between P and Q: {overlap_area_pq}')
    print(f'Overlap area between P and Inter: {overlap_area_pi}')
    print(f'Overlap area between Q and Inter: {overlap_area_qi}')

    plt.clf()
    plt.figure(figsize=(7, 7))
    plt.grid()

    plt.plot(x, pintra_kde, label='P Intra')
    plt.plot(x, qintra_kde, label='Q Intra')
    plt.plot(x, inter_kde, label='Inter')

    plt.legend()
    plt.savefig(filename)

def generate_samples(distribution, dim, n):
    if distribution == 'uniform':
        PSamples = uniformData(dim, n)
        QSamples = uniformData(dim, n)
    elif distribution == 'normal':
        PSamples = normalData(dim, n)
        QSamples = normalData(dim, n)
    elif distribution == 'normalwshift':
        PSamples = normalData(dim, n)
        QSamples = normalData(dim, n, 3/8)
    return PSamples, QSamples

def plot_matrices(distribution, fidelityMatrix, diversityMatrix, fidelityMetrics, diversityMetrics, N, K):
    def plot_matrix(ax, data, title, N, K, cmap):
        im = ax.imshow(data, cmap=cmap, norm=Normalize(vmin=0, vmax=1))
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_xticks(np.arange(len(K)))
        ax.set_yticks(np.arange(len(N)))
        ax.set_xticklabels(K)
        ax.set_yticklabels(N)
        ax.set_xlabel('k')
        ax.set_ylabel('Samples')
        return im

    for i in range(len(fidelityMetrics)):
        plt.clf()
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['green', 'yellow', 'red'])
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'{distribution} {fidelityMetrics[i]} and {diversityMetrics[i]}')

        im1 = plot_matrix(axs[0], fidelityMatrix[:, :, i], fidelityMetrics[i], N, K, cmap)
        im2 = plot_matrix(axs[1], diversityMatrix[:, :, i], diversityMetrics[i], N, K, cmap)

        # Adjust the layout for better spacing
        fig.tight_layout(rect=[0, 0, 1., 1.25])

        # Create a divider for the right subplot and add a colorbar
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.1)

        # Add colorbar to the last subplot (common for both)
        cbar = fig.colorbar(im2, cax=cax)

        # Save the figure ensuring nothing is cut off
        plt.savefig(f'./images/{distribution}_{fidelityMetrics[i]}_{diversityMetrics[i]}.png', 
                    bbox_inches='tight', pad_inches=0.2)


def channel_to_bandw(image_path, channel):
    image = cv2.imread(image_path)

    channels = ['r', 'g', 'b', 'h', 's', 'v']
    if channel not in channels:
        print('Invalid channel')
        return
    
    index = channels.index(channel)
    if index < 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[:, :, index]
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = image[:, :, index-3]

    return image

def gaussian_kernel(u):
    """Gaussian Kernel Function"""
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

def scott_bandwidth(data):
    """Scott's Rule for bandwidth"""
    n = len(data)
    d = 1  # Dimension of data
    std_dev = np.std(data)
    return n ** (-1.0 / (d + 4)) * std_dev

def silverman_bandwidth(data):
    """Silverman's Rule for bandwidth"""
    n = len(data)
    std_dev = np.std(data)
    return (4 / (3 * n)) ** (1 / 5) * std_dev

def kde(data, points, bandwidth_method='scott'):
    """
    Perform KDE on given data.
    
    Parameters:
    - data: array-like, 1D array of data points.
    - points: array-like, points where to evaluate the KDE.
    - bandwidth_method: 'scott' or 'silverman' for bandwidth selection.
    
    Returns:
    - kde_values: array of KDE values at each point.
    """
    n = len(data)
    
    # Select bandwidth
    if bandwidth_method == 'scott':
        h = scott_bandwidth(data)
    elif bandwidth_method == 'silverman':
        h = silverman_bandwidth(data)
    else:
        raise ValueError("Unsupported bandwidth method.")
    
    kde_values = np.zeros(len(points))
    
    # KDE formula
    for i, x in enumerate(points):
        kde_values[i] = (1 / (n * h)) * np.sum(gaussian_kernel((x - data) / h))
    
    return kde_values

def loocv_kde(x, y = None, dist='cityblock', method='silverman', q=0.05):
    if y is None:
        y = x
    
    if len(x) != len(y):
        raise ValueError('x and y must have the same length')
    
    n_samples = len(x)
    dists = cdist(x, y, metric=dist)

    if method == 'scott':
        bandwidth = scott_bandwidth(dists)
    elif method == 'silverman':
        bandwidth = silverman_bandwidth(dists)
    else:
        raise ValueError('Unsupported bandwidth method')

    weights = np.exp(-0.5 * (dists / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))
    kde_values = np.sum(weights, axis=1) / n_samples

    threshold = np.quantile(kde_values, q)

    l = np.zeros(n_samples)

    for i in range(n_samples):
        l[i] = (np.sum(weights[i]) - weights[i, i]) / (n_samples - 1)

    return l, threshold

def log_message(log_file, message):
    print(message)
    log_file.write(message + '\n')