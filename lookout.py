__package__ = 'lookout'

import numpy as np
from sklearn.neighbors import KernelDensity
from music import *
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from scipy.spatial.distance import pdist, cdist
from scipy.stats import gaussian_kde, zscore
import matplotlib.pyplot as plt
import os
import glob
from tools import *

from joblib import Parallel, delayed

#def compute_likelihood_for_q(dists, pkde):
#    """Computes the likelihood for a single sample q."""
#    return np.prod(pkde.evaluate(dists))
#
#def parallel_likelihood(QSamples, PSamples, pkde):
#    """Parallelizes the likelihood computation for all QSamples."""
#    # Compute pairwise distances (QSamples x PSamples)
#    dists = cdist(QSamples, PSamples, metric='cityblock')
#
#    # Parallelize the likelihood computation
#    likelihoods = Parallel(n_jobs=-1)(  # Use all available cores
#        delayed(compute_likelihood_for_q)(dists[i], pkde) 
#        for i, _ in enumerate(QSamples)
#    )
#    
#    return likelihoods


#def loocv_kde(x, y=None, kernel='gaussian', bandwidth=None, q=0.05):
#    if y is None:
#        y = x
#
#    if bandwidth is None:
#        bandwidths = 10 ** np.linspace(-10, 1, 100)
#        grid = GridSearchCV(KernelDensity(kernel=kernel), {'bandwidth': bandwidths}, cv=LeaveOneOut())
#        grid.fit(x)
#        bandwidth = grid.best_params_['bandwidth']
#    elif isinstance(bandwidth, float): 
#        #bandwidth = bandwidth  
#        pass  
#    elif bandwidth == 'scott':
#        bandwidth = 'scott'
#    elif bandwidth == 'silverman':
#        bandwidth = 'silverman'
#    else:
#        raise ValueError('Unsupported bandwidth method')
#
#    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(x)
#    log_density = kde.score_samples(y)
#
#    threshold = np.quantile(log_density, q)
#    return log_density, threshold, bandwidth


import sys

def print_loading_bar(progress, total, bar_length=30):
    """
    Prints a loading bar in the terminal that updates on the same line.

    Args:
        progress (int): The current progress value.
        total (int): The total value to reach 100% completion.
        bar_length (int): The length of the loading bar in characters.
    """
    percentage = progress / total
    filled_length = int(bar_length * percentage)
    bar = "=" * filled_length + "-" * (bar_length - filled_length)
    sys.stdout.write(f"\r[{bar}] {percentage:.0%}")
    sys.stdout.flush()

def testScarlattiLOOCV():
    PPaths = np.array(glob.glob('data/Scarlatti/real/train/**/*.mid', recursive=True))
    models = ['model_011809.ckpt', 'model_516209.ckpt', 'model_2077006.ckpt', 'model_7083228.ckpt', 'model_7969400.ckpt']
    
    metrics = [
        nNotesPerMeasure,
        nPitchesPerMeasure,
        pitchClassHist,
        pitchClassHistPerMeasure,
        pitchClassTransMatrix,
        pitchRange,
        avgPitchShift,
        avgIOI,
        noteLengthHist,
        noteLengthHistPerMeasure,
        noteLengthTransMatrix,
    ]

    n_samples = 1000
    min_index_sample = 200
    max_index_sample = n_samples - min_index_sample

    intra_score = np.zeros((len(models), max_index_sample-min_index_sample))
    inter_score = np.zeros((len(models), max_index_sample-min_index_sample))

    matrix_inter = {}

    with open('./logs/lookde_scarlatti_log.txt', 'w') as log_file:
        for m in metrics:
            log_message(log_file,'========================================')
            log_message(log_file,m.__name__)
            log_message(log_file,'========================================')

            PSamples = np.array([m(p) for p in PPaths[:max_index_sample-min_index_sample]])

            #calculate the psample kde
            pdists = pdist(PSamples, metric='cityblock')
            #create the j copy of kde for each job
            pkde = gaussian_kde(pdists)

            matrix_inter[m.__name__] = {}

            for i, mod in enumerate(models):
                log_message(log_file, f'- MODEL - {mod}')

                QPaths = np.array([f'data/Scarlatti/fake/{mod}/{i:010d}.mid' for i in range(min_index_sample, max_index_sample)])
                QSamples = np.array([m(p) for p in QPaths])

                matrix_inter[m.__name__][mod] = {}


                #likelihoods = parallel_likelihood(QSamples, PSamples, pkde)
                likelihoods = []
                dists = cdist(QSamples, PSamples, metric='cityblock')
                for j, _ in enumerate(QSamples):
                    print_loading_bar(j, len(QSamples))
                    likelihoods.append(np.mean(pkde.evaluate(dists[j])))

                threshold = np.quantile(likelihoods, 0.05)
                #print(threshold)
                #print(likelihoods)

                sample_index = np.argsort(likelihoods) + min_index_sample
                fp_index = sample_index[likelihoods < threshold]
                tp_index = sample_index[likelihoods >= threshold]

                matrix_inter[m.__name__][mod]["false_positives"] = fp_index
                matrix_inter[m.__name__][mod]["true_positives"] = tp_index[::-1]

                print(matrix_inter[m.__name__][mod]["false_positives"])


                """
                loo_intra, t_intra = loocv_kde(QSamples, QSamples, bandwidth = 'silverman')
                #loo_inter, t_inter = loocv_kde(QSamples, PSamples)
                loo_inter, t_inter = loocv_kde(QSamples, PSamples, bandwidth = 'silverman')

                log_message(log_file, f'avg Intra: {np.mean(loo_intra)}')
                log_message(log_file, f'avg Inter: {np.mean(loo_inter)}')

                log_message(log_file, f'std Intra: {np.std(loo_intra)}')
                log_message(log_file, f'std Inter: {np.std(loo_inter)}')

                log_message(log_file, f'threshold Intra: {t_intra}')
                log_message(log_file, f'threshold Inter: {t_inter}')

                log_message(log_file, f'n low likelihood intra: {np.sum(loo_intra < t_intra)}')
                log_message(log_file, f'n low likelihood inter: {np.sum(loo_inter < t_inter)}')

                # order samples by likelihood and save as true positives the ones with likelihood above the threshold and as false positives the ones below

                sample_index = np.argsort(loo_inter) + min_index_sample
                fp_index = sample_index[loo_inter < t_inter]
                tp_index = sample_index[loo_inter >= t_inter]

                matrix_inter[m.__name__][mod]['false_positives'] = fp_index
                matrix_inter[m.__name__][mod]['true_positives'] = tp_index[::-1]

                intra_score[i] += loo_intra < t_intra
                inter_score[i] += loo_inter < t_inter
                """

        best_inter = np.argmax(inter_score, axis=1)
        best_inter_path = QPaths[best_inter+min_index_sample]
        for i, p in enumerate(best_inter_path):
            log_message(log_file, f'Best inter model {models[i]}: {p}')

    np.save('./data/test_matrix_inter.npy', matrix_inter)

    plt.clf()

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    n_bins = len(metrics)+1
    bar_width = 0.2

    for i in range(len(models)):
        intra_binned = np.zeros(n_bins)
        inter_binned = np.zeros(n_bins)

        for j in range(n_bins-1):
            intra_binned[j] = np.sum(intra_score[i] == j)
            inter_binned[j] = np.sum(inter_score[i] == j)        

        positions = np.arange(n_bins) + i * bar_width

        axs[0].bar(positions, intra_binned, width=bar_width, label=models[i])
        axs[1].bar(positions, inter_binned, width=bar_width, label=models[i])

    group_center_positions = np.arange(n_bins) + bar_width  * len(models) - bar_width / 2
    axs[0].set_xticks(group_center_positions)
    axs[0].set_xticklabels([i for i in range(n_bins)])
    axs[0].set_title('Intra Score Distribution')
    axs[0].set_xlabel('n metrics')
    axs[0].set_ylabel('n points')
    axs[0].legend()

    axs[1].set_xticks(group_center_positions)
    axs[1].set_xticklabels([i for i in range(n_bins)])
    axs[1].set_title('Inter Score Distribution')
    axs[1].set_xlabel('n metrics')
    axs[1].set_ylabel('n points')
    axs[1].legend()

    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('./images/realworldexperiments/scarlatti/kde/hist.png')

    updateIndexHtml('./images/realworldexperiments/scarlatti/kde/examples/data.json', 
                    [m.__name__ for m in metrics],
                    models,
                    "./data/matrix_inter.npy")

