__package__ = 'tests'

from metrics import *
from tools import *
from music import *
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from scipy.spatial.distance import cdist, pdist

#from time import perf_counter_ns as pc

#silence tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from papers.generative_evaluation_prdc.prdc import *
from papers.improved_precision_and_recall_metric.precision_recall import *
from papers.Probablistic_precision_recall.pp_pr import *
from papers.precision_recall_distributions.prd_score import *

from midi2audio import FluidSynth
import requests

def testKSamples(distribution_):
    def calculate_fidelity_metrics(PSamples, QSamples, hyperparams, PkDistances, QkDistances, nJobs):
        return [
            precisionCover(PSamples, QSamples, hyperparams, QkDistances, nJobs),
            improvedPrecisionCover(PSamples, QSamples, hyperparams, PkDistances, nJobs),
            density(PSamples, QSamples, hyperparams, PkDistances, nJobs),
            probPrecisionCover(PSamples, QSamples, hyperparams, nJobs)
        ]

    def calculate_diversity_metrics(PSamples, QSamples, hyperparams, PkDistances, QkDistances, nJobs):
        return [
            recallCover(PSamples, QSamples, hyperparams, PkDistances, nJobs),
            improvedRecallCover(PSamples, QSamples, hyperparams, QkDistances, nJobs),
            coverage(PSamples, QSamples, hyperparams, PkDistances, nJobs),
            probRecallCover(PSamples, QSamples, hyperparams, nJobs)
        ]    

    N = [500, 1000, 2000, 4000, 8000, 16000]
    K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dim = 64
    nJobs = 16
    distribution = distribution_

    fidelityMetrics = ['precision', 'iPrecision', 'density', 'pPrecision']
    diversityMetrics = ['recall', 'iRecall', 'coverage', 'pRecall']

    fidelityMatrix = np.zeros((len(N), len(K), len(fidelityMetrics)))
    diversityMatrix = np.zeros((len(N), len(K), len(diversityMetrics)))

    for i, n in enumerate(N):

        PSamples, QSamples = generate_samples(distribution, dim, n)

        for j, k in enumerate(K):

            print(f'N: {n}, K: {k}')

            hyperparams = {'k': k, 'a': 1.2, 'C': 3, 'Rp': 1, 'Rq': 1}

            PkDistances = kDistances(PSamples, k, nJobs)
            QkDistances = kDistances(QSamples, k, nJobs)

            hyperparams['Rp'] = getRThreshold(PSamples, hyperparams, PkDistances, nJobs)
            hyperparams['Rq'] = getRThreshold(QSamples, hyperparams, QkDistances, nJobs)

            fidelityMatrix[i, j] = calculate_fidelity_metrics(PSamples, QSamples, hyperparams, PkDistances, QkDistances, nJobs)
            diversityMatrix[i, j] = calculate_diversity_metrics(PSamples, QSamples, hyperparams, PkDistances, QkDistances, nJobs)

            for idx, metric in enumerate(fidelityMetrics):
                print(f'{metric}: {fidelityMatrix[idx]}')
            for idx, metric in enumerate(diversityMetrics):
                print(f'{metric}: {diversityMatrix[idx]}')
            print()

    np.save(f'./data/{distribution}FidelityMatrix.npy', fidelityMatrix)
    np.save(f'./data/{distribution}DiversityMatrix.npy', diversityMatrix)

    plot_matrices(distribution, fidelityMatrix, diversityMatrix, fidelityMetrics, diversityMetrics, N, K)
def testNormShift():
    N = 1000
    delta = 0.05
    shift = [-1 + delta*i for i in range(41)]
    dim = 64
    nJobs_ = 16
    outlierShift = 1

    fidelityMetrics = ['precision', 'iPrecision', 'density', 'pPrecision']
    diversityMetrics = ['recall', 'iRecall', 'coverage', 'pRecall']

    fidelityMatrix = np.zeros((len(shift), len(fidelityMetrics), 3))
    diversityMatrix = np.zeros((len(shift), len(diversityMetrics), 3))
    hyperparams = {'k': 5, 'a': 1.2, 'C': 3, 'Rp': 1, 'Rq': 1}

    PSamples = normalData(dim, N)
    PkDistances_sqrt = kDistances(PSamples, int(np.sqrt(N)), nJobs_)
    PkDistances_3 = kDistances(PSamples, 3, nJobs_)
    PkDistances_5 = kDistances(PSamples, 5, nJobs_)
    PkDistances_4 = kDistances(PSamples, 4, nJobs_)
    PSamplesWithOutlier = np.vstack((PSamples[:-1], np.array([outlierShift*np.ones(dim)])))
    PkDistances_sqrt_withOutlier = kDistances(PSamplesWithOutlier, int(np.sqrt(N)), nJobs_)
    PkDistances_3_withOutlier = kDistances(PSamplesWithOutlier, 3, nJobs_)
    PkDistances_5_withOutlier = kDistances(PSamplesWithOutlier, 5, nJobs_)
    PkDistances_4_withOutlier = kDistances(PSamplesWithOutlier, 4, nJobs_)

    QSamples = normalData(dim, N)
    QSamples = QSamples + shift[0]*np.ones(dim)
    QkDistances_sqrt = kDistances(QSamples, int(np.sqrt(N)), nJobs_)
    QkDistances_3 = kDistances(QSamples, 3, nJobs_)
    #QkDistances_5 = kDistances(QSamples, 5, nJobs_)
    QkDistances_4 = kDistances(QSamples, 4, nJobs_)
    QSamplesWithOutlier = np.vstack((QSamples[:-1], np.array([outlierShift*np.ones(dim)])))
    QSamplesWithOutlier = QSamplesWithOutlier + shift[0]*np.ones(dim)
    QSamplesWithOutlier[-1] = QSamplesWithOutlier[-1] - shift[0]*np.ones(dim)
    QkDistances_sqrt_withOutlier = kDistances(QSamplesWithOutlier, int(np.sqrt(N)), nJobs_)
    QkDistances_3_withOutlier = kDistances(QSamplesWithOutlier, 3, nJobs_)
    #QkDistances_5_withOutlier = kDistances(QSamplesWithOutlier, 5, nJobs_)
    QkDistances_4_withOutlier = kDistances(QSamplesWithOutlier, 4, nJobs_)

    for i, s in enumerate(shift):
        
        print(f'Shift: {s}')
        hyperparams['k'] = int(np.sqrt(len(PSamples)))

        fidelityMatrix[i, 0, 0] = precisionCover(PSamples, QSamples, hyperparams, QkDistances_sqrt, nJobs_)
        fidelityMatrix[i, 0, 1] = precisionCover(PSamplesWithOutlier, QSamples, hyperparams, QkDistances_sqrt, nJobs_)
        fidelityMatrix[i, 0, 2] = precisionCover(PSamples, QSamplesWithOutlier, hyperparams, QkDistances_sqrt_withOutlier, nJobs_)
        diversityMatrix[i, 0, 0] = recallCover(PSamples, QSamples, hyperparams, PkDistances_sqrt, nJobs_)
        diversityMatrix[i, 0, 1] = recallCover(PSamplesWithOutlier, QSamples, hyperparams, PkDistances_sqrt_withOutlier, nJobs_)
        diversityMatrix[i, 0, 2] = recallCover(PSamples, QSamplesWithOutlier, hyperparams, PkDistances_sqrt, nJobs_)
        
        hyperparams['k'] = 3
        
        fidelityMatrix[i, 1, 0] = improvedPrecisionCover(PSamples, QSamples, hyperparams, PkDistances_3, nJobs_)
        fidelityMatrix[i, 1, 1] = improvedPrecisionCover(PSamplesWithOutlier, QSamples, hyperparams, PkDistances_3_withOutlier, nJobs_)
        fidelityMatrix[i, 1, 2] = improvedPrecisionCover(PSamples, QSamplesWithOutlier, hyperparams, PkDistances_3, nJobs_)
        diversityMatrix[i, 1, 0] = improvedRecallCover(PSamples, QSamples, hyperparams, QkDistances_3, nJobs_)
        diversityMatrix[i, 1, 1] = improvedRecallCover(PSamplesWithOutlier, QSamples, hyperparams, QkDistances_3, nJobs_)
        diversityMatrix[i, 1, 2] = improvedRecallCover(PSamples, QSamplesWithOutlier, hyperparams, QkDistances_3_withOutlier, nJobs_)

        hyperparams['k'] = 5

        fidelityMatrix[i, 2, 0] = density(PSamples, QSamples, hyperparams, PkDistances_5, nJobs_)
        fidelityMatrix[i, 2, 1] = density(PSamplesWithOutlier, QSamples, hyperparams, PkDistances_5_withOutlier, nJobs_)
        fidelityMatrix[i, 2, 2] = density(PSamples, QSamplesWithOutlier, hyperparams, PkDistances_5, nJobs_)
        diversityMatrix[i, 2, 0] = coverage(PSamples, QSamples, hyperparams, PkDistances_5, nJobs_)
        diversityMatrix[i, 2, 1] = coverage(PSamplesWithOutlier, QSamples, hyperparams, PkDistances_5_withOutlier, nJobs_)
        diversityMatrix[i, 2, 2] = coverage(PSamples, QSamplesWithOutlier, hyperparams, PkDistances_5, nJobs_)

        hyperparams['k'] = 4
        hyperparams['Rp'] = getRThreshold(PSamples, hyperparams, PkDistances_4, nJobs_)
        hyperparams['Rq'] = getRThreshold(QSamples, hyperparams, QkDistances_4, nJobs_)

        fidelityMatrix[i, 3, 0] = probPrecisionCover(PSamples, QSamples, hyperparams, nJobs = nJobs_)
        diversityMatrix[i, 3, 0] = probRecallCover(PSamples, QSamples, hyperparams, nJobs = nJobs_)

        temp = hyperparams['Rp']
        hyperparams['Rp'] = getRThreshold(PSamplesWithOutlier, hyperparams, PkDistances_4_withOutlier, nJobs_)
        fidelityMatrix[i, 3, 1] = probPrecisionCover(PSamplesWithOutlier, QSamples, hyperparams, nJobs = nJobs_)
        diversityMatrix[i, 3, 1] = probRecallCover(PSamplesWithOutlier, QSamples, hyperparams, nJobs = nJobs_)

        hyperparams['Rp'] = temp
        hyperparams['Rq'] = getRThreshold(QSamplesWithOutlier, hyperparams, QkDistances_4_withOutlier, nJobs_)
        fidelityMatrix[i, 3, 2] = probPrecisionCover(PSamples, QSamplesWithOutlier, hyperparams, nJobs = nJobs_)
        diversityMatrix[i, 3, 2] = probRecallCover(PSamples, QSamplesWithOutlier, hyperparams, nJobs = nJobs_)

        QSamples = QSamples + delta*np.ones(dim)
        QSamplesWithOutlier = np.vstack((QSamples[:-1], np.array([outlierShift*np.ones(dim)])))
        
        QkDistances_sqrt_withOutlier = kDistances(QSamplesWithOutlier, int(np.sqrt(N)), nJobs_)
        QkDistances_3_withOutlier = kDistances(QSamplesWithOutlier, 3, nJobs_)
        QkDistances_4_withOutlier = kDistances(QSamplesWithOutlier, 4, nJobs_)

    np.save(f'./data/shiftFidelityMatrix.npy', fidelityMatrix)
    np.save(f'./data/shiftDiversityMatrix.npy', diversityMatrix)

    for i in range(len(fidelityMetrics)):
        plt.clf()
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        fig.suptitle(f'Shifted {fidelityMetrics[i]} and {diversityMetrics[i]}')

        data1 = fidelityMatrix[:, i, :]
        data2 = diversityMatrix[:, i, :]

        for j in range(3):
            axs[0].plot(shift, data1[:, j], linestyle='--', label=f'Case {j+1}')
            axs[1].plot(shift, data2[:, j], linestyle='--', label=f'Case {j+1}')

        axs[0].set_title(fidelityMetrics[i])
        axs[0].set_xlabel('\u03B4')
        axs[0].set_ylabel(fidelityMetrics[i])
        axs[0].set_ylim([0, 1])
        axs[0].legend()


        axs[1].set_title(diversityMetrics[i])
        axs[1].set_xlabel('\u03B4')
        axs[1].set_ylabel(diversityMetrics[i])
        axs[1].set_ylim([0, 1])

        plt.savefig(f'./images/shift_{fidelityMetrics[i]}_{diversityMetrics[i]}.png')
def testSampleDimN():
    N = [50, 100, 200, 400, 800, 1600]
    D = [2, 4, 8, 16, 32, 64]
    smoothout = 25
    nJobs_ = 16

    fidelityMetrics = ['precision', 'iPrecision', 'density', 'pPrecision']
    diversityMetrics = ['recall', 'iRecall', 'coverage', 'pRecall']

    fidelityMatrix = np.zeros((len(N), len(D), len(fidelityMetrics)))
    diversityMatrix = np.zeros((len(N), len(D), len(diversityMetrics)))

    for i, n in enumerate(N):
        for j, d in enumerate(D):

            print(f'N: {n}, D: {d}')
            
            for _ in range(smoothout):

                PSamples = normalData(d, n)
                QSamples = normalData(d, n)

                hyperparams = {'k': 3, 'a': 1.2, 'C': 3, 'Rp': 1, 'Rq': 1}

                hyperparams['k'] = int(np.sqrt(n))
                fidelityMatrix[i, j, 0] += precisionCover(PSamples, QSamples, hyperparams, nJobs = nJobs_)
                diversityMatrix[i, j, 0] += recallCover(PSamples, QSamples, hyperparams, nJobs = nJobs_)
                
                hyperparams['k'] = 3
                fidelityMatrix[i, j, 1] += improvedPrecisionCover(PSamples, QSamples, hyperparams, nJobs = nJobs_)
                diversityMatrix[i, j, 1] += improvedRecallCover(PSamples, QSamples, hyperparams, nJobs = nJobs_)
                
                hyperparams['k'] = 5
                fidelityMatrix[i, j, 2] += density(PSamples, QSamples, hyperparams, nJobs = nJobs_)
                diversityMatrix[i, j, 2] += coverage(PSamples, QSamples, hyperparams, nJobs = nJobs_)

                hyperparams['k'] = 4
                hyperparams['Rp'] = getRThreshold(PSamples, hyperparams, nJobs = nJobs_)
                hyperparams['Rq'] = getRThreshold(QSamples, hyperparams, nJobs = nJobs_)
                fidelityMatrix[i, j, 3] += probPrecisionCover(PSamples, QSamples, hyperparams, nJobs = nJobs_)
                diversityMatrix[i, j, 3] += probRecallCover(PSamples, QSamples, hyperparams, nJobs = nJobs_)

            fidelityMatrix[i, j, :] /= smoothout
            diversityMatrix[i, j, :] /= smoothout
            print()

    np.save(f'./data/sampleDimNFidelityMatrix.npy', fidelityMatrix)
    np.save(f'./data/sampleDimNDiversityMatrix.npy', diversityMatrix)


    for i in range(len(fidelityMetrics)):
        plt.clf()
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        fig.suptitle(f'Sample Dimension N {fidelityMetrics[i]} and {diversityMetrics[i]}')

        for d in range(len(D)):
            axs[0].plot(N, fidelityMatrix[:, d, i], linestyle='-', label=f'Dim: {D[d]}')
            axs[1].plot(N, diversityMatrix[:, d, i], linestyle='-', label=f'Dim: {D[d]}')

        axs[0].set_title(f'{fidelityMetrics[i]}')
        axs[0].set_xlabel('N')
        axs[0].set_ylabel(fidelityMetrics[i])
        axs[0].set_ylim([0.0, 1.0])
        axs[0].legend()

        axs[1].set_title(f'{diversityMetrics[i]}')
        axs[1].set_xlabel('N')
        axs[1].set_ylabel(diversityMetrics[i])
        axs[1].set_ylim([0.0, 1.0])

        plt.savefig(f'./images/toyexperiments/ksampledim/sampleDimN_{fidelityMetrics[i]}_{diversityMetrics[i]}.png')
def testCurve(classifier, filename, l=5001, g=1001, k_mod="4", shift=1):
    #def f(x):
        #erf(((2x)/(m))) + ℯ^(x - m + 3)
        #return math.erf((3*x)/m) + math.exp(x - m + 3)

    gammas = [np.tan(np.pi/2 *i/(g+1)) for i in range(1, g)]
    lambdas = [np.tan(np.pi/2 *i/(l+1)) for i in range(1, l)]

    #lambdas.insert(0, 0.000000001) 
    #lambdas.append(100000000)          
    #it is possible that ˆα^M_λ > 1. As a remedy, we always complement 
    #F_M with the trivial classifiers f ≡ 1 and f ≡ 0 that predict either P or Q uniformly
    gammas.insert(0, 0.)
    gammas.append(100000000) #np.inf leads to overflow


    d = 64
    n = 10000
    PSamples = normalData(d, n)
    QSamples = normalData(d, n, shift/math.sqrt(d))

    k = 4
    if k_mod == "sqrt":
        k = int(np.sqrt(n)) #should i divide by 2?

    hyperparam = {'Lambda': lambdas, 'Gamma': gammas, 'k': k, 'nJobs': 16}

    print(f'calculating pr curves for {filename} ...')

    #plt.clf()
    #values = estimatePRCurve(PSamples, QSamples, hyperparam, classifier)
    #x = [v[0] for v in values]
    #y = [v[1] for v in values]

    #plt.figure(figsize=(7, 7))

    #plt.plot(x, y, linestyle='-', label=f'{filename}')
    #np.save(f'./data/PRCurve_{filename}_k{k_mod}_s{shift}.npy', values)

    #no split
    Dtrain = [(val, 1) for val in PSamples] + [(val, 0) for val in QSamples]
    Dtest = [(val, 1) for val in PSamples] + [(val, 0) for val in QSamples]

    func = classifier(Dtrain, hyperparam)

    values = estimatePRD(func, Dtest, hyperparam)

    np.save(f'./data/PRCurve_{filename}_nosplit_k{k_mod}_s{shift}.npy', values)


def testUnifyPrCurve(l=5001, g=1001, k_mod="4", shift=1):
    filenames = ['cov', 'ipr', 'knn', 'parzen']
    functions = [covClassifier, iprClassifier, knnClassifier, parzenClassifier]

    for i in range(len(filenames)):
        #check if the file exists
        try:
            values = np.load(f'./data/PRCurve_{filenames[i]}_nosplit_k{k_mod}_s{shift}.npy')
        except:
            testCurve(functions[i], filenames[i], l, g, k_mod, shift)
        
    plt.clf()
    plt.figure(figsize=(7, 7))

    for file in filenames:
        values = np.load(f'./data/PRCurve_{file}_nosplit_k{k_mod}_s{shift}.npy')
        x = [v[0] for v in values]
        y = [v[1] for v in values]

        plt.plot(x, y, linestyle='-', label=f'{file}')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.legend()
    plt.title('PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.savefig(f'./images/PRCurve_nosplit_k{k_mod}_s{shift}.png')
def testCompareResults():
    def printComparison(log_file, test, x, y):
        if x != y:
            log_message(log_file, f'{test} failed')
            log_message(log_file, f'my value: {x}')
            log_message(log_file, f'actual value: {y}')
            log_message(log_file, f'difference: {x - y}')
        else:
            log_message(log_file, f'{test} passed')

    def compare_improvedPrecisionAndRecallMetric(log_file, P_samples, Q_samples, hyperparam, P_distances, Q_distances, description):
        precision = improvedPrecisionCover(P_samples, Q_samples, hyperparam, P_distances, hyperparam['nJobs'])
        recall = improvedRecallCover(P_samples, Q_samples, hyperparam, Q_distances, hyperparam['nJobs'])
        state = knn_precision_recall_features(P_samples, Q_samples)
        printComparison(log_file, f'precision test for {description}', precision, state['precision'])
        printComparison(log_file, f'recall test for {description}', recall, state['recall'])
        return precision, recall

    def compare_generativeEvaluationPRDC(log_file, P_samples, Q_samples, hyperparam, P_distances, Q_distances, description):
        precision = improvedPrecisionCover(P_samples, Q_samples, hyperparam, P_distances, hyperparam['nJobs'])
        recall = improvedRecallCover(P_samples, Q_samples, hyperparam, Q_distances, hyperparam['nJobs'])
        density_ = density(P_samples, Q_samples, hyperparam, P_distances, hyperparam['nJobs'])
        coverage_ = coverage(P_samples, Q_samples, hyperparam, P_distances, hyperparam['nJobs'])
        prdc = compute_prdc(P_samples, Q_samples, hyperparam['k'])
        printComparison(log_file, f'precision test for {description}', precision, prdc['precision'])
        printComparison(log_file, f'recall test for {description}', recall, prdc['recall'])
        printComparison(log_file, f'density test for {description}', density_, prdc['density'])
        printComparison(log_file, f'coverage test for {description}', coverage_, prdc['coverage'])

    def compare_probabilisticPrecisionRecall(log_file, P_samples, Q_samples, hyperparam, P_distances, Q_distances, description):
        hyperparam['Rp'] = getRThreshold(P_samples, hyperparam, P_distances, hyperparam['nJobs'])
        hyperparam['Rq'] = getRThreshold(Q_samples, hyperparam, Q_distances, hyperparam['nJobs'])
        pairwise_distance_real, pairwise_distance_fake, _, _ = get_pairwise_distances(P_samples, Q_samples)
        Rp = get_average_of_knn_distance(pairwise_distance_real, hyperparam['k']) * hyperparam['a']
        Rq = get_average_of_knn_distance(pairwise_distance_fake, hyperparam['k']) * hyperparam['a']
        printComparison(log_file, f'Rp test for {description}', hyperparam['Rp'], Rp)
        printComparison(log_file, f'Rq test for {description}', hyperparam['Rq'], Rq)
        pp, pr = compute_pprecision_precall(P_samples, Q_samples, hyperparam['a'], hyperparam['k'])
        p_precision = probPrecisionCover(P_samples, Q_samples, hyperparam, hyperparam['nJobs'])
        p_recall = probRecallCover(P_samples, Q_samples, hyperparam, hyperparam['nJobs'])
        printComparison(log_file, f'p_precision test for {description}', p_precision, pp)
        printComparison(log_file, f'p_recall test for {description}', p_recall, pr)

    dim = 64
    n = 10000
    PNormalSamples = normalData(dim, n)
    PUniformSamples = uniformData(dim, n)
    QNormalSamples = normalData(dim, n)
    QNormalShiftedSamples = normalData(dim, n, 3 / math.sqrt(dim))
    QUniformSamples = uniformData(dim, n)

    hyperparam = {'k': 3, 'a': 1.2, 'C': 3, 'Rp': 1, 'Rq': 1, 'nJobs': 8}

    PkNormalDistances = kDistances(PNormalSamples, hyperparam['k'], hyperparam['nJobs'])
    PkUniformDistances = kDistances(PUniformSamples, hyperparam['k'], hyperparam['nJobs'])
    QkNormalDistances = kDistances(QNormalSamples, hyperparam['k'], hyperparam['nJobs'])
    QkNormalShiftedDistances = kDistances(QNormalShiftedSamples, hyperparam['k'], hyperparam['nJobs'])
    QkUniformDistances = kDistances(QUniformSamples, hyperparam['k'], hyperparam['nJobs'])

    with open('./logs/ext_impl_comp_log.txt', 'w') as log_file:
        log_message(log_file, 'Comparing results of my implementation with the actual implementation')
        log_message(log_file, '-------------------------------------------------------------')
        log_message(log_file, 'Improved Precision and Recall Metric Comparisons')
        compare_improvedPrecisionAndRecallMetric(log_file, PNormalSamples, QNormalSamples, hyperparam, PkNormalDistances, QkNormalDistances, 'normal distribution')
        compare_improvedPrecisionAndRecallMetric(log_file, PNormalSamples, QNormalShiftedSamples, hyperparam, PkNormalDistances, QkNormalShiftedDistances, 'shifted normal distribution')
        compare_improvedPrecisionAndRecallMetric(log_file, PUniformSamples, QUniformSamples, hyperparam, PkUniformDistances, QkUniformDistances, 'uniform distribution')

        log_message(log_file, '-------------------------------------------------------------')
        log_message(log_file, 'Generative Evaluation PRDC Comparisons')
        compare_generativeEvaluationPRDC(log_file, PNormalSamples, QNormalSamples, hyperparam, PkNormalDistances, QkNormalDistances, 'normal distribution')
        compare_generativeEvaluationPRDC(log_file, PNormalSamples, QNormalShiftedSamples, hyperparam, PkNormalDistances, QkNormalShiftedDistances, 'shifted normal distribution')
        compare_generativeEvaluationPRDC(log_file, PUniformSamples, QUniformSamples, hyperparam, PkUniformDistances, QkUniformDistances, 'uniform distribution')

        log_message(log_file, '-------------------------------------------------------------')
        log_message(log_file, 'Probabilistic Precision Recall Comparisons')
        #here k=4 would be recommended but since we already have the distances for k=3, we will use them
        compare_probabilisticPrecisionRecall(log_file, PNormalSamples, QNormalSamples, hyperparam, PkNormalDistances, QkNormalDistances, 'normal distribution')
        compare_probabilisticPrecisionRecall(log_file, PNormalSamples, QNormalShiftedSamples, hyperparam, PkNormalDistances, QkNormalShiftedDistances, 'shifted normal distribution')
        compare_probabilisticPrecisionRecall(log_file, PUniformSamples, QUniformSamples, hyperparam, PkUniformDistances, QkUniformDistances, 'uniform distribution')
def testButterflies():
    #load data
    n_images = 896
    n_bins = 180
    #n_examples = 5

    metrics = [hue_histogram, saturation_histogram, hsv_histogram, rgb_histogram, value_histogram, grayscale_histogram]
    PPaths = [f"data/butterflies_train/t_{i:05d}.jpg" for i in range(n_images)]
    QPaths = [f"data/butterflies/g_{i:05d}.jpg" for i in range(n_images)]

    with open('./logs/butterflies_log.txt', 'w') as log_file:
        for m in metrics:    
            log_message(log_file, '-------------------------------------------------------------')
            log_message(log_file, f"Metric: {m.__name__}")
            PSamples = np.array([tuple(m(image,n_bins)) for image in PPaths])
            QSamples = np.array([tuple(m(image,n_bins)) for image in QPaths])

            #split the data
            #split_ratio = 0.8
            #Dtrain, Dtest = createDtrainDtest(PSamples, QSamples, split_ratio)
            #no split
            Dtrain = [(val, 1) for val in PSamples] + [(val, 0) for val in QSamples]
            Dtest = [(val, 1) for val in PSamples] + [(val, 0) for val in QSamples]

            #k = 3
            k = int(np.sqrt(len(Dtrain)*.5))

            #compute prdc
            prdc = compute_prdc([ x for x, y in Dtrain if y == 1], [ x for x, y in Dtrain if y == 0], k)
            for key, value in prdc.items():
                log_message(log_file, f"{key}: {value}")

            h = {'k': k, 'nJobs': 8}
            #norm1
            ord_ = 1
            #norm2
            #ord_ = 2
            
            def norm(x, axis = None):
                return np.linalg.norm(x, axis=axis, ord=ord_)

            func = supportClassifier(Dtrain, h, norm)
        
            fpr = 0
            fnr = 0
            N = [0, 0]
            fp_example = []
            fn_example = []
            tp_example = []

            for val, res in Dtest:
                inP, inQ = func(val)
                N[res] += 1

                if inP and inQ and res == 0: #memorize only qSamples that are inside the PSamples support
                    tp_example.append(val)

                if inP and not inQ:
                    fnr += 1
                    #if len(fn_example) < n_examples:
                    fn_example.append(val)
                        #log_message(log_file,'false negative realism score wrt QSamples', realismScore(QSamples, val, h))
                        #log_message(log_file,'false negative realism score wrt PSamples', realismScore(np.array([p for p in PSamples if not np.array_equal(p, val)]), val, h))
                
                elif not inP and inQ:
                    fpr += 1
                    #if len(fp_example) < n_examples:
                    fp_example.append(val)
                        #log_message(log_file,'false positive realism score wrt PSamples', realismScore(PSamples, val, h))
                        #log_message(log_file,'false positive realism score wrt QSamples', realismScore(np.array([q for q in QSamples if not np.array_equal(q, val)]), val, h))

            log_message(log_file,f"False positive rate: {fpr/N[0]}")
            log_message(log_file,f"False negative rate: {fnr/N[1]}")
            log_message(log_file,f"True positive rate: {1 - fnr/N[1]}")
            log_message(log_file,f"n of false positives: {fpr}")
            log_message(log_file,f"n of false negatives: {fnr}")
            log_message(log_file,f"n of true positives/true negatives: {len(tp_example)}")
            log_message(log_file,"")

            #save the examples
            set_of_indices = set()
            for example in fp_example:
                for i in range(n_images):
                    if np.array_equal(QSamples[i], example) and i not in set_of_indices:
                        #log_message(log_file,f"False positive example: {i}")
                        set_of_indices.add(i)
                        break
            
            log_message(log_file,f"False positive examples: {set_of_indices}")
            fp_example = [i for i in set_of_indices]

            #set_of_indices.clear()
            #for example in fn_example:
            #    for i in range(n_images):
            #        if np.array_equal(PSamples[i], example) and i not in set_of_indices:
            #            #log_message(log_file,f"False negative example: {i}")
            #            set_of_indices.add(i)
            #            break
            #
            #log_message(log_file,f"False negative examples: {set_of_indices}")
            #fn_example = [i for i in set_of_indices]

            set_of_indices.clear()
            for example in tp_example:
                for i in range(n_images):
                    if np.array_equal(QSamples[i], example) and i not in set_of_indices:
                        #log_message(log_file,f"True positive example: {i}")
                        set_of_indices.add(i)
                        break
                
            log_message(log_file,f"True positive examples: {set_of_indices}")
            tp_example = [i for i in set_of_indices]

            fp_closest_images = []
            fp_distances = []
            for index in fp_example:
                example = np.array(QSamples[index])
                distances = np.linalg.norm(PSamples - example, axis=1, ord=ord_)
                closest_image = np.argmin(distances)
                fp_distances.append(distances[closest_image])
                #log_message(log_file,f"Closest image to false positive example: {closest_image}")
                #log_message(log_file,f"Distance: {min_distance}")
                fp_closest_images.append((QPaths[index], PPaths[closest_image]))
                
            tp_closest_images = []
            tp_distances = []
            for index in tp_example:
                example = np.array(QSamples[index])
                distances = np.linalg.norm(PSamples - example, axis=1, ord=ord_)
                closest_image = np.argmin(distances)
                tp_distances.append(distances[closest_image])
                #log_message(log_file,f"Closest image to true positive example: {closest_image}")
                #log_message(log_file,f"Distance: {min_distance}")
                tp_closest_images.append((QPaths[index], PPaths[closest_image]))
                                        

            #make a graph of false positives and closest images
            # Plot false positives and closest real images

            n_examples = 5
            #select n_examples with largest distances from fp_distances
            fp_example = [x for _, x in sorted(zip(fp_distances, fp_example), key=lambda pair: pair[0])]

            #select n_examples with smallest distances from tp_distances
            tp_example = [x for _, x in sorted(zip(tp_distances, tp_example), key=lambda pair: pair[0], reverse=True)] 

            print(fp_example[:n_examples])
            print(sorted(fp_distances[:n_examples]))
            print([realismScore(PSamples, fp_example[i], h, nJobs=16, f_dist=norm) for i in range(n_examples)])            
            print(tp_example[:n_examples])
            print(sorted(tp_distances[:n_examples], reverse=True))
            print([realismScore(PSamples, tp_example[i], h, nJobs=16, f_dist=norm) for i in range(n_examples)])

            if len(tp_example) < n_examples or len(fp_example) < n_examples:
                log_message(log_file,'not enough examples to plot')
                return
            
            fig, axs = plt.subplots(4, n_examples, figsize=(5 * n_examples, 5 * 4 ))
            fig.suptitle('False Positives and Closest Real Images')
            
            for i in range(n_examples):
                fp_path, fp_closest_image_path = fp_closest_images[i]
                tp_path, tp_closest_image_path = tp_closest_images[i]

                fp_image = cv2.imread(fp_path)
                fp_closest_image = cv2.imread(fp_closest_image_path)
                tp_image = cv2.imread(tp_path)
                tp_closest_image = cv2.imread(tp_closest_image_path)  

                axs[0, i].imshow(fp_image)
                axs[0, i].xaxis.set_visible(False)
                axs[0, i].set_yticks([])

                axs[1, i].imshow(fp_closest_image)
                axs[1, i].xaxis.set_visible(False)
                axs[1, i].set_yticks([])

                axs[2, i].imshow(tp_image)
                axs[2, i].xaxis.set_visible(False)
                axs[2, i].set_yticks([])

                axs[3, i].imshow(tp_closest_image)
                axs[3, i].xaxis.set_visible(False)
                axs[3, i].set_yticks([])

            for ax, row in zip(axs[:, 0], ['False Positive', 'Closest Real Image', 'True Positive', 'Closest Real Image']):
                ax.set_ylabel(row, rotation=90, size='large')
            
            #plt.tight_layout()
            plt.savefig(f'./images/fp_{m.__name__}.png')

            plotKDE(PSamples, QSamples, f'./images/kde_{m.__name__}.png', 'cityblock', 'silverman', 200)
            log_message(log_file,"")  
def testScarlattiReal():
    PPaths = np.array(glob.glob('data/Scarlatti/real/train/**/*.mid', recursive=True))

    QPaths = np.array(glob.glob('data/Scarlatti/real/test/**/*.mid', recursive=True))
    QPaths = np.concatenate((QPaths , np.array(glob.glob('data/Scarlatti/real/val/**/*.mid', recursive=True))))

    #print(PPaths.shape)
    #print(QPaths.shape)

    metrics = [
                nNotesPerMeasure,  
                nPitchesPerMeasure ,
                pitchClassHist,
                pitchClassHistPerMeasure,
                pitchClassTransMatrix,
                #pitchClassTransMatrixPerMeasure,
                pitchRange,
                avgPitchShift,
                avgIOI,
                noteLengthHist,
                noteLengthHistPerMeasure,
                noteLengthTransMatrix,
                #noteLengthTransMatrixPerMeasure
            ]
    
    for m in metrics:
        print(m.__name__)

        PSamples = np.array([m(p) for p in PPaths])
        QSamples = np.array([m(p) for p in QPaths])

        #dist = 'cityblock'
        plotKDE(PSamples, QSamples, f'images/TrainVSTest_{m.__name__}.png', 'cityblock', 'silverman', 200)
def testScarlatti():
    PPaths = np.array(glob.glob('data/Scarlatti/real/train/**/*.mid', recursive=True))
    models = ['model_011809.ckpt', 'model_516209.ckpt', 'model_2077006.ckpt', 'model_7083228.ckpt', 'model_7969400.ckpt']

    metrics = [
        nNotesPerMeasure,
        nPitchesPerMeasure,
        pitchClassHist,
        pitchClassHistPerMeasure,
        pitchClassTransMatrix,
        #pitchClassTransMatrixPerMeasure,
        pitchRange,
        avgPitchShift,
        avgIOI,
        noteLengthHist,
        noteLengthHistPerMeasure,
        noteLengthTransMatrix,
        #noteLengthTransMatrixPerMeasure
    ]

    dist = 'cityblock' #'euclidean'
    if dist == 'cityblock':
        ord_ = 1
    else:
        ord_ = 2

    method = 'silverman' #'scott'
    n_examples = 5 #want to see about all the false positives
    n_samples = 1000
    min_index_sample = 150
    max_index_sample = n_samples - min_index_sample
    n_kde_points = 200

    with open('./logs/scarlatti_log.txt', 'w') as log_file:
        for m in metrics:
            log_message(log_file,'========================================')
            log_message(log_file,m.__name__)
            log_message(log_file,'========================================')

            plt.clf()
            fig, axs = plt.subplots(1, len(models), figsize=(3*len(models), 8), sharey=True)
            fig.suptitle(f'Train vs Fake {m.__name__}')

            PSamples = np.array([m(p) for p in PPaths[:max_index_sample-min_index_sample]])
            PSamplesIntraDistances = pdist(PSamples, metric=dist)

            overlap_area = []
            fp = []

            for i, mod in enumerate(models):
                log_message(log_file, f'- MODEL - {mod}')

                QPaths = np.array([f'data/Scarlatti/fake/{mod}/{i:010d}.mid' for i in range(min_index_sample, max_index_sample)])
                QSamples = np.array([m(p) for p in QPaths])

                #CLASSIFICATION TASK

                #no split ( we dont want to split the data to retrieve the examples and since models are preatty accurate and we want a samplesset for false positives)
                Dtrain = [(val, 1) for val in PSamples] + [(val, 0) for val in QSamples]
                Dtest = [(val, 1) for val in PSamples] + [(val, 0) for val in QSamples]

                k = 4
                #k = int(np.sqrt(len(Dtrain)))

                #compute prdc
                prdc = compute_prdc([ x for x, y in Dtrain if y == 1], [ x for x, y in Dtrain if y == 0], k)
                for key, value in prdc.items():
                    log_message(log_file,f"{key}: {value}")

                h = {'k': k, 'nJobs': 8}
                                
                def norm(x, axis = None):
                    return np.linalg.norm(x, axis=axis, ord=ord_)

                func = supportClassifier(Dtrain, h, norm)
            
                fpr = 0
                fnr = 0
                N = [0, 0]
                fp_example = []
                fn_example = []
                tp_example = []

                for val, res in Dtest:
                    inP, inQ = func(val)
                    N[res] += 1

                    if inP and inQ and res == 0: #memorize only qSamples that are inside the PSamples support
                        tp_example.append(N[0]-1)

                    if inP and not inQ:
                        fnr += 1
                        if len(fn_example) < n_examples:
                            fn_example.append(N[1]-1)
                            #log_message(log_file,'false negative realism score wrt QSamples', realismScore(QSamples, val, h))
                            #log_message(log_file,'false negative realism score wrt PSamples', realismScore(np.array([p for p in PSamples if not np.array_equal(p, val)]), val, h))
                    
                    elif not inP and inQ:
                        fpr += 1
                        if len(fp_example) < n_examples:
                            fp_example.append(N[0]-1)
                            #log_message(log_file,'false positive realism score wrt PSamples', realismScore(PSamples, val, h))
                            #log_message(log_file,'false positive realism score wrt QSamples', realismScore(np.array([q for q in QSamples if not np.array_equal(q, val)]), val, h))

                log_message(log_file,f"False positive rate: {fpr/N[0]}")
                log_message(log_file,f"False negative rate: {fnr/N[1]}")
                log_message(log_file,f"True positive rate: {1 - fnr/N[1]}")
                log_message(log_file,f"n of false positives: {fpr}")
                log_message(log_file,f"n of false negatives: {fnr}")
                log_message(log_file,f"n of true positives/true negatives: {len(tp_example)}")
                log_message(log_file,"")
                
                if len(fp_example) > 0:
                    log_message(log_file,f'[')
                    fp_example_paths = [f'data/Scarlatti/fake/{mod}/{(min_index_sample+i):010d}.mid' for i in fp_example]
                    for path in fp_example_paths:
                        log_message(log_file,f'    {path},')                
                    log_message(log_file,f']')
                
                    log_message(log_file,"")

                fp.append(fpr)

                #KDE TASK

                QSamplesIntraDistances = pdist(QSamples, metric=dist)
                InterDistances = cdist(PSamples, QSamples, metric=dist).flatten()

                min_kde = min(np.min(PSamplesIntraDistances), np.min(QSamplesIntraDistances))
                max_kde = max(np.max(PSamplesIntraDistances), np.max(QSamplesIntraDistances))
                std_kde = max(np.std(PSamplesIntraDistances), np.std(QSamplesIntraDistances))
                kde_points = np.linspace(min_kde - std_kde, max_kde + std_kde, n_kde_points)

                pkde = kde(PSamplesIntraDistances, kde_points, method)
                qkde = kde(QSamplesIntraDistances, kde_points, method)

                overlap_area_pq = np.trapz(np.minimum(pkde, qkde), kde_points)

                log_message(log_file,f'Overlap area between P and Q: {overlap_area_pq}')
                log_message(log_file,'-------------------------------------------------------------')
                
                overlap_area.append(overlap_area_pq)

                sns.violinplot(data=PSamplesIntraDistances, bw_method=method, orient='v', inner=None, color='steelblue', saturation=0.75, linewidth=0, ax=axs[i])
                sns.violinplot(data=QSamplesIntraDistances, bw_method=method, orient='v', inner=None, color='mediumseagreen', saturation=0.75, linewidth=0, ax=axs[i])
                
                # Hide the right half for pintra_kde (first violin)
                for collection in axs[i].collections[::2]:  # Every other element corresponds to pintra_kde
                    for path in collection.get_paths():
                        path.vertices[:, 0] = np.clip(path.vertices[:, 0], -np.inf, 0)  # Keep only left half (clip on x-axis)

                # Hide the left half for qintra_kde (second violin)
                for collection in axs[i].collections[1::2]:  # Every other element corresponds to qintra_kde
                    for path in collection.get_paths():
                        path.vertices[:, 0] = np.clip(path.vertices[:, 0], 0, np.inf)  # Keep only right half (clip on x-axis)

                sns.boxplot(data=InterDistances, orient='v', color='whitesmoke',linecolor='darkslategray', saturation=0.75, width=0.02, fliersize=0, linewidth=2, ax=axs[i])
                axs[i].set_xlabel(f'dist {mod}')

                # Remove the spines (borders) of each subplot
                axs[i].spines['top'].set_visible(True)
                axs[i].spines['right'].set_visible(False)
                axs[i].spines['left'].set_visible(False)
                axs[i].spines['bottom'].set_visible(True)

                axs[i].set_xmargin(0.05)
                axs[i].set_ymargin(0.05)

                # Only show the bottom spine for the first subplot
                if i == 0:
                    axs[i].spines['left'].set_visible(True)
                    axs[i].set_ylabel('Density')

                if i == len(models) - 1:
                    axs[i].spines['right'].set_visible(True)

                # Remove y-ticks for all but the first subplot
                if i != 0:
                    axs[i].tick_params(left=False)

            plt.tight_layout()
            plt.savefig(f'images/TrainVSFake_{m.__name__}.png')

            log_message(log_file,'')
            log_message(log_file,'')
            best_oa = np.argmax(overlap_area)
            best_fp = np.argmin(fp)
            log_message(log_file,f'Best by overlap area: {models[best_oa]}')
            log_message(log_file,f'Best by numb. of false positives: {models[best_fp]}')

            log_message(log_file,'')
            log_message(log_file,'')
def testScarlattiLOOCV():
    PPaths = np.array(glob.glob('data/Scarlatti/real/train/**/*.mid', recursive=True))
    models = ['model_011809.ckpt', 'model_516209.ckpt', 'model_2077006.ckpt', 'model_7083228.ckpt', 'model_7969400.ckpt']

    metrics = [
        nNotesPerMeasure,
        nPitchesPerMeasure,
        pitchClassHist,
        pitchClassHistPerMeasure,
        pitchClassTransMatrix,
        #pitchClassTransMatrixPerMeasure,
        pitchRange,
        avgPitchShift,
        avgIOI,
        noteLengthHist,
        noteLengthHistPerMeasure,
        noteLengthTransMatrix,
        #noteLengthTransMatrixPerMeasure
    ]

    dist = 'cityblock' #'euclidean'
    if dist == 'cityblock':
        ord_ = 1
    else:
        ord_ = 2

    method = 'silverman' #'scott'
    n_samples = 1000
    min_index_sample = 150
    max_index_sample = n_samples - min_index_sample

    intra_score = np.zeros((len(models), max_index_sample-min_index_sample))
    inter_score = np.zeros((len(models), max_index_sample-min_index_sample))

    matrix_intra = np.zeros((len(metrics), len(models), max_index_sample-min_index_sample))
    matrix_inter = np.zeros((len(metrics), len(models), max_index_sample-min_index_sample))
    tresholds_intra = np.zeros((len(metrics), len(models)))
    tresholds_inter = np.zeros((len(metrics), len(models)))

    with open('./logs/lookde_scarlatti_log.txt', 'w') as log_file:
        for m in metrics:
            log_message(log_file,'========================================')
            log_message(log_file,m.__name__)
            log_message(log_file,'========================================')

            PSamples = np.array([m(p) for p in PPaths[:max_index_sample-min_index_sample]])

            for i, mod in enumerate(models):
                log_message(log_file, f'- MODEL - {mod}')

                QPaths = np.array([f'data/Scarlatti/fake/{mod}/{i:010d}.mid' for i in range(min_index_sample, max_index_sample)])
                QSamples = np.array([m(p) for p in QPaths])

                loo_intra, t_intra = loocv_kde(QSamples, QSamples)
                loo_inter, t_inter = loocv_kde(QSamples, PSamples)

                log_message(log_file, f'avg Intra: {np.mean(loo_intra)}')
                log_message(log_file, f'avg Inter: {np.mean(loo_inter)}')

                log_message(log_file, f'std Intra: {np.std(loo_intra)}')
                log_message(log_file, f'std Inter: {np.std(loo_inter)}')

                log_message(log_file, f'n low likelihood intra: {np.sum(loo_intra < np.quantile(loo_intra, 0.05))}')
                log_message(log_file, f'n low likelihood inter: {np.sum(loo_inter < np.quantile(loo_inter, 0.05))}')

                matrix_intra[metrics.index(m), i] = loo_intra
                matrix_inter[metrics.index(m), i] = loo_inter
                tresholds_intra[metrics.index(m), i] = t_intra
                tresholds_inter[metrics.index(m), i] = t_inter

                intra_score[i] += loo_intra < t_intra
                inter_score[i] += loo_inter < t_inter

        best_inter = np.argmax(inter_score, axis=1)
        best_inter_path = QPaths[best_inter+min_index_sample]
        for i, p in enumerate(best_inter_path):
            log_message(log_file, f'Best inter model {models[i]}: {p}')

    np.save('./data/matrix_intra.npy', matrix_intra)
    np.save('./data/matrix_inter.npy', matrix_inter)
    np.save('./data/tresholds_intra.npy', tresholds_intra)
    np.save('./data/tresholds_inter.npy', tresholds_inter)

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

def testScarlattiExamples():
    inter_lhood = np.load('./data/matrix_inter.npy')
    inter_thresholds = np.load('./data/tresholds_inter.npy')

    n_fp_examples = 3
    n_tp_examples = 3

    n_samples = 1000
    min_index_sample = 150
    max_index_sample = n_samples - min_index_sample

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

    # Create folder for examples if it doesn't exist
    os.makedirs('./images/realworldexperiments/scarlatti/kde/examples', exist_ok=True)
    
    for j, m in enumerate(metrics):
        os.makedirs(f'./images/realworldexperiments/scarlatti/kde/examples/{m.__name__}', exist_ok=True)
        
        for i, mod in enumerate(models):
            QPaths = np.array([f'data/Scarlatti/fake/{mod}/{i:010d}.mid' for i in range(min_index_sample, max_index_sample)])
            os.makedirs(f'./images/realworldexperiments/scarlatti/kde/examples/{m.__name__}/{mod}/fp_examples', exist_ok=True)
            os.makedirs(f'./images/realworldexperiments/scarlatti/kde/examples/{m.__name__}/{mod}/tp_examples', exist_ok=True)

            os.system(f'rm ./images/realworldexperiments/scarlatti/kde/examples/{m.__name__}/{mod}/fp_examples/*')
            os.system(f'rm ./images/realworldexperiments/scarlatti/kde/examples/{m.__name__}/{mod}/tp_examples/*')

            # Select the n_fp_examples with the lowest likelihood and the n_tp_examples with the highest likelihood
            fp_examples = np.argsort(inter_lhood[j, i])[:n_fp_examples]
            tp_examples = np.argsort(inter_lhood[j, i])[::-1][:n_tp_examples]

             # Initialize FluidSynth with the SoundFont from /usr/share/soundfonts/
            soundfont_path = '/usr/share/soundfonts/FluidR3_GM.sf2'
            if not os.path.exists(soundfont_path):
                print(f"SoundFont not found at {soundfont_path}. Please make sure it is installed.")
                return

            fs = FluidSynth(soundfont_path)
            
            for index in fp_examples:
                # Convert MIDI to MP3 for false positive examples
                wav_path = f'./images/realworldexperiments/scarlatti/kde/examples/{m.__name__}/{mod}/fp_examples/{index}.wav'
                fs.midi_to_audio(f'{QPaths[index]}', wav_path)

            for index in tp_examples:
                # Convert MIDI to MP3 for true positive examples
                wav_path = f'./images/realworldexperiments/scarlatti/kde/examples/{m.__name__}/{mod}/tp_examples/{index}.wav'
                fs.midi_to_audio(f'{QPaths[index]}', wav_path)

    #create a markdown file with the playable examples

    with open('./images/realworldexperiments/scarlatti/kde/examples/examples.md', 'w') as md_file:
        for m in metrics:
            md_file.write(f'## {m.__name__}\n')
            for mod in models:
                md_file.write(f'### {mod}\n')

                fp_paths = glob.glob(f'./images/realworldexperiments/scarlatti/kde/examples/{m.__name__}/{mod}/fp_examples/*.wav')
                tp_paths = glob.glob(f'./images/realworldexperiments/scarlatti/kde/examples/{m.__name__}/{mod}/tp_examples/*.wav')

                md_file.write(f'#### False Positives\n')
                for fp in fp_paths:
                    md_file.write(f'\n')
                    md_file.write(f'<audio controls>\n')
                    md_file.write(f'  <source src="{fp}" type="audio/wav">\n')
                    md_file.write(f'  Your browser does not support the audio element.\n')
                    md_file.write(f'</audio>\n')
                    md_file.write(f'\n')
                md_file.write(f'#### True Positives\n')
                for tp in tp_paths:
                    md_file.write(f'\n')
                    md_file.write(f'<audio controls>\n')
                    md_file.write(f'  <source src="{tp}">\n')
                    md_file.write(f'  Your browser does not support the audio element.\n')
                    md_file.write(f'</audio>\n')
                    md_file.write(f'\n')
            md_file.write(f'\n')