__package__ = 'tests'

import shutil
from metrics import *
from tools import *
from music import *
import numpy as np
import matplotlib.pyplot as plt
import glob
#from time import perf_counter_ns as pc

#silence tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from papers.generative_evaluation_prdc.prdc import *
from papers.improved_precision_and_recall_metric.precision_recall import *
from papers.Probablistic_precision_recall.pp_pr import *
from papers.precision_recall_distributions.prd_score import *

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
        axs[0].legend()

        axs[1].set_title(f'{diversityMetrics[i]}')
        axs[1].set_xlabel('N')
        axs[1].set_ylabel(diversityMetrics[i])

        plt.savefig(f'./images/sampleDimN_{fidelityMetrics[i]}_{diversityMetrics[i]}.png')
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
    values = estimatePRCurve(PSamples, QSamples, hyperparam, classifier)
    #x = [v[0] for v in values]
    #y = [v[1] for v in values]

    #plt.figure(figsize=(7, 7))

    #plt.plot(x, y, linestyle='-', label=f'{filename}')
    np.save(f'./data/PRCurve_{filename}_k{k_mod}_s{shift}.npy', values)

    #plt.xlim(0, 1)
    #plt.ylim(0, 1)

    #plt.legend()
    #plt.title('PR Curve')
    #plt.xlabel('Recall')
    #plt.ylabel('Precision')
    #plt.grid(True)
    #plt.savefig(f'./images/PRCurve_{filename}.png')
def testUnifyPrCurve(l=5001, g=1001, k_mod="4", shift=1):
    filenames = ['cov', 'ipr', 'knn', 'parzen']
    functions = [covClassifier, iprClassifier, knnClassifier, parzenClassifier]

    for i in range(len(filenames)):
        #check if the file exists
        try:
            values = np.load(f'./data/PRCurve_{filenames[i]}_k{k_mod}_s{shift}.npy')
        except:
            testCurve(functions[i], filenames[i], l, g, k_mod, shift)
        
    plt.clf()
    plt.figure(figsize=(7, 7))

    for file in filenames:
        values = np.load(f'./data/PRCurve_{file}_k{k_mod}_s{shift}.npy')
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
    plt.savefig(f'./images/PRCurve_k{k_mod}_s{shift}.png')
def testCompareResults():
    def log_message(log_file, message):
        print(message)
        log_file.write(message + '\n')

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

    with open('./results/log.txt', 'w') as log_file:
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
    n_bins = 256*3
    PSamplesImages = [f"data/butterflies_train/t_{i:05d}.jpg" for i in range(n_images)]
    QSamplesImages = [f"data/butterflies/g_{i:05d}.jpg" for i in range(n_images)]

    PSamples = np.array([tuple(compute_histogram(image,int(n_bins/3))) for image in PSamplesImages])
    QSamples = np.array([tuple(compute_histogram(image,int(n_bins/3))) for image in QSamplesImages])


    #split the data
    #Dtrain, Dtest = createDtrainDtest(PSamples, QSamples, 0.5)
    #no split
    Dtrain = [(val, 1) for val in PSamples] + [(val, 0) for val in QSamples]
    Dtest = [(val, 1) for val in PSamples] + [(val, 0) for val in QSamples]

    k = int(np.sqrt(len(Dtrain)))
    #k = 5

    #compute prdc
    prdc = compute_prdc([ x for x, y in Dtrain if y == 1], [ x for x, y in Dtrain if y == 0], k)
    for key, value in prdc.items():
        print(f"{key}: {value}")

    h = {'k': k, 'nJobs': 8}
    
    func = supportClassifier(Dtrain, h)
  
    n_examples = 5
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
            if len(fn_example) < n_examples:
                fn_example.append(val)
                #print('false negative realism score wrt QSamples', realismScore(QSamples, val, h))
                #print('false negative realism score wrt PSamples', realismScore(np.array([p for p in PSamples if not np.array_equal(p, val)]), val, h))
        
        elif not inP and inQ:
            fpr += 1
            if len(fp_example) < n_examples:
                fp_example.append(val)
                #print('false positive realism score wrt PSamples', realismScore(PSamples, val, h))
                #print('false positive realism score wrt QSamples', realismScore(np.array([q for q in QSamples if not np.array_equal(q, val)]), val, h))

    print(f"False positive rate: {fpr/N[0]}")
    print(f"False negative rate: {fnr/N[1]}")
    print(f"True positive rate: {1 - fnr/N[1]}")
    print(f"n of false positives: {fpr}")
    print(f"n of false negatives: {fnr}")
    print(f"n of true positives/true negatives: {len(tp_example)}")
    print()

    #clear the false positive and false negative folders
    shutil.rmtree('./results/false_positive_butterflies')
    os.mkdir('./results/false_positive_butterflies')

    shutil.rmtree('./results/false_negative_butterflies')
    os.mkdir('./results/false_negative_butterflies')
    os.mkdir('./results/false_positive_butterflies/closest_images')

    #save the examples
    set_of_indices = set()
    for example in fp_example:
        for i in range(n_images):
            if np.array_equal(QSamples[i], example) and i not in set_of_indices:
                print(f"False positive example: {i}")
                shutil.copyfile(QSamplesImages[i], f"./results/false_positive_butterflies/{i}.jpg")
                set_of_indices.add(i)
                break

    set_of_indices.clear()
    for example in fn_example:
        for i in range(n_images):
            if np.array_equal(PSamples[i], example) and i not in set_of_indices:
                print(f"False negative example: {i}")
                shutil.copyfile(PSamplesImages[i], f"./results/false_negative_butterflies/{i}.jpg")
                set_of_indices.add(i)
                break

    for i, example in enumerate(fp_example):
        example = np.array(example)
        distances = np.linalg.norm(PSamples - example, axis=1)
        closest_image = np.argmin(distances)
        min_distance = distances[closest_image]
        print(f"Closest image to false positive example: {closest_image}")
        print(f"Distance: {min_distance}")
        #save the closest image
        shutil.copyfile(PSamplesImages[closest_image], f"./results/false_positive_butterflies/closest_images/{closest_image}.jpg")
        
    for example in tp_example:
        for i in range(n_images):
            if np.array_equal(QSamples[i], example):
                print(f"True positive example: {i}")

                example = np.array(example)
                distances = np.linalg.norm(PSamples - example, axis=1)
                closest_image = np.argmin(distances)
                print(f"Closest image to true positive example: {closest_image}")

    plotHistComparison(PSamples, QSamples, fp_example, 'fp')
    plotHistComparison(PSamples, QSamples, fn_example, 'fn')

def testScarlatti():
    #load data 
    #fluidsynth /usr/share/soundfonts/FluidR3_GM.sf2 data/Scarlatti/real/val/0/0/8/8/0/000000000880_00.mid

    PPaths = glob.glob('data/Scarlatti/real/**/*.mid', recursive=True)
    QPaths = glob.glob('data/Scarlatti/fake/model_011809.ckpt/*.mid', recursive=False)

    metrics = [
                nPitchesPerMeasure,
                nNotesPerMeasure, 
                pitchClassHist,
                noteLengthHist 
            ]
    
    print(f'PPaths: {len(PPaths)}')
    print(f'PPaths example: {PPaths[0]}')
    print(f'QPaths: {len(QPaths)}')
    print(f'QPaths example: {QPaths[0]}')

    for m in metrics:
        print(f'{m.__name__}')

        cloud = midi2np(PPaths[0])
        print(f'cloud shape: {cloud[0].shape}')
        print(f'time signature for PPaths[0]: {cloud[1]}')
        print(f'{m.__name__} for PPaths[0]: {m(PPaths[0])}')

        cloud = midi2np(QPaths[0])
        print(f'cloud shape: {cloud[0].shape}')
        print(f'{m.__name__} for QPaths[0]: {m(QPaths[0])}')

        #PSamples = np.array([m(p) for p in PPaths])
        #QSamples = np.array([m(q) for q in QPaths])
#
        #print(f'PSamples shape: {PSamples.shape}')
        #print(f'QSamples shape: {QSamples.shape}')
        #print(f'PSamples: {PSamples}')
        #print(f'QSamples: {QSamples}')
        #print(f'PSamples example: {PSamples[0]}')
        #print(f'QSamples example: {QSamples[0]}')
#
        ##compute prdc
        #prdc = compute_prdc(PSamples, QSamples, 3)
        #for key, value in prdc.items():
        #    print(f"{key}: {value}")
#
        #print()