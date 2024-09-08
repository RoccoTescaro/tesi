__package__ = 'metrics'

from tools import kDistances
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import math

# PRECISION-RECALL COVER

# Precision and Recall are symmetric metrics which mean that could be implemented in the same function by switching the samples.
# Nonetheless, we will implement them separately for clarity.
def precisionCover(PSamples, QSamples, hyperparam, QkDistances = [], nJobs = 1, f_dist = np.linalg.norm):
    if len(QkDistances) == 0:
        QkDistances = kDistances(QSamples, hyperparam['k'], nJobs, f_dist)
    
    def worker(PSamples, QSample, QkDistance, k, C):
        distances = f_dist(PSamples - QSample, axis=1)
        return np.sum(distances <= QkDistance) >= k/C
        #i = 0
        #for x in range(len(PSamples)):
        #    if f_dist(QSample - PSamples[x]) <= QkDistance:
        #        i += 1
        #    if i >= k/C:
        #        return 1
        #return 0
    
    covered = sum(Parallel(n_jobs=nJobs)(delayed(worker)(PSamples, QSamples[y], QkDistances[y], hyperparam['k'], hyperparam['C']) for y in range(len(QSamples))))

    #covered = 0    
    #for y in range(len(QSamples)):
    #    i = 0
    #    for x in range(len(PSamples)):
    #        if f_dist(QSamples[y]-PSamples[x]) <= QkDistances[y]:
    #            i += 1
    #        if i >= hyperparam['k']:
    #            covered += 1
    #            break

    return covered/len(QSamples)
def recallCover(PSamples, QSamples, hyperparam, PkDistances = [], nJobs = 1, f_dist = np.linalg.norm):
    if len(PkDistances) == 0:
        PkDistances = kDistances(PSamples, hyperparam['k'], nJobs, f_dist)
    
    def worker(PSample, QSamples, PkDistance, k, C):
        distances = f_dist(QSamples - PSample, axis=1)
        return np.sum(distances <= PkDistance) >= k/C
        #i = 0
        #for x in range(len(QSamples)):
        #    if f_dist(PSample - QSamples[x]) <= PkDistance:
        #        i += 1
        #    if i >= k/C:
        #        return 1
        #return 0
    
    covered = sum(Parallel(n_jobs=nJobs)(delayed(worker)(PSamples[y], QSamples, PkDistances[y], hyperparam['k'], hyperparam['C']) for y in range(len(PSamples))))

    #covered = 0
    #for y in range(len(PSamples)):
    #    i = 0
    #    for x in range(len(QSamples)):
    #        if f_dist(PSamples[y]-QSamples[x]) <= PkDistances[y]:
    #            i += 1
    #        if i >= hyperparam['k']:
    #            covered += 1
    #            break

    return covered/len(PSamples)

def FBetaScore(precision, recall, beta):
    return (1+beta**2)*precision*recall/(beta**2*precision+recall)

# IMPROVED PRECISION-RECALL COVER

#As for precision-recall cover the two metrics are symmetric and therefore can be implemented as a single function.
#Nonetheless, we will implement them separately for clarity.
def improvedPrecisionCover(PSamples, QSamples, hyperparam, PkDistances = [], nJobs = 1, f_dist = np.linalg.norm):
    
    if len(PkDistances) == 0:
        PkDistances = kDistances(PSamples, hyperparam['k'], nJobs, f_dist)
    
    def worker(PSamples, QSample, PkDistances):
        distances = f_dist(QSample - PSamples, axis=1)
        return np.any(distances <= PkDistances)
        #for x in range(len(PSamples)):
        #    if f_dist(QSample - PSamples[x]) <= PkDistances[x]:
        #        return 1
        #return 0
    
    covered = sum(Parallel(n_jobs=nJobs)(delayed(worker)(PSamples, QSamples[y], PkDistances) for y in range(len(QSamples))))

    #covered = 0    
    #for y in range(len(QSamples)):
    #    for x in range(len(PSamples)):
    #        if f_dist(QSamples[y]-PSamples[x]) <= PkDistances[y]:
    #            covered += 1
    #            break

    return covered/len(QSamples)
def improvedRecallCover(PSamples, QSamples, hyperparam, QkDistances = [], nJobs = 1, f_dist = np.linalg.norm):
    
    if len(QkDistances) == 0:
        QkDistances = kDistances(QSamples, hyperparam['k'], nJobs, f_dist)
    
    def worker(PSample, QSamples, QkDistances):
        distances = f_dist(PSample - QSamples, axis=1)
        return np.any(distances <= QkDistances)        
        #for x in range(len(QSamples)):
        #    if f_dist(PSample - QSamples[x]) <= QkDistances[x]:
        #        return 1
        #return 0
    
    covered = sum(Parallel(n_jobs=nJobs)(delayed(worker)(PSamples[y], QSamples, QkDistances) for y in range(len(PSamples))))

    #covered = 0
    #for y in range(len(PSamples)):
    #    for x in range(len(QSamples)):
    #        if f_dist(PSamples[y]-QSamples[x]) <= QkDistances[y]:
    #            covered += 1
    #            break

    return covered/len(PSamples)

# DENSITY AND COVERAGE

def density(PSamples, QSamples, hyperparam, PkDistances = [], nJobs = 1, f_dist = np.linalg.norm):
    if len(PkDistances) == 0:
        PkDistances = kDistances(PSamples, hyperparam['k'], nJobs, f_dist)
    
    def worker(PSamples, QSample, PkDistances):
        distances = f_dist(QSample - PSamples, axis=1)
        return np.sum(distances <= PkDistances)
        #val = 0
        #for x in range(len(PSamples)):
        #    if f_dist(QSample - PSamples[x]) <= PkDistances[x]:
        #        val += 1
        #return val
    
    density = sum(Parallel(n_jobs=nJobs)(delayed(worker)(PSamples, QSamples[y], PkDistances) for y in range(len(QSamples))))

    #density = 0
    #for y in range(len(QSamples)):
    #    val = 0
    #    for x in range(len(PSamples)):
    #        if f_dist(QSamples[y]-PSamples[x]) <= PkDistances[y]:
    #            val += 1
    #    density += val

    return density/(len(QSamples)*hyperparam['k'])
def coverage(PSamples, QSamples, hyperparam, PkDistances = [], nJobs = 1, f_dist = np.linalg.norm):
    
    if len(PkDistances) == 0:
        PkDistances = kDistances(PSamples, hyperparam['k'], nJobs, f_dist)
    
    def worker(PSample, QSamples, PkDistance):
        distances = f_dist(PSample - QSamples, axis=1)
        return np.any(distances <= PkDistance)
        #for x in range(len(QSamples)):
        #    if f_dist(PSample - QSamples[x]) <= PkDistance:
        #        return 1
        #return 0

    coverage = sum(Parallel(n_jobs=nJobs)(delayed(worker)(PSamples[y], QSamples, PkDistances[y]) for y in range(len(PSamples))))

    #coverage = 0
    #for x in range(len(PSamples)):
    #    for y in range(len(QSamples)):
    #        if f_dist(PSamples[x]-QSamples[y]) <= PkDistances[y]:
    #            coverage += 1
    #            break

    return coverage/len(PSamples)

# PROBABILISTIC PRECISION-RECALL COVER

#Could be implemented as a single function by switching the samples.
def probPrecisionCover(PSamples, QSamples, hyperparam, nJobs = 1, f_dist = np.linalg.norm):

    def worker(PSamples, QSample, R):
        distances = f_dist(QSample - PSamples, axis=1)
        mask = distances >= R
        f = distances / R
        f[mask] = 1
        prod = np.prod(f)
        return 1 - prod
        #prod = 1
        #for x in range(len(QSamples)):
        #    if f_dist(PSample - QSamples[x]) < R:
        #        prod *= max(0,min(1,f_dist(PSample-QSamples[x])/R))
        #return 1-prod
    
    sum_ = sum(Parallel(n_jobs=nJobs)(delayed(worker)(PSamples, QSamples[y], hyperparam['Rp']) for y in range(len(QSamples))))

    #sum = 0
    #for y in range(len(QSamples)):
    #    prod = 1
    #    for x in range(len(PSamples)):
    #        if f_dist(QSamples[y]-PSamples[x]) < hyperparam['R']:
    #            prod *= max(0,min(1,f_dist(QSamples[y]-PSamples[x])/hyperparam['R']))
    #    sum += 1-prod

    return sum_/len(PSamples)
def probRecallCover(PSamples, QSamples, hyperparam, nJobs = 1, f_dist = np.linalg.norm):
    
    def worker(PSample, QSamples, R):
        distances = f_dist(PSample - QSamples, axis=1)
        mask = distances >= R
        f = distances / R
        f[mask] = 1
        prod = np.prod(f)
        return 1 - prod
    #    #prod = 1
    #    #for x in range(len(PSamples)):
    #    #    if f_dist(QSample - PSamples[x]) < R:
    #    #        prod *= max(0,min(1,f_dist(QSample-PSamples[x])/R))
    #    #return 1-prod
    #
    sum_ = sum(Parallel(n_jobs=nJobs)(delayed(worker)(PSamples[y], QSamples, hyperparam['Rq']) for y in range(len(PSamples))))

    #sum_ = 0
    #for y in range(len(PSamples)):
    #    prod = 1
    #    for x in range(len(QSamples)):
    #        if f_dist(PSamples[y]-QSamples[x]) < hyperparam['Rq']:
    #            prod *= max(0,min(1,f_dist(PSamples[y]-QSamples[x])/hyperparam['Rq']))
    #    sum_ += 1-prod

    return sum_/len(QSamples)


# ESTIMATED PRECISION-RECALL CURVE

def createDtrainDtest(PSamples, QSamples, p = 0.5):

    if len(PSamples) != len(QSamples):
        raise ValueError("The datasets must have the same size.")

    Dtrain = []
    Dtest = []

    for i in range(len(PSamples)):
        UTrainOrTest = np.random.binomial(1, p)
        UPorQ = np.random.binomial(1, 0.5)
        if UTrainOrTest == 1:
            Z = UPorQ*PSamples[i] + (1-UPorQ)*QSamples[i]
            Dtrain.append((tuple(Z), UPorQ))
        else:
            Z = UPorQ*PSamples[i] + (1-UPorQ)*QSamples[i]
            Dtest.append((tuple(Z), UPorQ))

    #for i in range(len(PSamples)):
    #    Ui = np.random.binomial(1, 0.5)
    #    Ztrain = Ui*PSamples[i] + (1-Ui)*QSamples[i]
    #    Ztest = (1-Ui)*PSamples[i] + Ui*QSamples[i]
    #    Dtrain.append((tuple(Ztrain), Ui))
    #    Dtest.append((tuple(Ztest), 1-Ui))

    return Dtrain, Dtest
def iprClassifier(Dtrain, hyperparam, f_dist = np.linalg.norm):

    PSamples = np.array([x for x, y in Dtrain if y == 1])
    QSamples = np.array([x for x, y in Dtrain if y == 0])

    k = hyperparam['k']
    nJobs = hyperparam['nJobs']

    PkDistances = kDistances(PSamples, k, nJobs, f_dist)
    QkDistances = kDistances(QSamples, k, nJobs, f_dist)

    #make a dictionary of sumP and sumQ for each x
    sumP = {}
    sumQ = {}

    def func(x, lambda_):    
        sumP_ = 0
        sumQ_ = 0

        if x in sumP and x in sumQ:
            if lambda_ < 1:
                return int( lambda_*sumP[x] > sumQ[x] )
            else:
                return int( lambda_*sumP[x] >= sumQ[x] )
        

        distancesP = f_dist(PSamples - x, axis=1)
        sumP_ += np.sum(distancesP <= PkDistances)

        distancesQ = f_dist(QSamples - x, axis=1)
        sumQ_ += np.sum(distancesQ <= QkDistances)  

        #for i in range(len(PSamples)):
        #    if f_dist(PSamples[i]-x) <= PkDistances[i]:
        #        sumP_ += 1
        
        #
        #for i in range(len(QSamples)):
        #    if f_dist(QSamples[i]-x) <= QkDistances[i]:
        #        sumQ_ += 1

        sumP[x] = sumP_
        sumQ[x] = sumQ_

        if lambda_ < 1:
            return int( lambda_*sumP_ > sumQ_ )
        else:
            return int( lambda_*sumP_ >= sumQ_ )
    return func
def covClassifier(Dtrain, hyperparam, f_dist = np.linalg.norm):

    PSamples = np.array([x for x, y in Dtrain if y == 1])
    QSamples = np.array([x for x, y in Dtrain if y == 0])

    sumP = {}
    sumQ = {}

    k = hyperparam['k']
    nJobs = hyperparam['nJobs']

    def func(x, lambda_):

        if x in sumP and x in sumQ:
            if lambda_ < 1:
                return int( lambda_*sumP[x] > sumQ[x] )
            else:
                return int( lambda_*sumP[x] >= sumQ[x] )

        # should return an error if the number of samples is less than k
        #if len(PSamples) < k or len(QSamples) < k:
        #    return 0
        
        #get the kth nearest neighbors distance of x in PSamples

        PSamples_ = np.vstack((PSamples, x))
        
        #TODO adapt for different distance functions
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=nJobs).fit(PSamples_)
        distances, _ = nbrs.kneighbors(PSamples_[-1].reshape(1, -1), n_neighbors=k+1)
        Pk = distances[0][-1]

        #get the kth nearest neighbors distance of x in QSamples

        QSamples_ = np.vstack((QSamples, x))

        #TODO adapt for different distance functions
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=nJobs).fit(QSamples_)
        distances, _ = nbrs.kneighbors(QSamples_[-1].reshape(1, -1), n_neighbors=k+1)
        Qk = distances[0][-1]

        sumP_ = 0
        sumQ_ = 0

        distancesP = f_dist(x - PSamples, axis=1)
        sumP_ += np.sum(distancesP <= Qk)

        distancesQ = f_dist(x - QSamples, axis=1)
        sumQ_ += np.sum(distancesQ <= Pk)

        sumP[x] = sumP_
        sumQ[x] = sumQ_

        if lambda_ < 1:
            return int( lambda_*sumP_ > sumQ_ )
        else:
            return int( lambda_*sumP_ >= sumQ_ )
    
    return func
def knnClassifier(Dtrain, hyperparam, f_dist = np.linalg.norm):
    PSamples = np.array([x for x, y in Dtrain if y == 1])
    QSamples = np.array([x for x, y in Dtrain if y == 0])
    samples = np.array([x for x, _ in Dtrain])

    sumP = {}
    sumQ = {}

    k = hyperparam['k']
    nJobs = hyperparam['nJobs']

    def func(x, lambda_):

        if x in sumP and x in sumQ:
            if lambda_ < 1:
                return int( lambda_*sumP[x] > sumQ[x] )
            else:
                return int( lambda_*sumP[x] >= sumQ[x] )

        samples_ = np.vstack((samples, x))

        #TODO adapt for different distance functions
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=nJobs).fit(samples_)
        distances, _ = nbrs.kneighbors(samples_[-1].reshape(1, -1), n_neighbors=k+1)
        kthSample = distances[0][-1]
        
        sumP_ = 0
        sumQ_ = 0


        distancesP = f_dist(x - PSamples, axis=1)
        sumP_ += np.sum(distancesP <= kthSample)

        distancesQ = f_dist(x - QSamples, axis=1)
        sumQ_ += np.sum(distancesQ <= kthSample)

        sumP[x] = sumP_
        sumQ[x] = sumQ_

        if lambda_ < 1:
            return int( lambda_*sumP_ > sumQ_ )
        else:
            return int( lambda_*sumP_ >= sumQ_ )
    
    return func
def parzenClassifier(Dtrain, hyperparam, f_dist = np.linalg.norm):
    PSamples = np.array([x for x, y in Dtrain if y == 1])
    QSamples = np.array([x for x, y in Dtrain if y == 0])
    
    pX = sum(kDistances(PSamples, hyperparam['k'], hyperparam['nJobs'], f_dist))/len(PSamples)
    qX = sum(kDistances(QSamples, hyperparam['k'], hyperparam['nJobs'], f_dist))/len(QSamples)

    sumP = {}
    sumQ = {}

    def func(x, lambda_):
        if x in sumP and x in sumQ:
            if lambda_ < 1:
                return int( lambda_*sumP[x] > sumQ[x] )
            else:
                return int( lambda_*sumP[x] >= sumQ[x] )
            
        sumP_ = 0
        sumQ_ = 0

        distancesP = f_dist(PSamples - x, axis=1)
        sumP_ += np.sum(distancesP <= pX)

        distancesQ = f_dist(QSamples - x, axis=1)
        sumQ_ += np.sum(distancesQ <= qX)

        sumP[x] = sumP_
        sumQ[x] = sumQ_

        if lambda_ < 1:
            return int( lambda_*sumP_ > sumQ_ )
        else:
            return int( lambda_*sumP_ >= sumQ_ )
    
    return func
def estimatePRD(func, Dtest, hyperparam, f_dist = np.linalg.norm):

    PRD = []
    errRates = []
        
    for g in hyperparam['Gamma']:
        N = [0, 0]
        fpr = 0
        fnr = 0

        for val, res in Dtest:
            fVal = func(val, g, f_dist)
            N[res] += 1
            
            if fVal == 1 and res == 0:
                fpr += 1
            if fVal == 0 and res == 1:
                fnr += 1

        fpr = fpr/N[0]
        fnr = fnr/N[1]
        errRates.append((fpr, fnr))

    for l in hyperparam['Lambda']:
        alpha_l = min(l*fpr + fnr for fpr, fnr in errRates)    
        #clip the alpha values between 0 and 1 and same for alpha/l
        PRD.append((max(0, min(1, alpha_l)), max(0, min(1, alpha_l/l))))

    return PRD
def estimatePRCurve(PSamples, QSamples, hyperparam, classifier, f_dist = np.linalg.norm):
    Dtrain, Dtest = createDtrainDtest(PSamples, QSamples)
    func = classifier(Dtrain, hyperparam)
    return estimatePRD(func, Dtest, hyperparam, f_dist)

