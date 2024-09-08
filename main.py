from tests import *
from tools import *
from metrics import *
from papers.mgeval.core import extract_feature
import glob

if __name__ == "__main__":
    # paths to data
    PPaths = glob.glob('data/Scarlatti/real/**/*.mid', recursive=True)
    QPaths = glob.glob('data/Scarlatti/fake/**/*.mid', recursive=True)

    PSamples = np.array([extract_feature(p) for p in PPaths])
    QSamples = np.array([extract_feature(q) for q in QPaths])

    print('PSamples:', PSamples.shape)
    print('QSamples:', QSamples.shape)

    
