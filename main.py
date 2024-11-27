from lookout import *
from tests import *

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

models = ['model_011809.ckpt', 'model_516209.ckpt', 'model_2077006.ckpt', 'model_7083228.ckpt', 'model_7969400.ckpt']

if __name__ == '__main__':
    testScarlattiSupp()
    #testScarlattiLOOCV()
    updateIndexHtml('./images/realworldexperiments/scarlatti/kde/examples/data.json',
                    [m.__name__ for m in metrics],
                    models,
                    'data/supp_examples_matrix.npy')#'data/test_matrix_inter.npy') #'data/matrix_inter.npy') 
