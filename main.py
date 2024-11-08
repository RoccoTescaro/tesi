from tests import *

if __name__ == '__main__':
    testUnifyPrCurve(5001,1001,"sqrt",1)
    testCurve(iprClassifier,'ipr',5001,1001,"sqrt",3)
    testCurve(covClassifier,'cov',5001,1001,"sqrt",3)
    testCurve(knnClassifier,'knn',5001,1001,"sqrt",3)
    testCurve(parzenClassifier,'parzen',5001,1001,"sqrt",3)
    testUnifyPrCurve(5001,1001,"sqrt",3)
    testCurve(iprClassifier,'ipr',5001,1001,"4",3)
    testCurve(covClassifier,'cov',5001,1001,"4",3)
    testCurve(knnClassifier,'knn',5001,1001,"4",3)
    testCurve(parzenClassifier,'parzen',5001,1001,"4",3)
    testUnifyPrCurve(5001,1001,"4",3)