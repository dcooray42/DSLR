from ml_utils import logreg_train
from utils import filter_features
import numpy as np
import sys

def main() :
    if len(sys.argv) == 2 :
        try :
            arr = np.loadtxt(sys.argv[1], delimiter=",", dtype=str)[:, 1:]
        except :
            print("Invalid csv")
            print("Usage : python3 logreg_train.py [.csv]")
            return
        target = arr[1:, 0]
        column_names, features = filter_features(arr)
        logreg_train(features, target)
    else :
        print("Usage : python3 logreg_train.py [.csv]")

if "__main__" == __name__ :
    main()