from ml_utils import logreg_train
from utils import filter_features
import numpy as np
import sys

def main() :
    if len(sys.argv) == 3 :
        try :
            arr = np.loadtxt(sys.argv[2], delimiter=",", dtype=str)[:, 1:]
            batch_size = int(sys.argv[1])
            target = arr[1:, 0]
            column_names, features = filter_features(arr)
            logreg_train(features, target, batch_size)
        except :
            print("Invalid csv or invalid batch_size")
            print("Usage : python3 logreg_train.py [batch_size] [.csv]")
            return
    else :
        print("Usage : python3 logreg_train.py [batch_size] [.csv]")

if "__main__" == __name__ :
    main()