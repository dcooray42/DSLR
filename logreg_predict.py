from ml_utils import logreg_predict
from utils import filter_features
import numpy as np
import pickle
import sys

def main() :
    if len(sys.argv) == 3 :
        try :
            arr = np.loadtxt(sys.argv[1], delimiter=",", dtype=str)[:, 1:]
            with open(sys.argv[2], "rb") as f :
                data = pickle.load(f)
        except :
            print("Invalid csv or invalid pickle")
            print("Usage : python3 logreg_train.py [.csv] [.pkl]")
            return
        target = arr[1:, 0]
        column_names, features = filter_features(arr)
        logreg_predict(features, data)
    else :
        print("Usage : python3 logreg_train.py [.csv] [.pkl]")

if "__main__" == __name__ :
    main()