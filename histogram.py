from utils import filter_features, histogram
import numpy as np
import sys

def main() :
    if len(sys.argv) == 2 :
        try :
            arr = np.loadtxt(sys.argv[1], delimiter=",", dtype=str)[:, 1:]
            target = arr[1:, 0]
            column_names, features = filter_features(arr)
            histogram(features, column_names, target)
        except :
            print("Invalid csv")
            print("Usage : python3 histogram.py [.csv]")
            return
    else :
        print("Usage : python3 historgram.py [.csv]")

if "__main__" == __name__ :
    main()