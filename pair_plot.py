from utils import filter_features, pair_plot
import numpy as np
import sys

def main() :
    if len(sys.argv) == 2 :
        try :
            arr = np.loadtxt(sys.argv[1], delimiter=",", dtype=str)[:, 1:]
        except :
            print("Invalid csv")
            print("Usage : python3 histogram.py [.csv]")
            return
        target = arr[1:, 0]
        column_names, features = filter_features(arr)
        pair_plot(features, column_names, target)
    else :
        print("Usage : python3 historgram.py [.csv]")

if "__main__" == __name__ :
    main()