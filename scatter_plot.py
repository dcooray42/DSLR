from utils import filter_features, scatter_plot
import numpy as np
import sys

def main() :
    if len(sys.argv) == 2 :
        try :
            arr = np.loadtxt(sys.argv[1], delimiter=",", dtype=str)[:, 1:]
            target = arr[1:, 0]
            column_names, features = filter_features(arr)
            scatter_plot(features, column_names, target)
        except :
            print("Invalid csv")
            print("Usage : python3 scatter_plot.py [.csv]")
            return
    else :
        print("Usage : python3 scatter_plot.py [.csv]")

if "__main__" == __name__ :
    main()