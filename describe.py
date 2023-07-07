from utils import filter_features, describe
import sys
import numpy as np
    
def main() :
    if len(sys.argv) == 2 :
        try :
            arr = np.loadtxt(sys.argv[1], delimiter=",", dtype=str)
        except :
            print("Invalid csv")
            print("Usage : python3 describe.py [.csv]")
            return
        column_names, features = filter_features(arr)
        describe(features, column_names)
    else :
        print("Usage : python3 describe.py [.csv]")

if "__main__" == __name__ :
    main()