import matplotlib.pyplot as plt
import numpy as np
import sys

def histogram(features, column_names) :
    features = ['Feature A', 'Feature B', 'Feature C', 'Feature D']
    counts = [10, 15, 7, 12]
    
    plt.bar(features, counts)
    plt.xlabel('Features')
    plt.ylabel('Counts')
    plt.title('Histogram with Feature Names')
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    plt.show()

def filter_features(features) :
    column_names = features[0]
    features = features[1:]
    numerical_data = np.array([])
    numerical_index = []
    for index_col in range(features.shape[1]) :
        try :
            current_col = np.array(features[:, index_col])
            mask = current_col == ""
            current_col = current_col[~mask]
            np.array(current_col).astype(float)
            if numerical_data.size == 0 :
                numerical_data = np.array(np.array(features[:, index_col])).reshape(-1, 1)
            else :
                numerical_data = np.append(numerical_data, np.array(np.array(features[:, index_col])).reshape(-1, 1), axis=1)
            numerical_index.append(index_col)
        except :
            pass
    return column_names[numerical_index], numerical_data

def main() :
    if len(sys.argv) == 2 :
        try :
            arr = np.loadtxt(sys.argv[1], delimiter=",", dtype=str)
        except :
            print("Invalid csv")
            print("Usage : python3 histogram.py [.csv]")
            return
        column_names, features = filter_features(arr)
        histogram(features, column_names)
    else :
        print("Usage : python3 historgram.py [.csv]")

if "__main__" == __name__ :
    main()