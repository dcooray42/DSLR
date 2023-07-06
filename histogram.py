import matplotlib.pyplot as plt
import numpy as np
import sys
from utils import TinyStatistician

def histogram(features, column_names, target) :
    ts = TinyStatistician()
    x = np.arange(1, len(column_names) + 1)
    arr = np.append(target.reshape(-1, 1), features, axis=1)
    indent = [-0.3, -0.1, 0.1, 0.3]
    for index, house in enumerate(sorted(set(target))) :
        data = arr[arr[:, 0] == house][:, 1:]
        mean_arr = []
        for index_col in range(data.shape[1]) :
            current_col = np.array(data[:, index_col])
            mask = current_col == ""
            current_col = current_col[~mask]
            mean_arr.append(ts.mean(current_col))
        plt.bar(x + indent[index], np.array(np.log10(mean_arr)), width=0.2, label=house)
    plt.xlabel("Course")
    plt.ylabel("Mean")
    plt.yscale("log")
    plt.title("Histogram of the mean of each course by houses")
    plt.xticks(x, column_names, rotation=20)
    plt.legend(loc="upper right", title="Houses")
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
            arr = np.loadtxt(sys.argv[1], delimiter=",", dtype=str)[:, 1:]
        except :
            print("Invalid csv")
            print("Usage : python3 histogram.py [.csv]")
            return
        print(arr.shape)
        target = arr[1:, 0]
        print(target[:10])
        column_names, features = filter_features(arr)
        histogram(features, column_names, target)
    else :
        print("Usage : python3 historgram.py [.csv]")

if "__main__" == __name__ :
    main()