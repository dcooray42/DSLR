import matplotlib.pyplot as plt
import numpy as np
import sys

def histogram(features, column_names, target) :
    arr = np.append(target.reshape(-1, 1), features, axis=1)
    if arr[arr[:, 0] == ""].shape[1] == arr.shape[1] :
        print("The target column is empty.")
        return
    for index_col in range(arr.shape[1] - 1) :
        plt.figure()
        for house in sorted(set(target)) :
            data = arr[arr[:, 0] == house][:, 1:]
            current_col = np.array(data[:, index_col])
            mask = current_col == ""
            current_col = current_col[~mask]
            plt.hist(np.array(current_col).astype(float), bins=30, alpha=.5, label=house)
        plt.xlabel("Score")
        plt.ylabel("Percentage of student")
        plt.title(f"Histogram of the score of each house for the course {column_names[index_col]}")
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
        target = arr[1:, 0]
        column_names, features = filter_features(arr)
        histogram(features, column_names, target)
    else :
        print("Usage : python3 historgram.py [.csv]")

if "__main__" == __name__ :
    main()