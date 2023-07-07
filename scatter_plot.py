import matplotlib.pyplot as plt
import numpy as np
import sys

def scatter_plot(features, column_names, target) :
    arr = np.append(target.reshape(-1, 1), features, axis=1)
    if arr[arr[:, 0] == ""].shape[0] == arr.shape[0] :
        print("The target column is empty.")
        return
    for index_col_one in range(arr.shape[1] - 1) :
        for index_col_two in range(index_col_one + 1, arr.shape[1] - 1) :
            plt.figure()
            for house in sorted(set(target)) :
                data = arr[arr[:, 0] == house][:, 1:]
                current_col_one = np.array(data[:, index_col_one])
                current_col_two = np.array(data[:, index_col_two])
                current_col_one = np.where(current_col_one == "", np.nan, current_col_one)
                current_col_two = np.where(current_col_two == "", np.nan, current_col_two)
                plt.scatter(
                    np.array(current_col_one).astype(float),
                    np.array(current_col_two).astype(float),
                    alpha=.5,
                    label=house)
            plt.xlabel(column_names[index_col_one])
            plt.ylabel(column_names[index_col_two])
            plt.title(f"Scatter ploat of each house for {column_names[index_col_two]} in function of {column_names[index_col_one]}")
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
        scatter_plot(features, column_names, target)
    else :
        print("Usage : python3 historgram.py [.csv]")

if "__main__" == __name__ :
    main()