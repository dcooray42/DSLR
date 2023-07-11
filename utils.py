from math import ceil, floor
from matplotlib import pyplot as plt
import numpy as np

class TinyStatistician :
    def __add_fractional_to_base(self, array, p) :
        array = sorted(array)
        rank = (p * (len(array) - 1)) / 100
        integer_lower_rank = floor(rank)
        integer_higher_rank = ceil(rank)
        fractional_rank = rank - integer_lower_rank
        if integer_lower_rank == rank :
            return array[integer_lower_rank]
        return float(array[integer_lower_rank] + ((array[integer_higher_rank] - array[integer_lower_rank]) * fractional_rank))

    def mean(self, array) :
        if type(array).__module__ != np.__name__ :
                return None
        if array.size <= 0 :
            return None
        array = np.squeeze(array).astype(float) if array.shape != (1, 1) else array.flatten().astype(float)
        result = 0
        for _, value in enumerate(array) :
            result += value
        return float(result / len(array))

    def median(self, array) :
        if type(array).__module__ != np.__name__ :
                return None
        if array.size <= 0 :
            return None
        array = np.squeeze(array).astype(float) if array.shape != (1, 1) else array.flatten().astype(float)
        return float(self.__add_fractional_to_base(array, 50))

    def quartile(self, array) :
        if type(array).__module__ != np.__name__ :
                return None
        if array.size <= 0 :
            return None
        array = np.squeeze(array).astype(float) if array.shape != (1, 1) else array.flatten().astype(float)
        array.sort()
        return [float(self.__add_fractional_to_base(array, 25)), float(self.__add_fractional_to_base(array, 75))]

    def percentile(self, array, p) :
        if type(array).__module__ != np.__name__ :
                return None
        if array.size <= 0 :
            return None
        array = np.squeeze(array).astype(float) if array.shape != (1, 1) else array.flatten().astype(float)
        return float(self.__add_fractional_to_base(array, p))
 
    def var(self, array) :
        if type(array).__module__ != np.__name__ :
                return None
        if array.size <= 0 :
            return None
        array = np.squeeze(array).astype(float) if array.shape != (1, 1) else array.flatten().astype(float)
        mean = float(self.mean(array))
        total_sum = 0
        for value in array :
            total_sum += (float(value) - mean) ** 2
        return float(total_sum / len(array))
    
    def std(self, array) :
        if type(array).__module__ != np.__name__ :
                return None
        if array.size <= 0 :
            return None
        array = np.squeeze(array).astype(float) if array.shape != (1, 1) else array.flatten().astype(float)
        return float(self.var(array) ** (1 / 2))
    
    def count(self, array) :
        if type(array).__module__ != np.__name__ :
                return None
        if array.size <= 0 :
            return None
        array = np.squeeze(array).astype(float) if array.shape != (1, 1) else array.flatten().astype(float)
        cnt = 0
        for _ in range(len(array)) :
            cnt += 1
        return float(cnt)
    
    def min(self, array) :
        if type(array).__module__ != np.__name__ :
                return None
        if array.size <= 0 :
            return None
        array = np.squeeze(array).astype(float) if array.shape != (1, 1) else array.flatten().astype(float)
        min_val = float("+inf")
        for val in array :
            if val < min_val :
                min_val = val
        return float(min_val)
    
    def max(self, array) :
        if type(array).__module__ != np.__name__ :
                return None
        if array.size <= 0 :
            return None
        array = np.squeeze(array).astype(float) if array.shape != (1, 1) else array.flatten().astype(float)
        max_val = float("-inf")
        for val in array :
            if val > max_val :
                max_val = val
        return float(max_val)
    
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
            if current_col.size :
                if numerical_data.size == 0 :
                    numerical_data = np.array(np.array(features[:, index_col])).reshape(-1, 1)
                else :
                    numerical_data = np.append(numerical_data, np.array(np.array(features[:, index_col])).reshape(-1, 1), axis=1)
                numerical_index.append(index_col)
        except :
            pass
    return column_names[numerical_index], numerical_data

def create_matrix_string(matrix, column_names) :
    matrix = np.round(matrix, decimals=2)
    left_column = [
        "Count",
        "Mean",
        "Var",
        "Std",
        "Min",
        "25%",
        "50%",
        "75%",
        "Max"
    ]
    left_column_max_len = len(max(left_column, key=len))
    matrix_string = ""
    matrix_size = np.zeros(matrix.shape[1]).astype(int)
    matrix_max_len = np.max(np.char.str_len(matrix.astype(str)), axis=0)
    for index in range(matrix_size.shape[0]) :
        matrix_size[index] = max(len(column_names[index]), matrix_max_len[index])
    for index_row in range(matrix.shape[0] + 1) :
        if index_row == 0 :
            matrix_string += "".ljust(left_column_max_len) + "".join([value.rjust(matrix_size[index] + 2) for index, value in enumerate(column_names)])
        else :
            matrix_string += left_column[index_row - 1].ljust(left_column_max_len) + "".join([str(value).rjust(matrix_size[index] + 2) for index, value in enumerate(matrix[index_row - 1, :])])
        if index_row != matrix.shape[0]:
            matrix_string += "\n"
    return matrix_string
    
def describe(features, column_names) :
    ts = TinyStatistician()
    function_list = [
        ts.count,
        ts.mean,
        ts.var,
        ts.std,
        ts.min,
        ts.quartile,
        ts.median,
        ts.max
    ]
    matrix_values = np.array([])
    for index_col in range(features.shape[1]) :
        stat_values = []
        current_col = np.array(features[:, index_col])
        mask = current_col == ""
        current_col = current_col[~mask]
        for func in function_list :
            result = func(current_col.astype(float))
            if isinstance(result, float) or isinstance(result, int) :
                stat_values.append(result)
            else :
                for sub_result in result :
                    stat_values.append(sub_result)
        stat_values = np.array(stat_values).reshape(-1, 1)
        if matrix_values.size == 0 :
            matrix_values = stat_values
        else :
            matrix_values = np.append(matrix_values, stat_values, axis=1)
    matrix_values[-2], matrix_values[-3] = matrix_values[-3].copy(), matrix_values[-2].copy()
    print(create_matrix_string(matrix_values, column_names))

def histogram(features, column_names, target) :
    try :
        index_col = list(column_names).index("Care of Magical Creatures")
    except :
        print("The attended column isn't present in the dataset.")
        return
    arr = np.append(target.reshape(-1, 1), features, axis=1)
    if arr[arr[:, 0] == ""].shape[0] == arr.shape[0] :
        print("The target column is empty.")
        return
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

def scatter_plot(features, column_names, target) :
    try :
        index_col_one = list(column_names).index("Astronomy")
        index_col_two = list(column_names).index("Defense Against the Dark Arts")
    except :
        print("The attended column isn't present in the dataset.")
        return
    arr = np.append(target.reshape(-1, 1), features, axis=1)
    if arr[arr[:, 0] == ""].shape[0] == arr.shape[0] :
        print("The target column is empty.")
        return
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

def pair_plot(features, column_names, target) :
    arr = np.append(target.reshape(-1, 1), features, axis=1)
    if arr[arr[:, 0] == ""].shape[0] == arr.shape[0] :
        print("The target column is empty.")
        return
    column_index = [1, 2, 3, 6, 11]
    fig, axs = plt.subplots(len(column_index) + 1,
                            len(column_index),
                            gridspec_kw={
                                "height_ratios": [1 for _ in range(len(column_index))] + [0.2]
                            }
                            )
    for y_index, y_col_index in enumerate(column_index):
        axs[y_index, 0].set_ylabel(column_names[y_col_index], rotation=80)
    for x_index, x_col_index in enumerate(column_index):
        axs[len(column_index)-1, x_index].set_xlabel(column_names[x_col_index])
    for y_index, y_col_index in enumerate(column_index) :
        for x_index, x_col_index in enumerate(column_index) :
            if y_index == x_index :
                for house in sorted(set(target)) :
                    data = arr[arr[:, 0] == house][:, 1:]
                    current_col = np.array(data[:, x_col_index])
                    current_col = np.where(current_col == "", np.nan, current_col)
                    axs[y_index, x_index].hist(np.array(current_col).astype(float), bins=30, alpha=.5, label=house)
            else :
                for house in sorted(set(target)) :
                    data = arr[arr[:, 0] == house][:, 1:]
                    current_col_one = np.array(data[:, x_col_index])
                    current_col_two = np.array(data[:, y_col_index])
                    current_col_one = np.where(current_col_one == "", np.nan, current_col_one)
                    current_col_two = np.where(current_col_two == "", np.nan, current_col_two)
                    axs[y_index, x_index].scatter(
                        np.array(current_col_one).astype(float),
                        np.array(current_col_two).astype(float),
                        alpha=.5,
                        label=house)
    fig.text(0.5, 0.04, "Features", ha="center", va="center")
    fig.text(0.03, 0.5, "Features", ha="center", va="center", rotation="vertical")
    fig.suptitle("Pair plots of the features used for the logistic regression", fontsize=30)
    legend_ax = axs[-1]
    for graph in legend_ax :
        graph.axis("off")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    legend_ax[int(len(column_index) / 2)].legend(handles, labels, loc="center", ncol=4, title="Houses", bbox_to_anchor=(0.5, -3.0))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.9, wspace=0.3, hspace=0.3)
    plt.show()
    plt.close()