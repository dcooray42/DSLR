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
        np.size,
        ts.mean,
        ts.var,
        ts.std,
        np.min,
        ts.quartile,
        ts.median,
        np.max
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
    arr = np.append(target.reshape(-1, 1), features, axis=1)
    if arr[arr[:, 0] == ""].shape[0] == arr.shape[0] :
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

def pair_plot(features, column_names, target) :
    fig, axs = plt.subplots(features.shape[1], features.shape[1], gridspec_kw={"height_ratios": [1 for _ in range(features.shape[1] - 1)] + [0.2]})
    plt.show()
#    arr = np.append(target.reshape(-1, 1), features, axis=1)
#    if arr[arr[:, 0] == ""].shape[0] == arr.shape[0] :
#        print("The target column is empty.")
#        return
#    for index_col_one in range(arr.shape[1] - 1) :
#        for index_col_two in range(index_col_one + 1, arr.shape[1] - 1) :
#            plt.figure()
#            for house in sorted(set(target)) :
#                data = arr[arr[:, 0] == house][:, 1:]
#                current_col_one = np.array(data[:, index_col_one])
#                current_col_two = np.array(data[:, index_col_two])
#                current_col_one = np.where(current_col_one == "", np.nan, current_col_one)
#                current_col_two = np.where(current_col_two == "", np.nan, current_col_two)
#                plt.scatter(
#                    np.array(current_col_one).astype(float),
#                    np.array(current_col_two).astype(float),
#                    alpha=.5,
#                    label=house)
#            plt.xlabel(column_names[index_col_one])
#            plt.ylabel(column_names[index_col_two])
#            plt.title(f"Scatter ploat of each house for {column_names[index_col_two]} in function of {column_names[index_col_one]}")
#            plt.legend(loc="upper right", title="Houses")
#            plt.show()
#
#    fig, axs = plt.subplots(features.shape[1], features.shape[1], gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.2]})
#    for lambda_, lambda_dict in y.items() :
#            for feature, y_values in lambda_dict.items() :
#                y_values = np.array(y_values)
#                axs[int(feature)].scatter(
#                    x_values,
#                    y_values,
#                    label=str(round(float(lambda_), 2))
#                )
#    for ax in axs[:-1].flatten():
#        ax.grid(True)
#    fig.text(0.5, 0.04, "Algorithm", ha="center", va="center")
#    fig.text(0.06, 0.5, "F1 score", ha="center", va="center", rotation="vertical")
#    legend_ax = axs[-1]
#    handles, labels = axs[0].get_legend_handles_labels()
#    legend_ax.axis("off")
#    legend_ax.legend(handles, labels, loc="center", ncol=3, title="Lambda")
#    plt.subplots_adjust(hspace=0.4)
#    plt.show()
#    plt.close()