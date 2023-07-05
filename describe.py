from utils import TinyStatistician
import sys
import numpy as np

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