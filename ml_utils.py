import numpy as np
import pickle

class MyLogisticRegression():
    supported_penalities = ["l2"]

    def __init__(self, thetas, alpha=0.001, max_iter=1000, penality="l2", lambda_=1.0):
        if type(thetas).__module__ != np.__name__ or not isinstance(alpha, float) or not isinstance(max_iter, int) or (not isinstance(penality, str) and penality != None) or not isinstance(lambda_, float) :
            return None
        if thetas.size <= 0 :
            return None
        if alpha < 0 or alpha > 1 or max_iter < 0 :
            return None
        Thetas = thetas.squeeze().astype(float) if thetas.shape != (1, 1) else thetas.flatten().astype(float)
        if Thetas.ndim != 1 :
            return None
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = Thetas
        self.penality = penality
        self.lambda_ = lambda_ if penality in self.supported_penalities else 0.0

    def fit_(self, x, y) :

        def gradient(x, y, theta) :

            def add_intercept(x) :
                if type(x).__module__ != np.__name__ :
                    return None
                if x.size <= 0 :
                    return None
                X = np.copy(x).astype(float, copy=False)
                return (np.insert(X, 0, 1, axis=1)
                        if X.ndim != 1
                        else np.insert(np.transpose(np.expand_dims(X, axis=0)), 0, 1, axis=1))

            if type(x).__module__ != np.__name__ or type(y).__module__ != np.__name__ or type(theta).__module__ != np.__name__ :
                return None
            if x.size <= 0 or y.size <= 0 or theta.size <= 0 or not isinstance(self.lambda_, float) :
                return None
            X = add_intercept(x.astype(float))
            Y = np.squeeze(y).astype(float)
            Theta = np.squeeze(theta).astype(float)
            if Y.ndim != 1 or Theta.ndim != 1 or X.shape[0] != Y.shape[0] or X.shape[1] != Theta.shape[0] :
                return None
            Theta_prime = Theta.copy()
            Theta_prime[0] = 0
            return (X.T.dot((1 / (1 + np.exp(-X.dot(Theta)))) - Y) + self.lambda_ * Theta_prime).reshape(-1, 1) / Y.shape[0]

        if type(x).__module__ != np.__name__ or type(y).__module__ != np.__name__ or type(self.thetas).__module__ != np.__name__ :
            return None
        if x.size <= 0 or y.size <= 0 or self.thetas.size <= 0 :
            return None
        if self.alpha < 0 or self.alpha > 1 :
            return None
        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            return None
        X = x.astype(float)
        Y = np.squeeze(y).astype(float) if y.shape != (1, 1) else y.flatten().astype(float)
        Theta = np.squeeze(self.thetas).astype(float) if self.thetas.shape != (1, 1) else self.thetas.flatten().astype(float)
        if Y.ndim != 1 or Theta.ndim != 1 or X.shape[0] != Y.shape[0] or X.shape[1] + 1 != Theta.shape[0] :
            return None
        for _ in range(self.max_iter) :
            gradient_descent = gradient(X, Y, Theta)
            Theta -= self.alpha * gradient_descent.squeeze()
        self.thetas = Theta.reshape(-1, 1)
        return self.thetas
    
    def predict_(self, x) :

        def add_intercept(x) :
            if type(x).__module__ != np.__name__ :
                return None
            if x.size <= 0 :
                return None
            X = np.copy(x).astype(float, copy=False)
            return (np.insert(X, 0, 1, axis=1)
                    if X.ndim != 1
                    else np.insert(X.reshape(X.shape[0], 1), 0, 1, axis=1))

        if type(x).__module__ != np.__name__ or type(self.thetas).__module__ != np.__name__ :
            return None
        if x.size <= 0 or self.thetas.size <= 0 :
            return None
        X = add_intercept(x)
        Theta = np.squeeze(self.thetas).astype(float) if self.thetas.shape != (1, 1) else self.thetas.flatten().astype(float)
        X_comp = X.shape[1] if X.ndim != 1 else 2
        if Theta.ndim != 1 or X_comp != Theta.shape[0] :
            return None
        return (1 / (1 + np.exp(-X.dot(Theta)))).reshape(-1, 1)

    def loss_(self, y, y_hat) :

        def l2(theta) :
            if type(theta).__module__ != np.__name__ :
                return None
            if theta.size <= 0 :
                return None
            Theta = np.squeeze(theta).astype(float) if theta.shape != (1, 1) else theta.flatten().astype(float)
            if Theta.ndim != 1 :
                    return None
            Theta[0] = 0
            return Theta.dot(Theta)

        if type(y).__module__ != np.__name__ or type(y_hat).__module__ != np.__name__ or type(self.thetas).__module__ != np.__name__ :
            return None
        if y.size <= 0 or y_hat.size <= 0 or self.thetas.size <= 0 or not isinstance(self.lambda_, float) :
            return None
        Y = y.squeeze().astype(float) if (y.shape != (1, 1) and y.shape != (1,)) else y.flatten().astype(float)
        Y_hat = y_hat.squeeze().astype(float) if (y_hat.shape != (1, 1) and y_hat.shape != (1,)) else y_hat.flatten().astype(float)
        Theta = np.squeeze(self.thetas).astype(float) if self.thetas.shape != (1, 1) else self.thetas.flatten().astype(float)
        if Y.ndim != 1 or Y_hat.ndim != 1 or Theta.ndim != 1 or Y.shape != Y_hat.shape :
            return None
        vec_ones = np.ones(Y.shape[0]).reshape(Y.shape[0], 1)
        Y = Y.reshape(1, Y.shape[0])
        Y_hat[Y_hat == 0] = 1e-15
        Y_hat[Y_hat == 1] = 1 - 1e-15
        Y_hat = Y_hat.reshape(Y_hat.shape[0], 1)
        result = ((Y.dot(np.log(Y_hat)) + (vec_ones.reshape(1, vec_ones.shape[0]) - Y).dot(np.log(vec_ones - Y_hat))) / (-Y_hat.shape[0])) + ((self.lambda_ * l2(Theta)) / (2 * Y_hat.shape[0]))
        return np.squeeze(result)

def data_spliter(x, y, proportion) :
    if type(x).__module__ != np.__name__ or type(y).__module__ != np.__name__ :
        return None
    if x.size <= 0 or y.size <= 0 :
        return None
    if not isinstance(proportion, float) :
        return None
    if proportion < 0 or proportion > 1 :
        return None
    X = x.astype(float)
    Y = y.squeeze().astype(float) if y.shape != (1, 1) else y.flatten().astype(float)
    if Y.ndim != 1 or X.shape[0] != Y.shape[0] :
        return None
    r_indexes = np.arange(X.shape[0])
    np.random.shuffle(r_indexes)
    X = X[r_indexes]
    Y = Y[r_indexes]
    split_num = int(X.shape[0] * proportion)
    return (X[:split_num, :] if X.ndim != 1 else X[:split_num],
            X[split_num:, :] if X.ndim != 1 else X[split_num:],
            Y[:split_num].reshape(-1, 1),
            Y[split_num:].reshape(-1, 1))

def zscore(x) :   
    if type(x).__module__ != np.__name__ :
            return None
    if x.size <= 0 :
        return None
    X = np.squeeze(x).astype(float)
    if X.ndim != 1 :
        return None
    return (X - np.mean(X)) / np.std(X)

def zscore_ori(x, x_ori) :   
    if type(x).__module__ != np.__name__ or type(x_ori).__module__ != np.__name__ :
        return None
    if x.size <= 0 or x_ori.size <= 0 :
        return None
    X = np.squeeze(x).astype(float)
    X_ori = x_ori.astype(float)
    if X.ndim != 1 or X_ori.ndim != 1 :
        return None
    return (X - np.mean(X_ori)) / np.std(X_ori)

def precision_score_(y, y_hat, pos_label=1) :
    if type(y).__module__ != np.__name__ or type(y_hat).__module__ != np.__name__ :
        return None
    if y.size <= 0 or y_hat.size <= 0 :
        return None
    Y = np.squeeze(y) if y.shape != (1, 1) else y.flatten()
    Y_hat = np.squeeze(y_hat) if y_hat.shape != (1, 1) else y_hat.flatten()
    if Y.ndim != 1 or Y_hat.ndim != 1 or Y.shape != Y_hat.shape :
        return None
    return ((sum((Y == Y_hat) & (Y == pos_label))) / sum(Y_hat == pos_label)).squeeze()

def recall_score_(y, y_hat, pos_label=1) :
    if type(y).__module__ != np.__name__ or type(y_hat).__module__ != np.__name__ :
        return None
    if y.size <= 0 or y_hat.size <= 0 :
        return None
    Y = np.squeeze(y) if y.shape != (1, 1) else y.flatten()
    Y_hat = np.squeeze(y_hat) if y_hat.shape != (1, 1) else y_hat.flatten()
    if Y.ndim != 1 or Y_hat.ndim != 1 or Y.shape != Y_hat.shape :
        return None
    return ((sum((Y == Y_hat) & (Y == pos_label))) / sum(Y == pos_label)).squeeze()

def f1_score_(y, y_hat, pos_label=1) :
    if type(y).__module__ != np.__name__ or type(y_hat).__module__ != np.__name__ :
        return None
    if y.size <= 0 or y_hat.size <= 0 :
        return None
    Y = np.squeeze(y) if y.shape != (1, 1) else y.flatten()
    Y_hat = np.squeeze(y_hat) if y_hat.shape != (1, 1) else y_hat.flatten()
    if Y.ndim != 1 or Y_hat.ndim != 1 or Y.shape != Y_hat.shape :
        return None
    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)
    return (2 * precision * recall) / (precision + recall)

def logreg_train(features, target) :
    column_index = [1, 2, 3, 6, 11]
    arr = np.append(target.reshape(-1, 1),
                    np.where(features == "", "0", features)[:, column_index].astype(float),
                    axis=1)
    if arr[arr[:, 0] == ""].shape[0] == arr.shape[0] :
        print("The target column is empty.")
        return
    features = arr[:, 1:]
    target = arr[:, 0]
    unique_target_values = sorted(set(target))
    data = {
        "unique_target_values" : unique_target_values,
        "zscore_features" : arr[:, 1:].copy(),
        "column_selected" : column_index
    }
    for index_col in range(features.shape[1]) :
        features[:, index_col] = zscore(features[:, index_col])
    for index, value in enumerate(unique_target_values) :
        target[target[:] == value] = index
    target = target.astype(float)
    x_train, x_test, y_train, y_test = data_spliter(features, target, 0.8)
    y_train = np.c_[y_train, np.ones(y_train.shape[0])]
    y_length = y_test.shape[0]
    for target_value in range(len(unique_target_values)) :
        data[target_value] = {
            "lambda" : 0.0,
            "f1_score" : 0.0,
            "thetas" : []
        }
    for lambda_ in np.linspace(0, 2000, 6) :
        lambda_ = round(float(lambda_), 2)
        y_predict = np.c_[np.zeros(y_length), np.zeros(y_length)]
        save_thetas = {}
        for target_value in range(len(unique_target_values)) :
            lr = MyLogisticRegression(np.zeros(len(column_index) + 1), alpha=1e-1, max_iter=500, lambda_=lambda_)
            y_train[:, 1] = np.ones(y_train.shape[0])
            y_train[y_train[:, 0] != target_value, 1] = 0
            save_thetas[target_value] = lr.fit_(x_train, y_train[:, 1])
            y_hat = lr.predict_(x_test).flatten()
            y_predict[y_predict[:, 1] < y_hat, 0] = target_value
            y_predict[y_predict[:, 1] < y_hat, 1] = y_hat[y_predict[:, 1] < y_hat]
        for target_value in range(len(unique_target_values)) :
            f1_score = f1_score_(y_test, y_predict[:, 0], target_value)
            print(lambda_, f1_score)
            if f1_score > data[target_value]["f1_score"] :
                data[target_value]["lambda"] = lambda_
                data[target_value]["f1_score"] = f1_score
                data[target_value]["thetas"] = save_thetas[target_value]
    target = np.c_[target, np.ones(target.shape[0])]
    for target_value in range(len(unique_target_values)) :
        lambda_ = data[target_value]["lambda"]
        lr = MyLogisticRegression(np.zeros(len(column_index) + 1), alpha=1e-1, max_iter=500, lambda_=lambda_)
        target[:, 1] = np.ones(target.shape[0])
        target[target[:, 0] != target_value, 1] = 0
        data[target_value]["thetas"] = lr.fit_(features, target[:, 1])
    print(data)

    with open("models.pkl", "wb") as f:
        pickle.dump(data, f)

def logreg_predict(features, target, data) :
    column_index = data["column_selected"]
    arr = np.append(target.reshape(-1, 1),
                    np.where(features == "", "0", features)[:, column_index].astype(float),
                    axis=1)
    if arr[arr[:, 0] == ""].shape[0] != arr.shape[0] :
        print("The target column isn't empty.")
        return
    features = arr[:, 1:]
    y_length = arr[:, 0].shape[0]
    if features.shape[1] != len(data["column_selected"]) :
        print("The original features passed in the pickle file have a different x_shape than the current features.")
        return
    for index_col in range(features.shape[1]) :
        features[:, index_col] = zscore_ori(features[:, index_col], data["zscore_features"][:, index_col])
    unique_target_values = data["unique_target_values"]
    y_predict = np.c_[np.zeros(y_length), np.zeros(y_length)]
    for target_value in range(len(unique_target_values)) :
        lr = MyLogisticRegression(data[target_value]["thetas"],
                                  lambda_=data[target_value]["lambda"])
        y_hat = lr.predict_(features).flatten()
        y_predict[y_predict[:, 1] < y_hat, 0] = target_value
        y_predict[y_predict[:, 1] < y_hat, 1] = y_hat[y_predict[:, 1] < y_hat]
    final_string = "Index,Hogwarts House\n"
    for index in range(y_predict.shape[0]) :
        final_string += f"{int(index)},{unique_target_values[int(y_predict[int(index), 0])]}"
        if index != y_predict.shape[0] - 1 :
            final_string += "\n"
    
    with open("houses.csv", "w") as f :
        f.write(final_string)