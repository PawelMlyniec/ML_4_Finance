import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.metrics._regression import _check_reg_targets, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_consistent_length


def timeseries_train_test_split(X, y, test_fraction):
    """
    Split the timeseries by taking first points
    as train and last test_fraction * len(X) points as test
    """
    test_points = int(test_fraction * len(X))

    X_train = X.iloc[:-test_points]
    y_train = y.iloc[:-test_points]
    X_test = X.iloc[-test_points:]
    y_test = y.iloc[-test_points:]

    return X_train, X_test, y_train, y_test


def autoregressive_cross_validation(model, timeseries_split, X, y):
    fold_errors = []
    for fold, (train_idx, test_idx) in enumerate(timeseries_split.split(X)):
        X_train, y_train = X.values[train_idx], y.values[train_idx].reshape(-1, 1)
        X_test, y_test = X.values[test_idx], y.values[test_idx].reshape(-1, 1)

        scaler = MinMaxScaler().fit(np.hstack((X_train, y_train)).ravel().reshape(-1, 1))
        n, d = X_train.shape
        n_test = len(y_test)
        X_train = scaler.transform(X_train.ravel().reshape(-1, 1)).reshape((n, d))
        y_train = scaler.transform(y_train.reshape(-1, 1)).reshape((n,))
        y_test = scaler.transform(y_test.reshape(-1, 1)).reshape((n_test,))

        trained = model.fit(X_train, y_train)

        autoregressive_prediction = []
        history = X_test[0, :].tolist()
        history = scaler.transform(history.reshape(-1, 1)).tolist()
        test_mse, autoregressive_test_mse = 0, 0
        for i in range(len(X_test)):
            pred = trained.predict([history]).tolist()
            autoregressive_test_mse += (pred[0] - y_test[i]) ** 2
            autoregressive_prediction.append(pred[0])
            history = pred + history[:-1]

        autoregressive_test_rmse = np.sqrt(autoregressive_test_mse / len(autoregressive_prediction))
        normalized_autoregressive_test_rmse = autoregressive_test_rmse / np.mean(y_test)
        fold_errors.append(normalized_autoregressive_test_rmse)
    mean_cv = sum(fold_errors) / len(fold_errors)
    std_cv = np.std(fold_errors)
    return mean_cv, std_cv


def one_step_cross_validation(model, timeseries_split, X, y):
    scaler = MinMaxScaler()
    pipeline = Pipeline(steps=[('scaler', scaler), ('model', model)])
    cv = cross_val_score(pipeline, X.values, y.values,
                         cv=timeseries_split,
                         scoring=make_scorer(normalized_rmse))
    mean_cv = cv.mean()
    std_cv = cv.std()
    return mean_cv, std_cv


def autoregressive_cross_validation(model, timeseries_split, X, y, pairs, folds=5):
    fold_errors = []
    cv_generators = []
    for pair in pairs:
        cv_generators.append(timeseries_split.split(X[X.pair == pair]))
    for fold in range(folds):
        X_train, y_train, X_test_dfs, y_test_dfs = prepare_pairs_fold(cv_generators, pairs, X, y)

        scaler = MinMaxScaler().fit(np.hstack((X_train, y_train.reshape(-1, 1))).ravel().reshape(-1, 1))
        n, d = X_train.shape
        X_train = scaler.transform(X_train.ravel().reshape(-1, 1)).reshape((n, d))
        y_train = scaler.transform(y_train.reshape(-1, 1)).reshape((n,))

        trained = model.fit(X_train, y_train)
        pairs_errors = []
        for X_test, y_test in zip(X_test_dfs, y_test_dfs):
            autoregressive_prediction = []
            X_test, y_test = X_test.drop('pair', axis=1).values, y_test.drop('pair', axis=1).values
            history = X_test[0, :]
            y_test = scaler.transform(y_test.reshape(-1, 1))
            history = scaler.transform(history.reshape(-1, 1)).reshape((history.shape[0],)).tolist()
            test_mse, autoregressive_test_mse = 0, 0
            for i in range(len(X_test)):
                pred = trained.predict([history]).tolist()
                autoregressive_test_mse += (pred[0] - y_test[i]) ** 2
                autoregressive_prediction.append(pred[0])
                history = pred + history[:-1]
            autoregressive_test_rmse = np.sqrt(autoregressive_test_mse[0] / len(autoregressive_prediction))
            normalized_autoregressive_test_rmse = autoregressive_test_rmse / np.mean(y_test)
            pairs_errors.append(normalized_autoregressive_test_rmse)

        fold_errors.append(pairs_errors)
    fold_errors = np.array(fold_errors)
    return fold_errors


def one_step_cross_validation(model, timeseries_split, X, y, pairs, folds=5):
    cv_generators = []
    for pair in pairs:
        cv_generators.append(timeseries_split.split(X[X.pair == pair]))
    fold_errors = []
    for fold in range(folds):
        X_train, y_train, X_test_dfs, y_test_dfs = prepare_pairs_fold(cv_generators, pairs, X, y)

        scaler = MinMaxScaler().fit(np.hstack((X_train, y_train.reshape(-1, 1))).ravel().reshape(-1, 1))
        n, d = X_train.shape
        X_train = scaler.transform(X_train.ravel().reshape(-1, 1)).reshape((n, d))
        y_train = scaler.transform(y_train.reshape(-1, 1)).reshape((n,))

        trained = model.fit(X_train, y_train)
        pairs_errors = []
        for X_test, y_test in zip(X_test_dfs, y_test_dfs):
            X_test, y_test = X_test.drop('pair', axis=1).values, y_test.drop('pair', axis=1).values
            n, d = X_test.shape
            X_test = scaler.transform(X_test.reshape(-1, 1)).reshape((n, d))
            y_test = scaler.transform(y_test.reshape(-1, 1))
            preds = trained.predict(X_test)
            normalized_rmse = mean_squared_error(y_test, preds, squared=False) / np.mean(y_test)
            pairs_errors.append(normalized_rmse)
        fold_errors.append(pairs_errors)
    fold_errors = np.array(fold_errors)
    return fold_errors


def prepare_pairs_fold(cv_generators, pairs, X, y):
    trainX_dfs, testX_dfs = [], []
    trainy_dfs, testy_dfs = [], []
    for i, generator in enumerate(cv_generators):
        train_idx, test_idx = next(generator)
        trainX_dfs.append(X[X.pair == pairs[i]].iloc[train_idx])
        testX_dfs.append(X[X.pair == pairs[i]].iloc[test_idx])
        trainy_dfs.append(y[y.pair == pairs[i]].iloc[train_idx])
        testy_dfs.append(y[y.pair == pairs[i]].iloc[test_idx])
    X_train = pd.concat(trainX_dfs).drop('pair', axis=1).values
    y_train = pd.concat(trainy_dfs).price.values
    return X_train, y_train, testX_dfs, testy_dfs


def add_result(results, model_name, mean_cv, std_cv, result_type):
    results[result_type]['model'].append(model_name)
    results[result_type]['mean'].append(mean_cv)
    results[result_type]['stddev'].append(std_cv)
    return results


def normalized_rmse(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    check_consistent_length(y_true, y_pred, sample_weight)
    output_errors = np.average((y_true - y_pred) ** 2, axis=0, weights=sample_weight)
    output_errors = np.sqrt(output_errors)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    # normalization by the mean
    return np.average(output_errors, weights=multioutput) / np.mean(y_true)
