import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.evaluation_utils import timeseries_train_test_split


def plot_results(model, X_train, X_test, y_train, y_test, prediction, currencies, plot_intervals=False):
    """
    Plot the results

    """
    plt.figure(figsize=(15, 7))
    plt.axvline(X_test.index[0], color='k', linestyle='--')
    plt.plot(np.concatenate([X_train.index, X_test.index]), prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(np.concatenate([X_train.index, X_test.index]), np.concatenate([y_train.values, y_test.values]),
             label="actual", linewidth=2.0, alpha=0.8)

    if plot_intervals:
        ts_cv = TimeSeriesSplit(n_splits=5)
        cv = cross_val_score(model, np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)),
                             cv=ts_cv,
                             scoring="neg_mean_absolute_error")
        mean_cv = cv.mean() * (-1)
        std_cv = cv.std()

        scale = 1.96
        lower_bound = prediction - (mean_cv + scale * std_cv)
        upper_bound = prediction + (mean_cv + scale * std_cv)

        plt.plot(np.concatenate([X_train.index, X_test.index]), lower_bound, "r--", label="confidence intervals",
                 alpha=0.5)
        plt.plot(np.concatenate([X_train.index, X_test.index]), upper_bound, "r--", alpha=0.5)

    plt.title(f"Autoregresion for currency pair: {currencies}", fontsize=16, fontweight='bold')
    plt.legend(loc="best")
    plt.rcParams['axes.labelsize'] = 16
    plt.xticks(rotation=45)


def plot_last_fold_autoregression(model, X, y, pair):
    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_fraction=0.2)
    trained = model.fit(X_train, y_train)

    # rolling based prediction on test set
    autoregressive_prediction = []
    history = X_test.iloc[0].values.tolist()
    for i in range(len(X_test.index)):
        pred = trained.predict([history]).tolist()
        autoregressive_prediction.append(pred[0])
        history = pred + history[:-1]

    train_prediction = trained.predict(X_train.values)
    prediction = np.concatenate([train_prediction, autoregressive_prediction])
    plot_results(trained, X_train, X_test, y_train, y_test,
                 prediction, pair, plot_intervals=False)


def plot_comparison(results, pair):
    plt.rcParams['axes.labelsize'] = 10
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for idx, model_type in enumerate(results.keys()):
        res_ = results[model_type]
        axes[idx].bar(range(len(res_['model'])), height=res_['mean'], color='C0', yerr=res_['stddev'])
        if model_type == 'one-step':
            axes[idx].set_ylabel("")
            axes[idx].set_yscale("log")
        else:
            axes[idx].set_ylabel("5-fold cross validation average normalized RMSE")
            axes[idx].set_yscale("log")
        axes[idx].set_xticks(range(len(res_['model'])))
        axes[idx].set_title(f"{model_type} regime")
        axes[idx].set_xticklabels(res_['model'], rotation=90)
    fig.suptitle(pair)
    return fig
