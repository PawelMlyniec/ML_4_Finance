import os

import argparse
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from src.evaluation_utils import *
from src.models import LastValueRegressor


def cross_validate_regimes(pair_results, model, model_name, folds):
    timeseries_split = TimeSeriesSplit(n_splits=folds)
    mean_cv, std_cv = autoregressive_cross_validation(model, timeseries_split, X, y)
    pair_results = add_result(pair_results, model_name, mean_cv, std_cv, 'autoregressive')
    mean_cv, std_cv = one_step_cross_validation(model, timeseries_split, X, y)
    pair_results = add_result(pair_results, model_name, mean_cv, std_cv, 'one-step')
    return pair_results


def prepare_data(data_dir, filename, win_size):
    df = pd.read_csv(os.path.join(data_dir, filename))
    df.timestamp = pd.to_datetime(df.timestamp, format='%Y-%m-%dT%H:%M:%S.000Z')
    df = df.set_index("timestamp").sort_index()
    df1 = df.copy()
    for i in range(1, win_size):
        df1[f"price_{i}"] = df.price.shift(i)
    X = df1.dropna()[[f"price_{i}" for i in range(1, win_size)]]
    y = df1.dropna().price
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate conversation samples from accomoji dataset')

    # File parameters
    parser.add_argument('--sushiswap-dir', required=True,
                        help='Path to the directory with sushiswap transaction csvs')
    parser.add_argument('--output-dir', required=True,
                        help='Path to the output directory with results of baseline models')
    parser.add_argument('--window-size', default=7, required=False, type=int,
                        help='Window size for forecasting, default: 7')
    parser.add_argument('--folds', default=5, required=False, type=int,
                        help='Number of folds to use with cross-validation, default: 7')

    args = parser.parse_args()
    results = {}
    for filename in tqdm(os.listdir(args.sushiswap_dir)):
        X, y = prepare_data(args.sushiswap_dir, filename, args.window_size)
        pair = filename[:-4]
        pair_results = {"autoregressive": {'model': [], 'mean': [], 'stddev': []},
                        "one-step": {'model': [], 'mean': [], 'stddev': []}}
        dummy_regressor = LastValueRegressor(win_size=args.window_size)
        pair_results = cross_validate_regimes(pair_results, model=dummy_regressor,
                                              model_name='dummy regressor', folds=args.folds)
        lr = LinearRegression()
        pair_results = cross_validate_regimes(pair_results, model=lr, model_name='linear regression', folds=args.folds)
        results[pair] = pair_results

    with open(os.path.join(args.output_dir, "coin_pair_results.pkl"), 'wb') as f:
        pickle.dump(results, f)
