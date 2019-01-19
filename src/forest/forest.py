from IPython.display import display
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype
from sklearn.ensemble import RandomForestRegressor

from fastai.imports import *
from fastai.structured import *

# from fastai import train_cats


def get_raw_data():
    obj_file = Path('../../data/bulldozer/df_raw.obj')
    if obj_file.is_file():
        df_raw_file = open('../../data/bulldozer/df_raw.obj', 'rb')
        df_raw = pickle.load(df_raw_file)
    else:
        df_raw = pd.read_csv('../../data/bulldozer/train.csv', low_memory=False,
                             parse_dates=["saledate"])
        train_cats(df_raw)
        df_raw.saledate = pd.to_timedelta(df_raw.saledate).dt.total_seconds().astype(int)
        save_df_object(df_raw)

    return df_raw


def save_df_object(df_raw):
    file_pi = open('../../data/bulldozer/df_raw.obj', 'wb')
    pickle.dump(df_raw, file_pi)


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)


def display_head(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        print(df.head())


def split_vals(a,n): return a[:n].copy(), a[n:].copy()


if __name__ == "__main__":

    df_raw = get_raw_data()

    display_head(df_raw)
    print(df_raw.info())

    df, y, nas = proc_df(df_raw, 'SalePrice')

    n_valid = 12000  # same as Kaggle's test set size
    n_trn = len(df) - n_valid
    raw_train, raw_valid = split_vals(df_raw, n_trn)
    X_train, X_valid = split_vals(df, n_trn)
    y_train, y_valid = split_vals(y, n_trn)

    # set_rf_samples(20000)

    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5,
                              n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)

    # train score > valid score -> overfitting
    print(m.score(X_train, y_train))
    print(m.score(X_valid, y_valid))
    print(m.oob_score_)
