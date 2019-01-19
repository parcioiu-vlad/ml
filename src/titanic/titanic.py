import pickle
from pathlib import Path

import pandas as pd
from fastai.imports import *
from fastai.structured import *
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def get_raw_data():
    obj_file = Path('../../data/titanic/df_raw.obj')
    if obj_file.is_file():
        df_raw_file = open('../../data/titanic/df_raw.obj', 'rb')
        df_raw = pickle.load(df_raw_file)
    else:
        df_raw = pd.read_csv('../../data/titanic/train.csv', low_memory=False)
        save_df_object(df_raw)

    return df_raw


def save_df_object(df_raw):
    file_pi = open('../../data/titanic/df_raw.obj', 'wb')
    pickle.dump(df_raw, file_pi)


def split_vals(a,n): return a[:n].copy(), a[n:].copy()


if __name__ == "__main__":
    df_raw = get_raw_data()

    print(df_raw.isnull().sum())

    print(df_raw['Parch'].value_counts())
    print(df_raw.info())

    print(df_raw.groupby('Sex')['Survived'].sum())

    df_raw['family'] = df_raw['SibSp'] + df_raw['Parch'] + 1
    df_raw['isAlone'] = np.where(df_raw['family'] > 1, False, True)
    df_raw['isChild'] = np.where(df_raw['Age'] < 18, True, False)

    print(df_raw.groupby('family')['Survived'].sum())

    df_raw['title'] = df_raw['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])

    print(df_raw.groupby('title')['Survived'].mean())
    #print(df_raw['title'])

    # df_raw.plot(x='Embarked', y='Fare')
    #
    # plt.show()

    #TODO add deck from Cabin column
    #TODO check mother corelation

    #print all columns
    # pd.set_option('display.expand_frame_repr', False)
    # print(df_raw.head())

    # print(df_raw.count())

    df_raw.drop(['Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)

    print(df_raw.head())

    train_cats(df_raw)
    df, y, nas = proc_df(df_raw, 'Survived')
    n_trn = 700
    X_train, X_valid = split_vals(df, n_trn)
    y_train, y_valid = split_vals(y, n_trn)

    m = RandomForestRegressor(n_estimators=40,
                               n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)

    for feat, importance in zip(df.columns, m.feature_importances_):
        print(
        'feature: {f}, importance: {i}'.format(f=feat, i=importance))

    print(m.score(X_train, y_train))
    print(m.score(X_valid, y_valid))
    print(m.oob_score_)

    y_pred = m.predict(X_valid)
    print("Accuracy:", metrics.accuracy_score(y_pred.round(), y_valid))