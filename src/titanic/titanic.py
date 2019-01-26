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


def add_features(df_input):
    df_input['family'] = df_input['SibSp'] + df_input['Parch'] + 1
    df_input['isAlone'] = np.where(df_input['family'] > 1, False, True)
    df_input['isChild'] = np.where(df_input['Age'] < 18, True, False)
    df_input['title'] = df_input['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
    return df_input


def train_model():
    df_raw = get_raw_data()
    df_raw = add_features(df_raw)
    df_raw.drop(['Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)
    train_cats(df_raw)
    df, y, nas = proc_df(df_raw, 'Survived')
    n_trn = 700
    X_train, X_valid = split_vals(df, n_trn)
    y_train, y_valid = split_vals(y, n_trn)

    m = RandomForestRegressor(n_estimators=40,
                              n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)

    # for feat, importance in zip(df.columns, m.feature_importances_):
    #     print(
    #     'feature: {f}, importance: {i}'.format(f=feat, i=importance))
    #
    # print(m.score(X_train, y_train))
    # print(m.score(X_valid, y_valid))
    # print(m.oob_score_)
    #
    # y_pred = m.predict(X_valid)
    # print("Accuracy:", metrics.accuracy_score(y_pred.round(), y_valid))

    return m, df


if __name__ == "__main__":
    # print(df_raw.isnull().sum())
    #
    # print(df_raw['Parch'].value_counts())
    # print(df_raw.info())
    #
    # print(df_raw.groupby('Sex')['Survived'].sum())
    #
    # print(df_raw.groupby('family')['Survived'].sum())
    #
    # print(df_raw.groupby('title')['Survived'].mean())
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

    result = train_model()
    m = result[0]
    df_input = result[1]
    print(df_input.info())

    df_raw_test = pd.read_csv('../../data/titanic/test.csv', low_memory=False)

    df_raw_test['Fare'].fillna(df_raw_test['Fare'].mean(), inplace=True)

    print(df_raw_test.info())

    df_raw_test = add_features(df_raw_test)

    passengers_ids = df_raw_test['PassengerId']

    df_raw_test.drop(['Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)
    train_cats(df_raw_test)
    df, y, nas = proc_df(df_raw_test)

    y_pred = m.predict(df).round().astype(int)

    preds = pd.concat([pd.DataFrame(passengers_ids), pd.DataFrame(y_pred)], axis=1)

    preds.to_csv('gender_submission.csv', sep=',', index=False)
