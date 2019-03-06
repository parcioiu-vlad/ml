import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from fastai.structured import *
from sklearn.naive_bayes import MultinomialNB

if __name__ == "__main__":
    df_train_raw = pd.read_csv('../data/app/train.csv', low_memory=False)
    df_test_raw = pd.read_csv('../data/app/test.csv', low_memory=False)
    # print(df_train_raw.head)
    # print(df_test_raw.head)

    # tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
    #                         stop_words='english')
    # features = tfidf.fit_transform(df_train_raw.description).toarray()
    # labels = df_train_raw.type
    # print(features)
    # print(features.shape)
    # print('\n')
    # df_train = df_train_raw.join(pd.DataFrame(features))
    # df_train_raw.drop('name', axis=1, inplace=True)
    # df_train_raw.drop('description', axis=1, inplace=True)
    # train_cats(df_train_raw)
    # df, y, nas = proc_df(df_train_raw, 'type')

    count_vect = CountVectorizer()
    df = count_vect.fit_transform(df_train_raw.description)

    le = LabelEncoder()
    y = le.fit_transform(df_train_raw.type)

    print(df)

    clf = MultinomialNB().fit(df, y)

    for index, row in df_test_raw.iterrows():
        words_vect = count_vect.transform([row["description"]])
        print(clf.predict(words_vect))


