import random
from os.path import join

import numpy
import numpy as np
import pandas
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import adam, Adam
from keras.regularizers import l2
from sklearn import preprocessing
from sklearn.base import TransformerMixin
import pandas as pd
from sklearn.preprocessing import Imputer
import math


def cat_to_num(data):
    categories = np.unique(data)
    features = []
    for cat in categories:
        binary = (data == cat)
        features.append(binary.astype("int"))
    return features


class CategoricalImputer(TransformerMixin):
    def fit(self, X, y=None):
        # uniques, counts = np.unique(X, return_counts=True)
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                              index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def impute(data):
    sex_as_cat = cat_to_num(data[['Sex']])
    data['female'] = sex_as_cat[0]
    data['male'] = sex_as_cat[1]
    data = data.drop('Sex', axis=1)

    data = data.drop('PassengerId', axis=1)
    data = data.drop('Name', axis=1)
    data = data.drop('Ticket', axis=1)
    data = data.drop('Cabin', axis=1)

    data['Age'] = Imputer(strategy='mean').fit_transform(data[['Age']].values)
    data['Fare'] = Imputer(strategy='mean').fit_transform(data[['Fare']].values)

    embark_imputer = CategoricalImputer()
    embark_trans = embark_imputer.fit_transform(data[['Embarked']])
    # print(embark_trans)
    embark_as_cat = cat_to_num(embark_trans)
    data['embarked_C'] = embark_as_cat[0]
    data['embarked_Q'] = embark_as_cat[1]
    data['embarked_S'] = embark_as_cat[2]
    data = data.drop('Embarked', axis=1)

    pclass_cat = cat_to_num(data[['Pclass']])
    data['pclass_1'] = pclass_cat[0]
    data['pclass_2'] = pclass_cat[1]
    data['pclass_3'] = pclass_cat[2]
    data = data.drop('Pclass', axis=1)

    return data


def normalize(data):
    norm_data = preprocessing.normalize(data[['Age', 'SibSp', 'Parch', 'Fare']].values, axis=0)
    # norm_data = models.scale(norm_data, axis=0)
    data[['Age', 'SibSp', 'Parch', 'Fare']] = norm_data
    return data


def build_model(X, y, test_X, test_y, real_test):

    model = Sequential()
    model.add(Dense(12, input_dim=12, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(48, activation='relu', kernel_regularizer=l2(l=0.01), kernel_initializer='he_uniform'))
    model.add(Dense(48, activation='relu', kernel_regularizer=l2(l=0.01), kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='tanh', kernel_regularizer=l2(l=0.01), kernel_initializer='he_uniform'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.005, decay=0.0005), metrics=['accuracy'])
    model.fit(X, y, epochs=1500, batch_size=1024, verbose=2)
    scores = model.evaluate(X, y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    y_hat = model.predict(test_X, batch_size=1024, verbose=1)
    # print(y_hat)
    rounded = [math.fabs(round(x[0])) for x in y_hat]
    z = [a[0][0] == a[1] for a in zip(test_y, rounded)]
    print(z)
    print(z.count(True) / len(z))

    print('--- predict real test ----')
    pred = model.predict(real_test, batch_size=1024)
    rounded = [int(math.fabs(round(x[0]))) for x in pred]
    return rounded


def main():

    dir = '../data'
    data = pandas.read_csv(join(dir, 'train.csv'))

    data = impute(data)
    data = normalize(data)
    print('-------- entire data set after normalize -------------')
    print(data.head(20))
    print(data.isnull().sum())

    train, validate, test = numpy.split(data.sample(frac=1), [int(.6*len(data)), int(.8*len(data))])

    X = train.values[:, 1:]
    y = train.values[:, 0:1]

    test_X = test.values[:, 1:]
    test_y = test.values[:, 0:1]

    # rt = None
    print('-------- test data ---------')
    rt = pandas.read_csv(join(dir, 'test.csv'))
    passenger_ids = rt[['PassengerId']].values[:, 0]

    print(rt.head(20))
    print(rt.isnull().sum())
    rt = impute(rt)
    rt = normalize(rt)
    print('-------- test data after processing ---------')
    print(rt.head(20))
    print(rt.isnull().sum())

    prediction = build_model(X, y, test_X, test_y, rt.values)
    pd = pandas.DataFrame()
    pd['PassengerId'] = passenger_ids
    pd['Survived'] = prediction
    pd.to_csv(join(dir, 'submission.csv'), index=False)

main()
