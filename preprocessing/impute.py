import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import adam, Adam
from sklearn import preprocessing
from sklearn.base import TransformerMixin
import pandas as pd
from sklearn.preprocessing import Imputer


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

    values = data.values
    age_imputer = Imputer(strategy='mean')
    age_trans = age_imputer.fit_transform(values[:, 2:3])
    data['Age'] = age_trans

    embark_imputer = CategoricalImputer()
    embark_trans = embark_imputer.fit_transform(pd.DataFrame(values[:, 6:7]))
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
    # print(data.values[1:5, 2:3])
    # norm_age = preprocessing.normalize(data.values[:, 1:2], axis=0)
    # print(norm_age)

    # print(data.values[1:5, 5:6])
    # norm_fare = preprocessing.normalize(data.values[:, 4:5], axis=0)

    # data['Age'] = norm_age
    # data['Fare'] = norm_fare

    # print(data.head(20))
    # scale_age = preprocessing.scale(norm_age, axis=0)

    norm_data = preprocessing.normalize(data.values[:, 1:5], axis=0)
    # norm_data = preprocessing.scale(norm_data, axis=0)
    # print(norm_data)
    data['Age'] = norm_data[:, 0:1]
    data['SibSp'] = norm_data[:, 1:2]
    data['Parch'] = norm_data[:, 2:3]
    data['Fare'] = norm_data[:, 3:4]
    return data


def build_model(X, y, test_X, test_y):
    import math
    # pass

    # print(y)
    # num features: 10

    model = Sequential()
    model.add(Dense(12, input_dim=12, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=1000, batch_size=1000, verbose=2)
    # scores = model.evaluate(X, y)
    # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    y_hat = model.predict(test_X, batch_size=1000, verbose=1)
    # print(y_hat)
    rounded = [math.fabs(round(x[0])) for x in y_hat]
    z = [a[0][0] == a[1] for a in zip(test_y, rounded)]
    print(z)
    print(z.count(True) / len(z))

