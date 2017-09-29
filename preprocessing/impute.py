import numpy as np
from sklearn.base import TransformerMixin
from sklearn.linear_model import LinearRegression
# from pandas import read_csv
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
    embark_as_cat = cat_to_num(embark_trans)
    data['embarked_C'] = embark_as_cat[0]
    data['embarked_Q'] = embark_as_cat[1]
    data['embarked_S'] = embark_as_cat[2]
    data = data.drop('Embarked', axis=1)

    return data
