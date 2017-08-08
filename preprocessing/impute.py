import numpy
from sklearn.linear_model import LinearRegression
from pandas import read_csv
from sklearn.preprocessing import Imputer

def cat_to_num(data):
    categories = numpy.unique(data)
    features = []
    for cat in categories:
        binary = (data == cat)
        features.append(binary.astype("int"))
    return features

train = read_csv('../data/train.csv')

sex_as_cat = cat_to_num(train[['Sex']])
train['female'] = sex_as_cat[0]
train['male'] = sex_as_cat[1]
train = train.drop('Sex', axis=1)

train = train.drop('Name', axis=1)
train = train.drop('Ticket', axis=1)
train = train.drop('Cabin', axis=1)

# embark_as_cat = cat_to_num(train[['Embarked']])
# train['embarked_C'] = embark_as_cat[0]
# train['embarked_Q'] = embark_as_cat[1]
# train['embarked_S'] = embark_as_cat[2]

# print(train.describe())
print(train.head(20))
print(train.isnull().sum())



values = train.values
imputer = Imputer()
# transformed = imputer.fit_transform(values[:, 5:6])
# print(transformed.head(20))
# print(transformed.isnull().sum())
# reg = LinearRegression()