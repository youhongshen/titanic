import pandas
import numpy as np
from sklearn import preprocessing

from preprocessing.impute import impute

train = pandas.read_csv('../data/train.csv')
# print(train.head(20))
# print(train.isnull().sum())

train = impute(train)
print(train.head(20))
# print(train.isnull().sum())

print(train.values[1:5, 2:3])
norm_age = preprocessing.normalize(train.values[:, 2:3], axis=0)
print(norm_age)

print(train.values[1:5, 5:6])
norm_fare = preprocessing.normalize(train.values[:, 5:6], axis=0)

train['Age'] = norm_age
train['Fare'] = norm_fare

print(train.head(20))

# scale_age = preprocessing.scale(norm_age)
# print(scale_age)

