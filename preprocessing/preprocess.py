import pandas
import numpy as np

from preprocessing.impute import impute

train = pandas.read_csv('../data/train.csv')
print(train.head(20))
print(train.isnull().sum())

train = impute(train)
print(train.head(20))
print(train.isnull().sum())

