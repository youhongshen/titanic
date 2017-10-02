import numpy
import pandas

from preprocessing.impute import impute, normalize, build_model

data = pandas.read_csv('../data/train.csv')


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
build_model(X, y, test_X, test_y)
