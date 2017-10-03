import pandas

from models.impute import impute, normalize, build_model

data = pandas.read_csv('../data/train.csv')
# print(train.head(20))
# print(train.isnull().sum())

# In [305]: train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

train = data.head(600)
test = data.tail(291)
print('---------- test data ---------')
print(test.isnull().sum())

train = impute(train)
train = normalize(train)
print(train.head(20))

X = train.values[:, 1:]
y = train.values[:, 0:1]

# print(test.isnull().sum())
test = impute(test)
test = normalize(test)
print('----------- test after normalize ---------')
print(test.head(20))
print(test.isnull().sum())

test_X = test.values[:, 1:]
test_y = test.values[:, 0:1]
build_model(X, y, test_X, test_y)
