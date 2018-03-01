import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('Data/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)

df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])


XTrain, XTest, yTrain, yTest = cross_validation.train_test_split(
    X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(XTrain, yTrain)

accuracy = clf.score(XTest, yTest)

print(accuracy)

exampleMeasures = np.array(
    [[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
exampleMeasures = exampleMeasures.reshape(len(exampleMeasures), -1)
prediction = clf.predict(exampleMeasures)
print(prediction)
