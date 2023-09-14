from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import genfromtxt
from sklearn import tree

irisDataset = genfromtxt('IrisNew.csv', delimiter=',',dtype=None,encoding=None)
x = pd.DataFrame(irisDataset[1:,:4])  # Extracting the features that is the first four columns
y = pd.DataFrame(irisDataset[1:,4]) # Extracting the target variable that is the 5th column
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)
clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
