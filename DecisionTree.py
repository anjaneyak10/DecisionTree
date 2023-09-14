import numpy.version
import pandas.util.version
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import genfromtxt
from sklearn import tree

# Reading the data from csv file using numpy
irisDataset = genfromtxt('IrisNew.csv', delimiter=',',dtype=None,encoding=None)
# Extracting the features that is the first four columns
x = pd.DataFrame(irisDataset[1:,:4])
# Extracting the target variable that is the 5th column
y = pd.DataFrame(irisDataset[1:,4])


# Training the code on 67% of the data and testing with 33%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)
clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
print("Accuracy with training 67 % of the data and testing with 33% ",clf.score(x_test,y_test))
with open("iris20Percent.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

#Training the model with classifier criterion = entropy
clf = tree.DecisionTreeClassifier()
clf.criterion='entropy'
clf.fit(x_train,y_train)
print("Accuracy with training 67 % of the data and testing with 33% for criterion = entropy",clf.score(x_test,y_test))
with open("irisEntropy.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

#Training the model with classifier criterion = log_loss
clf = tree.DecisionTreeClassifier()
clf.criterion='log_loss'
clf.fit(x_train,y_train)
print("Accuracy with training 67 % of the data and testing with 33% for criterion = log_loss",clf.score(x_test,y_test))
with open("irisLogLoss.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

#Training the model twice
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=12)
clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
#Retraining the classifier using the 67% of the remaining data and then testing with the remaining 33%
x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.33, random_state=12)
clf.fit(x_train,y_train)
print("Accuracy of the model after retraining the model with 67% of the remaining data",clf.score(x_test,y_test))
with open("irisRetraining.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
