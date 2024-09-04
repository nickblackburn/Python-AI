import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import sklearn.tree as tree

# Read the CSV file
path = "/Users/nicholas/Desktop/Machine Learning/data/drug200.csv"
my_data = pd.read_csv(path, delimiter=",")

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].copy()

le = preprocessing.LabelEncoder()

for column in ['Sex', 'BP', 'Cholesterol']:
    X[column] = le.fit_transform(X[column])

y = my_data["Drug"]

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

print('Training set:', X_trainset.shape, y_trainset.shape)
print('Testing set:', X_testset.shape, y_testset.shape)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree.fit(X_trainset, y_trainset)

predTree = drugTree.predict(X_testset)

print("Decision Tree's Accuracy: {:.2%}".format(metrics.accuracy_score(y_testset, predTree)))

plt.figure(figsize=(12, 8))
tree.plot_tree(drugTree, feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'], class_names=sorted(y.unique()), filled=True)
plt.show()
