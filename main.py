# To perform operations on dataset

import pandas as pd

import numpy as np

# Machine learning model

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

# Visualization

from sklearn import metrics

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import export_graphviz

df = pd.read_csv('D:\\test\\dataset.csv')

dot_file = 'D:\\test\\tree.dot'

confusion_matrix_file = 'D:\\test\\confusion_matrix.png'
print(df.head())
X = df.iloc[:, :-1]

y = df.iloc[:, -1]


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
model = DecisionTreeClassifier()

model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print(metrics.classification_report(ypred, ytest))

print("\n\nAccuracy Score:", metrics.accuracy_score(ytest, ypred).round(2)*100, "%")
mat = confusion_matrix(ytest, ypred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label');

plt.savefig(confusion_matrix_file)
export_graphviz(model, out_file=dot_file, feature_names=X.columns.values)

