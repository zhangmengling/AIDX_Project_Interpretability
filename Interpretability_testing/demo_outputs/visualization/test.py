import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.head()

X_train, X_test, Y_train, Y_test = train_test_split(df[data.feature_names], df['target'], random_state=0)


clf = DecisionTreeClassifier(max_depth = 2, random_state = 0)
clf.fit(X_train, Y_train)

clf.predict(X_test.iloc[0].values.reshape(1, -1))

score = clf.score(X_test, Y_test)
print(score)

# fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 300)
# tree.plot_tree(clf)
# # plt.show()
# fig.savefig('plottreedefault.png')
#
# # Putting the feature names and class names into variables
# fn = ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
# cn = ['setosa', 'versicolor', 'virginica']
#
# fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 300)
#
# tree.plot_tree(clf,
#                feature_names = fn,
#                class_names=cn,
#                filled = True);
# fig.savefig('plottreefncn.png')

