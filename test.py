#!/usr/bin/env python3

import graphviz 

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0, 50, 100]
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

test_target = iris.target[test_idx]
test_data   = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
clf.predict(test_data)


dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 

#dot_data = tree.export_graphviz(clf, out_file=None, 
#    feature_names=iris.feature_names,  
#    class_names=iris.target_names,  
#    filled=True, rounded=True,  
##    special_characters=True)
#graph = graphviz.Source(dot_data)  
#graph 
