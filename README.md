# Decision-Tree-on-Iris-dataset-for-multiclass-classification
Exploring disicion tree on iris dataset

Data source: https://archive.ics.uci.edu/ml/datasets/iris

The below link is used to implement decision tree:
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

I have used 4 combinations of criteria and splitter strategy:

*clf_decisiontree1 = DecisionTreeClassifier(criterion = "gini",splitter="best")*

*clf_decisiontree2 = DecisionTreeClassifier(criterion = "entropy",splitter="best")*

*clf_decisiontree3 = DecisionTreeClassifier(criterion = "gini",splitter="random")*

*clf_decisiontree4 = DecisionTreeClassifier(criterion = "entropy",splitter="random")*

The impurity measurement can be done using either Gini inde or Entropy calculation:


![alt text](https://miro.medium.com/max/700/1*hmWktuMpZo5hX1AavOc92w.png)

But there is no much difference as to which of these two criteria to be used and also the below paper suggest that there might be 2% poosible difference.

https://www.unine.ch/files/live/sites/imi/files/shared/documents/papers/Gini_index_fulltext.pdf

One possible reason why gini is the default value in scikit-learn is that entropy might be a little slower to compute (because it makes use of a logarithm).


### Results:
Validation Accuracy 1: 0.9666666666666667

Validation Accuracy 2: 0.9666666666666667

Validation Accuracy 3: 0.9666666666666667

Validation Accuracy 4: 0.9666666666666667

To check which is important:

*column feature importance*

2       petal_length    0.584546

3       petal_width     0.389133

1       sepal_width     0.018801

0       sepal_length    0.007520

We can see that patel lenght is most important and so is at the Root Node


