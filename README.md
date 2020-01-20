# Decision-Tree-on-Iris-dataset-for-multiclass-classification
Exploring disicion tree on iris dataset

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
