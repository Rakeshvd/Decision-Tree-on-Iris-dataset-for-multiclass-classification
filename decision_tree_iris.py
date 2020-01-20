# Load libraries
import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn import datasets
import pydotplus
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target


# Create decision tree classifer object
#clf = DecisionTreeClassifier(random_state=0)
clf_decisiontree1 = DecisionTreeClassifier(criterion = "gini",splitter="best")
clf_decisiontree2 = DecisionTreeClassifier(criterion = "entropy",splitter="best")
clf_decisiontree3 = DecisionTreeClassifier(criterion = "gini",splitter="random")
clf_decisiontree4 = DecisionTreeClassifier(criterion = "entropy",splitter="random")

#clf = DecisionTreeClassifier(criterion = "gini",random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 70% training and 30% test
# Train model
model1 = clf_decisiontree1.fit(X_train, y_train)
model2 = clf_decisiontree2.fit(X_train, y_train)
model3 = clf_decisiontree3.fit(X_train, y_train)
model4 = clf_decisiontree4.fit(X_train, y_train)

#predict on validation
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)
y_pred4 = model4.predict(X_test)

print("Validation Accuracy 1:",metrics.accuracy_score(y_test, y_pred1))
print("Validation Accuracy 2:",metrics.accuracy_score(y_test, y_pred2))
print("Validation Accuracy 3:",metrics.accuracy_score(y_test, y_pred3))
print("Validation Accuracy 4:",metrics.accuracy_score(y_test, y_pred4))

# Create DOT data
dot_data1 = tree.export_graphviz(clf_decisiontree1, out_file=None, 
                                feature_names=iris.feature_names,  
                                class_names=iris.target_names)
dot_data2 = tree.export_graphviz(clf_decisiontree2, out_file=None, 
                                feature_names=iris.feature_names,  
                                class_names=iris.target_names)
dot_data3 = tree.export_graphviz(clf_decisiontree3, out_file=None, 
                                feature_names=iris.feature_names,  
                                class_names=iris.target_names)
dot_data4 = tree.export_graphviz(clf_decisiontree4, out_file=None, 
                                feature_names=iris.feature_names,  
                                class_names=iris.target_names)

# Draw graph
graph1 = pydotplus.graph_from_dot_data(dot_data1)  
graph2 = pydotplus.graph_from_dot_data(dot_data2)  
graph3 = pydotplus.graph_from_dot_data(dot_data3)  
graph4 = pydotplus.graph_from_dot_data(dot_data4)  

# Show graph
Image(graph1.create_png())
Image(graph2.create_png())
Image(graph3.create_png())
Image(graph4.create_png())

# Create PDF
graph1.write_pdf("iris_dot_file1.pdf")
graph2.write_pdf("iris_dot_file2.pdf")
graph3.write_pdf("iris_dot_file3.pdf")
graph4.write_pdf("iris_dot_file4.pdf")

# Create PNG
graph1.write_png("iris_dot_file1.png")
graph2.write_png("iris_dot_file2.png")
graph3.write_png("iris_dot_file3.png")
graph4.write_png("iris_dot_file4.png")

importances = pd.DataFrame({'feature':['sepal_length','sepal_width','petal_length','petal_width'],'importance':clf_decisiontree1.feature_importances_})
importances = importances.sort_values('importance',ascending=False)
print(importances)
