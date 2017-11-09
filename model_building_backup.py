import os
import numpy as np
import pandas as pd
import implement_functions as mf
import random
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
import time


##################################################################################################
##################################################################################################
##################################################################################################

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/constructed_data')

dat_1 = pd.read_csv('dat_31_07_17.csv')
X_train = dat_1.iloc[:,:-1]
Y_train = dat_1.iloc[:,-1]
dat_0 = pd.read_csv('dat_zero_31_07_17.csv')
all_index = range(dat_0.shape[0])

##coef_list = []
##inter_list = []
##
##for i in range(50):
##    start = time.time()
##    test_index = random.sample(all_index,5000)
##    validation_index = [x for x in all_index if x in test_index]
##    dat_0_test = dat_0.loc[test_index,:].reset_index().drop('index',1)
##    dat_0_validate = dat_0.loc[validation_index,:].reset_index().drop('index',1)
##    X_test = dat_0_test.iloc[:,:-1]
##    Y_test = dat_0_test.iloc[:,-1]
##
####    X_validate = dat_0_validate.iloc[:,:-1]
####    X_validate = X_validate.drop(['0','5','8','10','11','12','16','18','22'],axis=1)
####    Y_validate = dat_0_validate.iloc[:,-1]
##
##    X_all = pd.concat([X_train,X_test],axis=0).reset_index().drop('index',1)
##    X_all = X_all.drop(['3','4','8','10','11','12','13','14','16','17','18','22'],axis=1)
##    Y_all = pd.concat([Y_train,Y_test],axis=0).reset_index().drop('index',1)
##
##    X_train, X_test, y_train, y_test,test_index = mf.train_test_split_my(X_all, Y_all, test_size=0.3)
##    clf = LogisticRegression( max_iter=10000,C=0.5,verbose=0)
##    clf.fit(X_train,y_train)
##    coef_list.append(clf.coef_)
##    inter_list.append(clf.intercept_)
##    print(i+1,(time.time() - start)/60)
##    
##coef_all = pd.DataFrame([x.tolist()[0] for x in coef_list])
##inter_all = [x.tolist()[0] for x in inter_list]

test_index = random.sample(all_index,5000)
validation_index = [x for x in all_index if x in test_index]
dat_0_test = dat_0.loc[test_index,:].reset_index().drop('index',1)
dat_0_validate = dat_0.loc[validation_index,:].reset_index().drop('index',1)
X_test = dat_0_test.iloc[:,:-1]
Y_test = dat_0_test.iloc[:,-1]

##    X_validate = dat_0_validate.iloc[:,:-1]
##    X_validate = X_validate.drop(['0','5','8','10','11','12','16','18','22'],axis=1)
##    Y_validate = dat_0_validate.iloc[:,-1]

X_all = pd.concat([X_train,X_test],axis=0).reset_index().drop('index',1)
X_all = X_all.drop(['3','4','8','10','11','12','13','14','16','17','18','22'],axis=1)
Y_all = pd.concat([Y_train,Y_test],axis=0).reset_index().drop('index',1)
clf_logit = mf.prelim_logit(X_all,Y_all,0.8,1)
clf_gbm = mf.prelim_gbm(X_all,Y_all,0.8)

print(clf_logit.intercept_,clf_logit.coef_)

##estimator = tree.DecisionTreeClassifier(min_samples_leaf=25)
##estimator = estimator.fit(X_all,Y_all)
####dot_data = tree.export_graphviz(clf, out_file=None)
####graph = pydotplus.graph_from_dot_data(dot_data)
####pdf_name = "temp_vis.pdf"
####graph.write_pdf(pdf_name)
##
##n_nodes = estimator.tree_.node_count
##children_left = estimator.tree_.children_left
##children_right = estimator.tree_.children_right
##feature = estimator.tree_.feature
##threshold = estimator.tree_.threshold
##
##
### The tree structure can be traversed to compute various properties such
### as the depth of each node and whether or not it is a leaf.
##node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
##is_leaves = np.zeros(shape=n_nodes, dtype=bool)
##stack = [(0, -1)]  # seed is the root node id and its parent depth
##while len(stack) > 0:
##    node_id, parent_depth = stack.pop()
##    node_depth[node_id] = parent_depth + 1
##
##    # If we have a test node
##    if (children_left[node_id] != children_right[node_id]):
##        stack.append((children_left[node_id], parent_depth + 1))
##        stack.append((children_right[node_id], parent_depth + 1))
##    else:
##        is_leaves[node_id] = True
##
##print("The binary tree structure has %s nodes and has "
##      "the following tree structure:"
##      % n_nodes)
##for i in range(n_nodes):
##    if is_leaves[i]:
##        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
##    else:
##        print("%snode=%s test node: go to node %s if X[:, %s] <= %ss else to "
##              "node %s."
##              % (node_depth[i] * "\t",
##                 i,
##                 children_left[i],
##                 feature[i],
##                 threshold[i],
##                 children_right[i],
##                 ))
##print()
