#!/usr/bin/python
import sys
import csv
import re
import dircache
import os
import string
import numpy as np
import scipy as sp
from sklearn import tree
from sklearn.cross_validation import train_test_split

def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll
#training data
X = np.genfromtxt(sys.argv[1],delimiter=',')   
X = np.asarray(X)
y = np.genfromtxt(sys.argv[2])
#test data
X_test = np.genfromtxt(sys.argv[3],delimiter=',')  
X_test = np.asarray(X_test)
#range of paramters
depth = range(int(sys.argv[4]),int(sys.argv[5]))
#shuffle data

logloss = []
for d in depth:
	temp = []
	for j in range(1,6):
		data_train, data_test, labels_train, labels_test = train_test_split(X, y, test_size=0.2, random_state=42)
		clf = tree.DecisionTreeClassifier(max_depth=d)
		clf = clf.fit(data_train,labels_train)
		t = clf.predict_proba(data_test)
		act = t[:,1]
		temp.append(llfun(labels_test,act))
	logloss.append(np.mean(temp))
best_depth = depth[np.argmin(logloss)]

np.savetxt('logloss.txt',logloss,fmt='%f',delimiter='\n')
#predict labels for testing data
clf = tree.DecisionTreeClassifier(max_depth=13)
clf = clf.fit(X,y)
t = clf.predict_proba(X_test)
w = t[:,1]
np.savetxt('predict_labels.txt',w,fmt='%f',delimiter='\n')

