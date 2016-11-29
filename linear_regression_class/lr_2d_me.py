# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 18:15:59 2016

@author: Pavan
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#load the data
X=[]
Y=[]

for line in open('data_2d.csv'):
    x1,x2,y = line.split(',')
    X.append([1,float(x1),float(x2)]) #add the bias term X0=1
    Y.append(float(y))
    
Y=np.array(Y)
X=np.array(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()

#calculate weights
W = np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))

Yhat= np.dot(X, W)
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)
print(r2)