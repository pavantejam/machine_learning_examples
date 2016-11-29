import numpy as np
import matplotlib.pyplot as plt

# load the data
X=[]
Y=[]

Header=0
for line in open('practice_1_lr.csv'):
    Header = Header+1
    if Header>1 :
        alpha = line.split(',')
        X_row=[]        
        for x in range(1,len(alpha)):
            X_row.extend([float(alpha[x].strip(" "))])
        X_row.extend([1])
        X.append(X_row)  
        Y.append(float(alpha[0].strip(" ")))
        

Y=np.array(Y)
X=np.array(X)

for pred in range(len(X_row)):
    plt.scatter(X[:,pred], Y)
    plt.title("pred = %d" % pred)
    plt.show()


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X[:,0], X[:,1], Y)
#plt.show()

#calculate weights
W = np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))

Yhat= np.dot(X, W)
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)
print(r2)