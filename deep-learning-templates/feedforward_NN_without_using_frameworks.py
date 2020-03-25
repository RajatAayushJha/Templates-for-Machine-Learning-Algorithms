#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as col
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score,mean_squared_error
from tqdm import tqdm_notebook
from sklearn.datasets import make_blobs

#Generate Data
data,labels=make_blobs(n_samples=1000,n_features=2,centers=4,random_state=0)
print(data.shape,labels.shape)


#Visualising data
my_cmap=col.LinearSegmentedColormap.from_list("",["red","green","yellow","blue"])
plt.scatter(data[:,0],data[:,1],c=labels,cmap=my_cmap)
plt.show()

#changing data into binary class
labels_changed=labels
labels_changed=np.mod(labels_changed,2)

#FNN class - shape of model - [2,2,1]
class FNN:
    def __init__(self):
        self.w1=np.random.randn()
        self.w2=np.random.randn()
        self.w3=np.random.randn()
        self.w4=np.random.randn()
        self.w5=np.random.randn()
        self.w6=np.random.randn()
        self.b1=0
        self.b2=0
        self.b3=0
    
    def sigmoid (self,x):
        return 1.0/(1.0+np.exp(-x))
    
    def forward_pass(self,x):
        self.x1,self.x2=x
        self.a1=self.w1*self.x1+self.w2*self.x2
        self.h1=self.sigmoid(self.a1)
        self.a2=self.w3*self.x1+self.w4*self.x2
        self.h2=self.sigmoid(self.a2)
        self.a3=self.w5*self.h1+self.w6*self.h2
        self.h3=self.sigmoid(self.a3)
        return self.h3
    
    def grad(self,x,y):
        self.forward_pass(x)
        
        self.dw5=(self.h3-y)*self.h3*(1-self.h3)*self.h1
        self.dw6=(self.h3-y)*self.h3*(1-self.h3)*self.h2
        self.db3=(self.h3-y)*self.h3*(1-self.h3)
        
        self.dw1=(self.h3-y)*self.h3*(1-self.h3)*self.w5*self.h1*(1-self.h1)*self.x1
        self.dw2=(self.h3-y)*self.h3*(1-self.h3)*self.w5*self.h1*(1-self.h1)*self.x2
        self.db1=(self.h3-y)*self.h3*(1-self.h3)*self.w5*self.h1*(1-self.h1)
        
        self.dw3=(self.h3-y)*self.h3*(1-self.h3)*self.w6*self.h2*(1-self.h2)*self.x1
        self.dw4=(self.h3-y)*self.h3*(1-self.h3)*self.w6*self.h2*(1-self.h2)*self.x2
        self.db2=(self.h3-y)*self.h3*(1-self.h3)*self.w6*self.h2*(1-self.h2)
    
    def predict(self,X):
            Y_pred=[]
            for x in X:
                y_pred=self.forward_pass(x)
                Y_pred.append(y_pred)
            return np.array(Y_pred)
        
    def fit(self,X,Y,epochs=1,learning_rate=1,initialize=False,display_loss=True):
        if initialize:
            self.w1=np.random.randn()
            self.w2=np.random.randn()
            self.w3=np.random.randn()
            self.w4=np.random.randn()
            self.w5=np.random.randn()
            self.w6=np.random.randn()
            self.b1=0
            self.b2=0
            self.b3=0
        
        if display_loss:
            loss={}
        
        for i in tqdm_notebook(range(epochs),total=epochs,unit="epoch"):
            
            dw1,dw2,dw3,dw4,dw5,dw6,db1,db2,db3=[0]*9
            
            for x,y in zip(X,Y):
                self.grad(x,y)
                dw1+=self.dw1
                dw2+=self.dw2
                dw3+=self.dw3
                dw4+=self.dw4
                dw5+=self.dw5
                dw6+=self.dw6
                db1+=self.db1
                db2+=self.db2
                db3+=self.db3
            
            m=X.shape[1]
            #standardizing with m
            self.w1-=learning_rate*dw1/m
            self.w2-=learning_rate*dw2/m
            self.w3-=learning_rate*dw3/m
            self.w4-=learning_rate*dw4/m
            self.w5-=learning_rate*dw5/m
            self.w6-=learning_rate*dw6/m
            self.b1-=learning_rate*db1/m
            self.b2-=learning_rate*db2/m
            self.b3-=learning_rate*db3/m
            
            if(display_loss):
                Y_pred=self.predict(X)
                loss[i]=mean_squared_error(Y_pred,Y)
        if display_loss:
            plt.plot(np.array(list(loss.values())).astype("float"))
            plt.xlabel("Epochs")
            plt.ylabel("Mean Squared Error")
            plt.show()

ffn=FNN()
ffn.fit(X_train,Y_train,epochs=1000,learning_rate=.01,initialize=False,display_loss=True)

# Evaluating model 
Y_pred_train=ffn.predict(X_train)
Y_binarised_pred_train=(Y_pred_train>=0.5).astype(int).ravel()
Y_pred_test=ffn.predict(X_test)
Y_binarised_pred_test=(Y_pred_test>=0.5).astype(int).ravel()
train_accuracy=accuracy_score(Y_binarised_pred_train,Y_train)
test_accuracy=accuracy_score(Y_binarised_pred_test,Y_test)

print(train_accuracy)
print(test_accuracy)    
        
        
            
    
                