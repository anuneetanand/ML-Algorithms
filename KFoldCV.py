# Anuneet Anand

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

np.random.seed(0)

class MyKfoldCV():
    """
    My implementation of K-Fold cross validation
    """
    def __init__(self,Model,X,y,k=3,plot=False):
        self.Model = Model
        self.X = X
        self.y = y
        self.k = k
        self.plot = plot
        self.Record = {}                                                    # To store errors for different folds

    def run(self):
        '''
        - Trains and evaluates the model on K folds and record errors for different folds.
        - It outputs the best fold.
        - If plot is specified as True, it plots training and validation loss for the best fold.
        - Returns average of errors on different folds.
        '''
        print("----------------------------------------------------------------------------------------------------------------------------")
        n = self.y.shape[0]
        d = n//self.k
        error = 0

        for i in range(self.k):
            X_train = np.concatenate((self.X[:i*d],self.X[i*d+d:]))
            y_train = np.concatenate((self.y[:i*d],self.y[i*d+d:]))
            X_test = self.X[i*d:i*d+d]
            y_test = self.y[i*d:i*d+d]

            self.Model.fit(X_train, y_train)
            ypred =self.Model.predict(X_test)
            self.Model.evaluate(X_train,y_train,X_test,y_test)
            I = self.Model.info()
            self.Record[i+1] = I[0]
            error+=I[0]

            print("Fold : "+str(i+1)+" "+str(I[1])+" : "+str(I[0]))
        
        print("----------------------------------------------------------------------------------------------------------------------------")
        

        Best_Error = min(list(self.Record.values()))
        Average_Error =error/self.k
        print("Best Fold Error : ",Best_Error)
        print(" K = "+str(self.k)+" :> Average Error : ",Average_Error)
        
        i = 0
        for fold in self.Record:
            if self.Record[fold]==Best_Error:
                i = fold-1
        
        print("Best performance on Fold : "+str(i+1)+" -> Testing Set ["+str(i*d)+","+str(i*d+d)+"]")

        X_train = np.concatenate((self.X[:i*d],self.X[i*d+d:]))
        y_train = np.concatenate((self.y[:i*d],self.y[i*d+d:]))
        X_test = self.X[i*d:i*d+d]
        y_test = self.y[i*d:i*d+d]
        self.Model.evaluate(X_train,y_train,X_test,y_test,all=True,plot=self.plot)

        return Average_Error
