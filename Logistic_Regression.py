# Anuneet Anand

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

np.random.seed(0)

class MyLogisticRegression():
    """
    My implementation of Logistic Regression.
    """

    def __init__(self,alpha=0.01,epoch=1000,gradient_descent="BGD"):
        self.theta = np.empty((0))
        self.alpha = alpha
        self.epoch = epoch
        self.error = 0
        self.accuracy = 0
        self.gradient_descent = gradient_descent
        self.training_loss = []
        self.validation_loss = []

    def fit(self, X, y):
        """
        Fitting (training) the logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """

        n = X.shape[0]
        self.theta = np.zeros((X.shape[1],1))
        y = y.reshape(y.shape[0],1)

        if self.gradient_descent == "BGD":                                      # Batch Gradient Descent
            for i in range(self.epoch):
                H = self.sigmoid(np.dot(X,self.theta))
                D = np.dot(X.T,H-y)*(1/n)
                self.theta = self.theta - self.alpha * D

        elif self.gradient_descent == "SGD":                                    # Stochastic Gradient Descent
            for i in range(self.epoch):
                r = np.random.randint(0,n)
                H = self.sigmoid(np.dot(X[r].reshape(1,X.shape[1]),self.theta))
                D = np.dot(X[r].reshape(1,X.shape[1]).T,H-y[r].reshape(1,1))
                self.theta = self.theta - self.alpha * D


        # fit function has to return an instance of itself or else it won't work with test.py
        return self

    def predict(self, X):
        """
        Predicting values using the trained logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array y which contains the predicted values
        return np.round(self.sigmoid(np.dot(X,self.theta)))

    def evaluate(self,X_train,y_train,X_val,y_val,X_test,y_test,all=False,plot=False):
        '''
        - It calculates the appropriate loss for the model on Validation set
        - If all is specified as True, it calculates loss for each epoch.
        - If Plot is specified as True, it plots the training and validation loss.
        - Returns loss of model
        '''
        e = self.epoch    
        if plot == True : all = True

        n = y_train.shape[0]
        self.training_loss = []
        self.validation_loss = []
        iterations = [i for i in range(e)]

        y_train = y_train.reshape((y_train.shape[0],1))
        y_val = y_val.reshape((y_val.shape[0],1))
        y_test = y_test.reshape((y_test.shape[0],1))

        if all == False:
            self.error = self.error = self.loss(X_test,y_test)
            return self.error

        if self.gradient_descent == "BGD":                                          # Batch Gradient Descent

            self.theta = np.zeros((X_train.shape[1],1))
            
            for i in range(e):
                H = self.sigmoid(np.dot(X_train,self.theta))
                D = np.dot(X_train.T,H-y_train)*(1/n)
                self.theta = self.theta - self.alpha * D

                self.training_loss.append(self.loss(X_train,y_train))
                self.validation_loss.append(self.loss(X_val,y_val))

                if i == self.epoch - 1 : self.error = self.loss(X_test,y_test)

            if plot == True:

                plt.figure()
                plt.title("Batch Gradient Descent [ alpha : "+str(self.alpha)+" ]")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.plot(iterations,self.training_loss, label = "Training Loss")
                plt.plot(iterations,self.validation_loss, label = "Validation Loss")
                plt.legend()
                plt.show()

        elif self.gradient_descent == "SGD":                                            # Stochastic Gradient Descent

            self.theta = np.zeros((X_train.shape[1],1))
            
            for i in range(e):
                r = np.random.randint(0,n)
                H = self.sigmoid(np.dot(X_train[r].reshape(1,X_train.shape[1]),self.theta))
                D = np.dot(X_train[r].reshape(1,X_train.shape[1]).T,H-y_train[r].reshape(1,1))
                self.theta = self.theta - self.alpha * D

                self.training_loss.append(self.loss(X_train,y_train))
                self.validation_loss.append(self.loss(X_val,y_val))

                if i == self.epoch - 1: self.error = self.error = self.loss(X_test,y_test)

            if plot == True:

                plt.figure()
                plt.title("Stochastic Gradient Descent [ alpha : "+str(self.alpha)+" ]")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.plot(iterations,self.training_loss, label = "Training Loss")
                plt.plot(iterations,self.validation_loss, label = "Validation Loss")
                plt.legend()
                plt.show()

        return self.error

    def calculate_accuracy(self,y_pred,y_true):
        '''
        returns accuracy of prediction
        '''
        c = 0.0
        n = y_true.shape[0]
        for i in range(n):
            c += int(y_true[i]==y_pred[i])
        self.accuracy=c/n
        return c/n


    def loss(self,X,y):
        '''
        returns cross entropy loss
        '''
        n = X.shape[0]
        e = 10**(-5)
        H = self.sigmoid(np.dot(X,self.theta)+e)
        return (1/n)*(np.dot((-y).T , np.log(H + e))-np.dot((1-y).T , np.log(1 - H + e)))[0]

    def sigmoid(self,x):
        '''
        returns sigmoid(x)
        '''
        return 1 / (1 + np.exp(-x))

    def info(self):
        '''
        returns model details
        '''
        return [self.accuracy,self.error,self.alpha,self.epoch]