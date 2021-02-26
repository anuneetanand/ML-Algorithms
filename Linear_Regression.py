# Anuneet Anand

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

np.random.seed(0)

class MyLinearRegression():

    """
    My implementation of Linear Regression.
    """

    def __init__(self,loss_function="MAE",alpha=0.01,epoch=1000):
        self.theta = np.empty((0))
        self.loss_function = loss_function
        self.alpha = alpha
        self.epoch = epoch
        self.error = 0
        self.training_loss = []
        self.validation_loss = []

    def normal_fit(self,X,y,X_test):
        '''
        - It uses normal equation to calculate the optimal parameters.
        - Returns the values predicted using optimal parameters on X_test.
        '''
        Optimal_Theta = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(y)
        return np.dot(X_test,Optimal_Theta)

    def fit(self, X, y):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """
        n = X.shape[0]
        self.theta = np.zeros((X.shape[1],1))                                   # Initialising parameters as 0
        y = y.reshape(y.shape[0],1)

        if self.loss_function=="RMSE":                                          # Gradient Descent for RMSE
            for i in range(self.epoch):
                H = np.dot(X,self.theta)
                D = ((1/n)**0.5) * (1/(sum(np.square(H-y))**0.5)) * np.dot(X.T,H-y)
                self.theta = self.theta - self.alpha*D     

        elif self.loss_function=="MAE":                                         # Gradient Descent for MAE
            for i in range(self.epoch):
                H = np.dot(X,self.theta)
                D = (1/n) * np.dot(X.T,np.sign(H-y))
                self.theta = self.theta - self.alpha*D 

        # fit function has to return an instance of itself or else it won't work with test.py
        return self

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array y which contains the predicted values

        y = np.dot(X,self.theta)

        return y

    def evaluate(self,X_train,y_train,X_test,y_test,all=False,plot=False):
        '''
        - It calculates the appropriate error for the model on Test set
        - If all is specified as True, it calculates error for each epoch.
        - If Plot is specified as True, it plots the training and validation loss.
        - Returns error of model (MAE/RMSE)
        '''
        if plot == True : all = True

        e = self.epoch
        n = y_train.shape[0]
        self.training_loss = []                                
        self.validation_loss = []
        iterations = [i for i in range(e)]
        y_train = y_train.reshape((y_train.shape[0],1))
        y_test = y_test.reshape((y_test.shape[0],1))


        if all == False:
            self.error = self.MAE(self.predict(X_test),y_test)
            return self.error

        if self.loss_function == "RMSE":

            self.theta = np.zeros((X_train.shape[1],1))
            for i in range(e):                                                          # Gradient Descent

                H = np.dot(X_train,self.theta)
                D = ((1/n)**0.5) * (1/(sum(np.square(H-y_train))**0.5)) * np.dot(X_train.T,H-y_train)
                self.theta = self.theta - self.alpha*D

                y_pred = np.dot(X_train,self.theta)
                self.training_loss.append(self.RMSE(y_pred,y_train))
                y_pred = np.dot(X_test,self.theta)
                self.validation_loss.append(self.RMSE(y_pred,y_test))

                if i == self.epoch : self.error = self.MAE(self.predict(X_test),y_test)

            if plot == True:

                plt.figure()
                plt.title("Linear Regression [Loss Function : RMSE, alpha :"+str(self.alpha)+" ]")
                plt.xlabel("Epoch")
                plt.ylabel("RMSE")
                plt.plot(iterations,self.training_loss, label = "Training Loss")
                plt.plot(iterations,self.validation_loss, label = "Validation Loss")
                plt.legend()
                plt.show()

        elif self.loss_function == "MAE":                                               # Gradient Descent

            self.theta = np.zeros((X_train.shape[1],1))
            for i in range(e):

                H = np.dot(X_train,self.theta)
                D = (1/n) * np.dot(X_train.T,np.sign(H-y_train))
                self.theta = self.theta - self.alpha*D 

                y_pred = np.dot(X_train,self.theta)
                self.training_loss.append(self.MAE(y_pred,y_train))
                y_pred = np.dot(X_test,self.theta)
                self.validation_loss.append(self.MAE(y_pred,y_test))

                if i == self.epoch : self.error = self.MAE(self.predict(X_test),y_test)

            if plot == True:

                plt.figure()
                plt.title("Linear Regression [Loss Function : MAE, alpha :"+str(self.alpha)+" ]")
                plt.xlabel("Epoch")
                plt.ylabel("MAE")
                plt.plot(iterations,self.training_loss, label = "Training Loss")
                plt.plot(iterations,self.validation_loss, label = "Validation Loss")
                plt.legend()
                plt.show()

        return self.error

    def RMSE(self,y_pred,y_true):
        '''
        returns RMSE
        '''
        return ((sum(np.square(y_pred-y_true))/y_true.shape[0])**0.5)[0]

    def MAE(self,y_pred,y_true):
        '''
        returns MAE
        '''
        return (sum(np.abs(y_pred-y_true))/y_true.shape[0])[0]

    def info(self):
        '''
        returns model details
        '''
        return [self.error,self.loss_function,self.alpha,self.epoch]
