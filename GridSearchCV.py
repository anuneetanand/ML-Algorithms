# Anuneet Anand

import h5py
import pickle 
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import train_test_split

np.random.seed(0)

class MyGridSearchCV():
	"""
	My implementation of GridSearchCV.
	"""
	def __init__(self,Estimator,Parameters,Folds=4,verbose=0):
		self.Estimator = Estimator
		self.Parameters = Parameters
		self.Folds = Folds
		self.Combinations = []
		self.Best_Model = ""
		self.Best_Result = []
		self.Training_Accuracy = []
		self.Validation_Accuracy = []
		self.verbose = verbose

	def fit(self,X,Y):
		"""
			Fit Estimator with all sets of parameters

			Parameters
			----------
			X : 2-dimensional numpy array of shape (n_samples, n_features) 
			Y : 1-dimensional numpy array of shape (n_samples,)
			
			Returns
			-------
			self : an instance of self
		"""

		self.parse()

		print("-------------------------------------------------------------------------------------------")
		print("Fitting "+str(len(self.Combinations))+" combinations of parameters on "+str(self.Folds)+" Folds")
		print("-------------------------------------------------------------------------------------------")

		n = Y.shape[0]
		d = n//self.Folds
		a = 0

		for p in self.Combinations:
			
			if self.verbose == 1:
				print("-------------------------------------------------------------------------------------------")
				print("[>] Parameters: "+str(p))
				print("-------------------------------------------------------------------------------------------")
			at = 0
			av = 0
			for i in range(self.Folds):
				
				X_train = np.concatenate((X[:i*d],X[i*d+d:]))
				Y_train = np.concatenate((Y[:i*d],Y[i*d+d:]))
				X_val = X[i*d:i*d+d]
				Y_val = Y[i*d:i*d+d]
		
				self.Estimator.set_params(**p)
				self.Estimator.fit(X_train,Y_train)
				t = self.Estimator.score(X_train,Y_train)
				v = self.Estimator.score(X_val,Y_val)
				s = v
				if s>a:
					a = s
					self.Best_Result = [p,i+1,s]
					self.Best_Model = self.Estimator
					with open("Model.pkl", "wb") as file:  pickle.dump(CV.Best_Model,file)
				if self.verbose == 1: print("::> Fold: "+str(i+1)+" Score: "+str(s))
				at+=t
				av+=v
			self.Training_Accuracy.append(at/self.Folds)
			self.Validation_Accuracy.append(av/self.Folds)
	
		with open("Model.pkl", "rb") as file:  self.Best_Model = pickle.load(file)
		print("----------------------------------------------------------------------------------------------------------------------------------")
		print("[o] Best Result :> "+"Parameters: "+str(self.Best_Result[0])+" Fold: "+str(self.Best_Result[1])+" Score: "+str(self.Best_Result[2]))
		print("----------------------------------------------------------------------------------------------------------------------------------")

	def predict(self,X):
		"""
			Predicting values using the Best Estimator.

			Parameters
			----------
			X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

			Returns
			-------
			Y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted labels.
		"""
		Y = self.Best_Model.predict(X)
		return Y

	def parse(self):
		"""
			Generates all possible combinations of parameters.
		"""
		K = [i for i in self.Parameters]
		X = [self.Parameters[i] for i in K]
		P = list(product(*X))
		for s in P:
			D = {}
			for i in range(len(K)):
				D[K[i]]=s[i]
			self.Combinations.append(D)

	def plot(self):
		"""
			Plots Training & Validation Accuracy
		"""
		plt.figure()
		plt.title("Accuracy Vs. Max Depth")
		plt.xlabel("Max Depth")
		plt.ylabel("Accuracy")
		plt.plot([i+1 for i in range(len(self.Training_Accuracy))],self.Training_Accuracy, label = "Training Accuracy")
		plt.plot([i+1 for i in range(len(self.Validation_Accuracy))],self.Validation_Accuracy, label = "Validation Accuracy")
		plt.legend()
