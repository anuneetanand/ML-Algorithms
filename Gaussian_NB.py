# Anuneet Anand

import h5py 
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.model_selection import train_test_split

delta = 10**(-64)
np.random.seed(0)

class MyGaussianNB():
	"""
	My implementation of Guassian Naive Bayes.
	"""
	def __init__(self):
		self.Size = ()
		self.Std = {}
		self.Mean = {}
		self.Prob = {}
		self.Classes = {}

	def fit(self,X,Y):
		"""
			Fitting (training) the Classifier.

			Parameters
			----------
			X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.
			Y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
			
			Returns
			-------
			self : an instance of self
		"""
		self.Size=X.shape
		self.Classes = {i:[] for i in np.unique(Y)}

		# Partitioning dataset based on class labels
		for i in range(self.Size[0]):
			self.Classes[Y[i]].append(X[i])

		# Calculating Priori Probabilities
		for c in self.Classes:
			self.Prob[c]=np.log(len(self.Classes[c])/self.Size[0])

		# Calculating Mean & Standard Deviation for each Feature in a given class
		for c in self.Classes:
			self.Classes[c]=np.array(self.Classes[c])
			self.Mean[c]=np.mean(self.Classes[c],axis=0)
			self.Std[c]=np.std(self.Classes[c],axis=0)


	def predict(self,X):
		"""
			Predicting values using the Classifier.

			Parameters
			----------
			X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

			Returns
			-------
			Y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted labels.
		"""

		Y = []
		for x in X:
			for c in self.Classes:
				self.Prob[c]=np.log(len(self.Classes[c])/self.Size[0])

				# Calculating Posteriori Probabilities
				for i in range(self.Size[1]):
					p = np.log(self.GD(x[i],self.Mean[c][i],self.Std[c][i]))
					self.Prob[c]+=p

			# Predicting Class label based on Posteriori Probabilities
			mp = max(list(self.Prob.values()))
			for c in self.Prob:
				if self.Prob[c]==mp:
					Y.append(c)
					break

		Y=np.array(Y)
		Y.reshape((len(Y),))
		return Y

	def GD(self,x,u,s):
		"""
			Calculates Gaussian Probability

			Parameters
			----------
			x : Observed value
			u : Mean
			s : Standard Deviation

			Returns
			-------
			p : Guassian Probability
		"""

		if s==0:
			if x==u:return 1
			else:return delta
		e = np.exp(-((x - u) ** 2 / (2 * s ** 2)))
		p = (1 / (np.sqrt(2 * np.pi) * s)) * e
		#p = sp.stats.norm(u,s).pdf(x)
		return min(max(p,delta),1)

	def score(self,Y_pred,Y_test):
		"""
			Calculates accuracy score

			Parameters
			----------
			Y_pred : 1-dimensional numpy array of shape (n_samples,) which contains the predicted labels.
			Y_test : 1-dimensional numpy array of shape (n_samples,) which contains the original labels.

			Returns
			-------
			a : Accuracy
		"""
		c = 0
		n = len(Y_test)
		for i in range(n):
			c+= int(Y_pred[i]==Y_test[i])
		a = c/n
		return a
