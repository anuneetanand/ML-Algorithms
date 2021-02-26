# Anuneet Anand

import copy
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

class MyNeuralNetwork():
	
	"""
	My implementation of a Neural Network Classifier.
	"""

	acti_fns = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax']
	weight_inits = ['zero', 'random', 'normal']

	def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs):
		"""
		Initializing a new MyNeuralNetwork object

		Parameters
		----------
		n_layers : int value specifying the number of layers

		layer_sizes : integer array of size n_layers specifying the number of nodes in each layer

		activation : string specifying the activation function to be used
					 possible inputs: relu, sigmoid, linear, tanh

		learning_rate : float value specifying the learning rate to be used

		weight_init : string specifying the weight initialization function to be used
					  possible inputs: zero, random, normal

		batch_size : int value specifying the batch size to be used

		num_epochs : int value specifying the number of num_epochs to be used
		"""

		if activation not in self.acti_fns:
			raise Exception('Incorrect Activation Function')

		if weight_init not in self.weight_inits:
			raise Exception('Incorrect Weight Initialization Function')
		
		self.n_layers = n_layers
		self.layer_sizes = layer_sizes
		self.activation = activation
		self.learning_rate = learning_rate
		self.weight_init = weight_init
		self.batch_size = batch_size
		self.num_epochs = num_epochs

		# Weight Initialisation
		if self.weight_init == 'zero':
			self.weights = [self.zero_init((self.layer_sizes[i-1],self.layer_sizes[i])) for i in range(1,self.n_layers)]
		elif self.weight_init == 'random':
			self.weights = [self.random_init((self.layer_sizes[i-1],self.layer_sizes[i])) for i in range(1,self.n_layers)]
		elif self.weight_init == 'normal':
			self.weights = [self.normal_init((self.layer_sizes[i-1],self.layer_sizes[i])) for i in range(1,self.n_layers)]

		# Bias Initialisation
		self.bias = [self.zero_init((1,self.layer_sizes[i])) for i in range(1,self.n_layers)]

		# For Plots
		self.plot = False
		self.X_Train = None
		self.Y_Train = None
		self.X_Test = None
		self.Y_Test = None
		self.last_hidden_layer = None
		self.training_error = []
		self.testing_error = []

	def relu(self, X):
		"""
		Calculating the ReLU activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""

		Y = np.maximum(X,0)
		return Y

	def relu_grad(self, X):
		"""
		Calculating the gradient of ReLU activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		G = np.maximum(0,np.sign(X))
		return G

	def sigmoid(self, X):
		"""
		Calculating the Sigmoid activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""

		Y = 1.0/(1.0 + np.exp(-X))
		return Y

	def sigmoid_grad(self, X):
		"""
		Calculating the gradient of Sigmoid activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		G = self.sigmoid(X)*(1-self.sigmoid(X))
		return G

	def linear(self, X):
		"""
		Calculating the Linear activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""

		Y = X
		return Y

	def linear_grad(self, X):
		"""
		Calculating the gradient of Linear activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		G = 1
		return G

	def tanh(self, X):
		"""
		Calculating the Tanh activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""

		#Y = (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))
		Y = np.tanh(X)
		return Y

	def tanh_grad(self, X):
		"""
		Calculating the gradient of Tanh activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		G = 1 - np.square(self.tanh(X))
		return G

	def softmax(self, X):
		"""
		Calculating the Softmax activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""

		e = np.exp(X - np.max(X))
		Y = e/e.sum(axis=-1,keepdims=True)
		return Y

	def softmax_grad(self, X):
		"""
		Calculating the gradient of Softmax activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		S = self.softmax(X).reshape(-1,1)
		G = np.diagflat(S) - np.dot(S, S.T)
		return G

	def zero_init(self, shape):
		"""
		Calculating the initial weights after Zero Activation for a particular layer

		Parameters
		----------
		shape : tuple specifying the shape of the layer for which weights have to be generated 

		Returns
		-------
		weight : 2-dimensional numpy array which contains the initial weights for the requested layer
		"""

		W = np.zeros(shape)
		return W

	def random_init(self, shape):
		"""
		Calculating the initial weights after Random Activation for a particular layer

		Parameters
		----------
		shape : tuple specifying the shape of the layer for which weights have to be generated 

		Returns
		-------
		weight : 2-dimensional numpy array which contains the initial weights for the requested layer
		"""
		W = np.random.randn(shape[0],shape[1])*0.01
		return W

	def normal_init(self, shape):
		"""
		Calculating the initial weights after Normal(0,1) Activation for a particular layer

		Parameters
		----------
		shape : tuple specifying the shape of the layer for which weights have to be generated 

		Returns
		-------
		weight : 2-dimensional numpy array which contains the initial weights for the requested layer
		"""
		W = np.random.normal(0,1,size=shape)*0.01
		return W

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

		# fit function has to return an instance of itself or else it won't work with test.py

		n = X.shape[0]

		# One Hot Encoding
		ohe = np.zeros((y.size, y.max()+1))
		ohe[np.arange(y.size),y] = 1
		Y = ohe

		for i in range(self.num_epochs):

			# For Plots
			if self.plot :
				self.training_error.append(self.loss(self.predict_proba(self.X_Train),self.Y_Train))
				self.testing_error.append(self.loss(self.predict_proba(self.X_Test),self.Y_Test))
			
			for s in range(0,n,self.batch_size):

				# Selecting Batch
				t = min(s+self.batch_size,n)
				x = X[s:t]
				m = x.shape[0]
				y_true = Y[s:t]

				P = []										# Previous Outputs
				A = []										# Previous Activated Outputs
				old_weights = copy.deepcopy(self.weights)

				## FORWARD PASS ##

				A.append(x)
				for j in range(self.n_layers-2):
					x = np.dot(x,self.weights[j]) + self.bias[j]

					if self.activation == 'relu':
						y = np.apply_along_axis(self.relu,0,x)
					elif self.activation == 'sigmoid':
						y = np.apply_along_axis(self.sigmoid,0,x)
					elif self.activation == 'linear':
						y = np.apply_along_axis(self.linear,0,x)
					elif self.activation == 'tanh':
						y = np.apply_along_axis(self.tanh,0,x)
					
					P.append(x)
					A.append(y)
					x = y

				# Last Layer
				x = np.dot(x,self.weights[self.n_layers-2]) + self.bias[self.n_layers-2]
				y = self.softmax(x)
				P.append(x)
				A.append(y)
				y_pred = y
				error = self.loss(y_true,y_pred)

				## BACK PROPOGATION ##

				# Finding Gradients for Last Layer
				dZ = y_pred - y_true
				dW = (A[self.n_layers-2].T).dot(dZ)/m 
				dB = np.sum(dZ, axis=0, keepdims=True)/m

				# Updating Weight for Last Layer
				self.weights[self.n_layers-2] = self.weights[self.n_layers-2] - self.learning_rate*dW
				self.bias[self.n_layers-2] = self.bias[self.n_layers-2] - self.learning_rate*dB

				dZ = dZ.dot(old_weights[self.n_layers-2].T)

				# Updating Weights of Other Layers
				for k in range(self.n_layers-3,-1,-1):

					if self.activation == 'relu':
						G = np.apply_along_axis(self.relu_grad,0,P[k])
					elif self.activation == 'sigmoid':
						G = np.apply_along_axis(self.sigmoid_grad,0,P[k])
					elif self.activation == 'linear':
						G = np.apply_along_axis(self.linear_grad,0,P[k])
					elif self.activation == 'tanh':
						G = np.apply_along_axis(self.tanh_grad,0,P[k])

					dZ = dZ * G
					dW = (A[k].T).dot(dZ)/m
					dB = np.sum(dZ, axis=0, keepdims=True)/m

					self.weights[k] = self.weights[k] - self.learning_rate*dW
					self.bias[k] = self.bias[k] - self.learning_rate*dB

					dZ = dZ.dot(old_weights[k].T)

		return self

	def predict_proba(self, X):
		"""
		Predicting probabilities using the trained linear model.

		Parameters
		----------
		X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

		Returns
		-------
		y : 2-dimensional numpy array of shape (n_samples, n_classes) which contains the 
			class wise prediction probabilities.
		"""

		# return the numpy array y which contains the predicted values
		
		n = X.shape[0]

		for i in range(self.n_layers-2):
			X = np.dot(X,self.weights[i]) + self.bias[i]
			
			if self.activation == 'relu':
				y = np.apply_along_axis(self.relu,0,X)
			elif self.activation == 'sigmoid':
				y = np.apply_along_axis(self.sigmoid,0,X)
			elif self.activation == 'linear':
				y = np.apply_along_axis(self.linear,0,X)
			elif self.activation == 'tanh':
				y = np.apply_along_axis(self.tanh,0,X)

			X = y

		self.last_hidden_layer = X
		y = np.dot(X,self.weights[self.n_layers-2]) + self.bias[self.n_layers-2]
		y = self.softmax(y)

		return y

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

		n = X.shape[0]

		for i in range(self.n_layers-2):
			X = np.dot(X,self.weights[i]) + self.bias[i]

			if self.activation == 'relu':
				y = np.apply_along_axis(self.relu,0,X)
			elif self.activation == 'sigmoid':
				y = np.apply_along_axis(self.sigmoid,0,X)
			elif self.activation == 'linear':
				y = np.apply_along_axis(self.linear,0,X)
			elif self.activation == 'tanh':
				y = np.apply_along_axis(self.tanh,0,X)

			X = y

		y = np.dot(X,self.weights[self.n_layers-2]) + self.bias[self.n_layers-2]
		y = self.softmax(y)
		y = np.argmax(y,axis=1)
		return y

	def score(self, X, y):
		"""
		Predicting values using the trained linear model.

		Parameters
		----------
		X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

		y : 1-dimensional numpy array of shape (n_samples,) which acts as testing labels.

		Returns
		-------
		acc : float value specifying the accuracy of the model on the provided testing set
		"""

		# return the numpy array y which contains the predicted values

		y_pred = self.predict(X)
		y_true = y
		n = len(y)

		c = 0
		for i in range(n):
			c += int(y_true[i]==y_pred[i])

		acc = c/n

		return acc

	def loss(self,y_true,y_pred):
		'''
		returns cross entropy loss
		'''
		L = -np.mean(y_true * np.log(y_pred + 1e-8))
		return L

	def plot_error(self,X_Train,Y_Train,X_Test,Y_Test):
		
		'''
		make necessary plots
		'''

		self.plot = True
		self.X_Train = X_Train
		self.Y_Train = Y_Train
		
		ohe = np.zeros((self.Y_Train.size, self.Y_Train.max()+1))
		ohe[np.arange(self.Y_Train.size),self.Y_Train] = 1
		self.Y_Train = ohe

		self.X_Test = X_Test
		self.Y_Test = Y_Test

		ohe = np.zeros((self.Y_Test.size, self.Y_Test.max()+1))
		ohe[np.arange(self.Y_Test.size),self.Y_Test] = 1
		self.Y_Test = ohe

		self.fit(X_Train,Y_Train)

		iterations = [i for i in range(self.num_epochs)]

		plt.figure()
		plt.title("Activation Function : "+str(self.activation))
		plt.xlabel("Epoch")
		plt.ylabel("Error")
		plt.plot(iterations,self.training_error, label = "Training Error")
		plt.plot(iterations,self.testing_error, label = "Testing Error")
		plt.legend()

	def LHL(self,X):
		'''
		returns last hidden layer
		'''
		self.predict_proba(X)
		return self.last_hidden_layer


