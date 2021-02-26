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

def roc(Y_true,Y_prob):
	"""
		Plots ROC-Curve for Binary & Multi Class 

		Parameters
		----------
		Y_pred : 1-dimensional numpy array of shape (n_samples,) which contains the predicted labels.
		Y_prob : m-dimensional numpy array of shape (n_samples,m_features) which contains the probabilities of classes.

	"""

	n = Y_prob.shape[0]
	m = Y_prob.shape[1]

	plt.figure()	
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.plot([0, 1], [0, 1],'r--',label="Random")
	plt.xlim([0, 1])
	plt.ylim([0, 1])

	if m == 2:

		plt.title('Binary Class ROC')
		
		tpr = []
		fpr = []
		threshold = np.linspace(0,1,101)
		for x in threshold:
			tp=0
			fp=0
			tn=0
			fn=0
			for i in range(n):
				p = int(Y_prob[:,1][i]>x)
				o = Y_true[i]
				if p==1 and o==1:tp+=1
				elif p==1 and o==0:fp+=1
				elif p==0 and o==1:fn+=1
				elif p==0 and o==0:tn+=1
			tpr.append(tp/(tp+fn))
			fpr.append(fp/(fp+tn))

		P = [(0,0),(1,1)]
		for i in range(len(tpr)):
			P.append((fpr[i],tpr[i]))
		P.sort()
		x=[]
		y=[]
		for i in range(len(P)):
			x.append(P[i][0])
			y.append(P[i][1])

		plt.plot(x, y,label="ROC")
	
	else:
		# BONUS PART
		plt.title('Multi Class ROC')
		
		for c in range(m):
			tpr = []
			fpr = []
			threshold = np.linspace(0,1,101)
			for x in threshold:
				tp=0
				fp=0
				tn=0
				fn=0
				for i in range(n):
					p = int(Y_prob[:,c][i]>x)
					o = int(Y_true[i]==c)
					if p==1 and o==1:tp+=1
					elif p==1 and o==0:fp+=1
					elif p==0 and o==1:fn+=1
					elif p==0 and o==0:tn+=1
				tpr.append(tp/(tp+fn))
				fpr.append(fp/(fp+tn))

			P = [(0,0),(1,1)]
			for i in range(len(tpr)):
				P.append((fpr[i],tpr[i]))
			P.sort()
			x=[]
			y=[]
			for i in range(len(P)):
				x.append(P[i][0])
				y.append(P[i][1])

			plt.plot(x, y,label="Class "+str(c))

	plt.legend(loc="lower right")

def evaluation_metric(Y_true,Y_pred,Y_prob):
	"""
		Calculates Accuracy, Precision, Recall, F1-Score
		Plots ROC-Curve

		Parameters
		----------
		Y_pred : 1-dimensional numpy array of shape (n_samples,) which contains the predicted labels.
		Y_true : 1-dimensional numpy array of shape (n_samples,) which contains the actual labels.
		Y_prob : m-dimensional numpy array of shape (n_samples,m_features) which contains the probabilities of classes.

		Returns
		--------
		Confusion_Matrix : Confusion Matrix
	"""

	roc(Y_true,Y_prob)

	n = len(Y_pred)
	m = max(max(Y_true),max(Y_pred))+1

	Confusion_Matrix = [[0 for i in range(m)] for i in range(m)]
	for i in range(n):
		p = Y_pred[i]
		a = Y_true[i]
		Confusion_Matrix[p][a]+=1

	Accuracy = sum([Confusion_Matrix[i][i] for i in range(m)])/sum([sum(x) for x in Confusion_Matrix])

	Precision = [0 for i in range(m)]
	Recall = [0 for i in range(m)]
	F1_Score = [0 for i in range(m)]

	for i in range(m):
		Precision[i] = Confusion_Matrix[i][i]/sum([Confusion_Matrix[i][j] for j in range(m)])
		Recall[i] = Confusion_Matrix[i][i]/sum([Confusion_Matrix[j][i] for j in range(m)])
		F1_Score[i] = 2*(Precision[i]*Recall[i])/(Precision[i]+Recall[i])

	Metric = {}
	if m==2:
		Metric["Accuracy"]=Accuracy
		Metric["Precision"]=Precision[1]
		Metric["Recall"]=Recall[1]
		Metric["F1_Score"]=F1_Score[1]
	else:
		Metric["Accuracy"]=Accuracy
		Metric["Macro Precision"]=sum(Precision)/m
		Metric["Macro Recall"]=sum(Recall)/m
		Metric["Macro F1_Score"]=sum(F1_Score)/m
		Metric["Micro Precision"]=Accuracy
		Metric["Micro Recall"]=Accuracy
		Metric["Micro F1_Score"]=Accuracy
		# Metric["Weighted Precision"]=sum([(Y_true==i).sum()*Precision[i] for i in range(m)])/n
		# Metric["Weighted Recall"]=sum([(Y_true==i).sum()*Recall[i] for i in range(m)])/n
		# Metric["Weighted F1_Score"]=sum([(Y_true==i).sum()*F1_Score[i] for i in range(m)])/n

	D = Metric
	M = Confusion_Matrix
	print("----------------------------------------------------------------------------------------------------------------------------------")
	print("Evaluation Metrics")
	print("----------------------------------------------------------------------------------------------------------------------------------")
	for k in D:
		print(k+" : "+str(D[k]))
	print("----------------------------------------------------------------------------------------------------------------------------------")
	print("Confusion Matrix")
	print("----------------------------------------------------------------------------------------------------------------------------------")
	for x in M:
		print(*x)
	print("----------------------------------------------------------------------------------------------------------------------------------")

	return Confusion_Matrix
