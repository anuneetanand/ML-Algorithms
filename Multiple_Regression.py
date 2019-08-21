#Anuneet Anand
#2018022
#A-6

import csv
import numpy as np

Titles=[]												#Name Of Journals
Titles_IF={}
SJR=[]													#Scientific Journal Ranking
H_Index=[]												#H_Index
Doc_2017=[]												#Docs In 2017
Docs=[]													#Docs Of Past 3yrs
Refs=[]													#References
Cites=[]												#Total Cites Of Past 3yrs
Citable_Docs=[]											#Total Citable Docs Of Past 3yrs
Cites_per_Doc=[]										#Cites/Docs For Past 2yrs
Refs_per_Doc=[]											#References/Docs
Impact_Factors=[]                 						#Impact Factors Of Journals
Mean_Absolute_Error={}
Mean_Squared_Error={}
Beta={}
Index={}

#Reading Journals' Data
f=-1
with open('Impact_Factors_Journals.csv','r') as ImpactFactorsFile:
	reader = csv.reader(ImpactFactorsFile,delimiter=",")
	for data in reader:
		if(f!=-1):
			Titles_IF[(((((((data[1]).lower()).strip()).replace(" ","")).replace(",","")).replace("-","")).replace(":",""))]=float(data[3])
		f=f+1
ImpactFactorsFile.close()

f=-1
with open('Journals-Raw.csv','r') as JournalsFile:
	reader = csv.reader(JournalsFile,delimiter=";")
	for data in reader:
		if(f==-1):
			f=0
		elif "," in data[5] and ((((((data[2]).lower()).strip()).replace(" ","")).replace(",","")).replace("-","")).replace(":","") in Titles_IF:
			Titles.append(data[2])
			Impact_Factors.append(Titles_IF[((((((data[2]).lower()).strip()).replace(" ","")).replace(",","")).replace("-","")).replace(":","")])
			SJR.append(float(str(data[5]).replace(",",".")))
			H_Index.append(float(data[7]))
			Doc_2017.append(float(data[8]))
			Docs.append(float(data[9]))
			Refs.append(float(data[10]))
			Cites.append(float(data[11]))
			Citable_Docs.append(float(data[12]))
			Cites_per_Doc.append(float(str(data[13]).replace(",",".")))
			Refs_per_Doc.append(float(str(data[14]).replace(",",".")))
			f=f+1
JournalsFile.close()

with open('Error.csv', 'w') as csvFile:
	writer = csv.writer(csvFile)
	writer.writerow(["S.No.","Combination Of Parameters","Mean Absolute Error","Mean Squared Error"])
csvFile.close()

#Decimal To Binary Converter
def Bin(x):
	s=""
	for i in range(9):
		s=str(x%2)+s
		x=x//2
	return s

#To Generate Input Matrices
def GM(s,L,U):
	M=[]                                 					#Training Data
	K=[]								 					#Testing Data
	c=[]								 					#Combination
	M.append([1.0 for i in range(L)])           			#Column Of 1 For Constants
	K.append([1.0 for i in range(L,U)])
	if(s[0]=='1'):
		M.append(SJR[:L])
		K.append(SJR[L:U])
		c.append("SJR")
	if(s[1]=='1'):
		M.append(H_Index[:L])
		K.append(H_Index[L:U])
		c.append("H_Index")
	if(s[2]=='1'):
		M.append(Doc_2017[:L])
		K.append(Doc_2017[L:U])
		c.append("Doc_2017")
	if(s[3]=='1'):
		M.append(Docs[:L])
		K.append(Docs[L:U])
		c.append("Docs")
	if(s[4]=='1'):
		M.append(Refs[:L])
		K.append(Refs[L:U])
		c.append("Refs")
	if(s[5]=='1'):
		M.append(Cites[:L])
		K.append(Cites[L:U])
		c.append("Cites")
	if(s[6]=='1'):
		M.append(Citable_Docs[:L])
		K.append(Citable_Docs[L:U])
		c.append("Citable_Docs")
	if(s[7]=='1'):
		M.append(Cites_per_Doc[:L])
		K.append(Cites_per_Doc[L:U])
		c.append("Cites_per_Doc")
	if(s[8]=='1'):
		M.append(Refs_per_Doc[:L])
		K.append(Refs_per_Doc[L:U])
		c.append("Refs_per_Doc")
	c.sort()
	return np.transpose(np.array(M)),np.transpose(np.array(K)),c

#Training and Testing Regression Model
for i in range(1,256):
	X,W,N=GM(Bin(i),496,620)                              		#Generating Input Matrix
	Y=np.transpose(np.array(Impact_Factors[:496]))
	Xt=np.transpose(X)
	XtX=np.dot(Xt,X)
	XtY=np.dot(Xt,Y)
	B=np.linalg.solve(XtX,XtY)									#Finding Beta Matrix From Training Data
	V=np.transpose(np.array(Impact_Factors[496:620]))
	P=np.dot(W,B)												#Predicting Impact Factors On Test Data
	AE=abs(P-V)
	MAE=sum(AE)/len(V)											#Mean Absolute Error
	MSE=sum([i**2 for i in AE])/len(V)							#Mean Squared Error
	
	n=N[0]
	for j in N[1:]:
		n=n+", "+j
	Mean_Absolute_Error[n]=MAE         
	Mean_Squared_Error[n]=MSE
	Beta[n]=B
	Index[n]=i
	OUTPUT=[i,n,MAE,MSE]										#Writing Error & Combo Onto File
	
	with open('Error.csv', 'a') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerow(OUTPUT)
	csvFile.close()

#Reporting Minimum Error
print("Finished Processing Dataset Of Size:",len(Titles))
for Combo,Error in Mean_Absolute_Error.items():
	if Error==min(list(Mean_Absolute_Error.values())):
		BC1=Combo
		print("Minimum Mean Absolute Error:",Error,"With Combo:",Combo)

for Combo,Error in Mean_Squared_Error.items():
	if Error==min(list(Mean_Squared_Error.values())):
		BC2=Combo
		print("Minimum Mean Squared Error:",Error,"With Combo:",Combo)

#Working On Conference Data
Titles=[]												#Name Of Conferences
SJR=[]													#Scientific Journal Ranking
H_Index=[]												#H_Index
Doc_2017=[]												#Docs In 2017
Docs=[]													#Docs Of Past 3yrs
Refs=[]													#References
Cites=[]												#Total Cites Of Past 3yrs
Citable_Docs=[]											#Total Citable Docs Of Past 3yrs
Cites_per_Doc=[]										#Cites/Docs For Past 2yrs
Refs_per_Doc=[]											#References/Docs
Impact_Factors=[]                 						#Impact Factors Of Conferences

f=-1
with open('Conferences-Raw.csv','r') as ConferencesFile:
	reader = csv.reader(ConferencesFile,delimiter=";")
	for data in reader:
		if(f==-1):
			f=0
		elif "," in data[5]:
			Titles.append(data[2])
			SJR.append(float(str(data[5]).replace(",",".")))
			H_Index.append(float(data[7]))
			Doc_2017.append(float(data[8]))
			Docs.append(float(data[9]))
			Refs.append(float(data[10]))
			Cites.append(float(data[11]))
			Citable_Docs.append(float(data[12]))
			Cites_per_Doc.append(float(str(data[13]).replace(",",".")))
			Refs_per_Doc.append(float(str(data[14]).replace(",",".")))
			f=f+1
ConferencesFile.close()

with open('Impact_Factors_Conferences.csv', 'w') as csvFile:
	writer = csv.writer(csvFile)
	writer.writerow(["Title","Impact_Factor_Combo_1","Impact_Factor_Combo_2"])
csvFile.close()

X1,W,N=GM(Bin(Index[BC1]),len(Titles),len(Titles))
Y1=np.dot(X1,Beta[BC1])
X2,W,N=GM(Bin(Index[BC2]),len(Titles),len(Titles))
Y2=np.dot(X2,Beta[BC2])

#Writing Impact Factors Of Conferences Onto File
for i in range(len(Titles)):
	with open('Impact_Factors_Conferences.csv', 'a') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerow([Titles[i],float(Y1[i]),float(Y2[i])])
	csvFile.close()
print("Predicted Impact Factors For",len(Titles),"Conferences Using Best Model(s)")

#END

'''
The numpy functions were used as user defined functions make execution of script extremely slow.

#To Multiply Matrices
def M(A,B):
	C=[[0 for i in range(len(B[0]))] for j in range(len(A))]
	for i in range(len(A)):
		for j in range(len(B[0])):
			for k in range(len(A[0])):
				C[i][j]=C[i][j]+A[i][k]*B[k][j]
	return C

#To Find Transpose Of Matrix
def T(A):
	B=[[0 for i in range(len(A))] for j in range(len(A[0]))]
	for i in range(len(A)):
		for j in range(len(A[0])):
			B[j][i]=A[i][j]
	return B

#To Find Determinant Of Matrix
def D(A):
	if(len(A)==1):
		return A[0][0]
	if(len(A)==2):
		return A[0][0]*A[1][1]-A[0][1]*A[1][0]
	d=0
	for i in range(len(A)):
		d=d+((-1)**i)*A[0][i]*D([r[:i]+r[i+1:] for r in (A[:0]+A[1:])])
	return d

#To Find Inverse Of Matrix
def I(A):
	C=[]
	for i in range(len(A)):
		CR=[]
		for j in range(len(A)):
			CR.append(((-1)**(i+j))*D([r[:j]+r[j+1:] for r in (A[:i]+A[i+1:])]))
		C.append(CR)
	I=[[C[i][j]/D(C) for i in range(len(C))] for j in range(len(C))]
	return I
'''
