#Anuneet Anand
#2018022
#A-6

import pandas 															#Used only for reading and writing csv file
import matplotlib.pyplot as plt

J = pandas.read_csv('Journals-HI-IF.txt',names=["Name Of Journal","H index","Impact Factor"],sep=";",usecols=[0,1,2])
J.index=J.index + 1
J.to_csv("Journals-HI-IF.csv")
T=J.sample(frac=0.8)													#Training Data
S=J[~J.index.isin(T.index)]												#Test Data
P=[]																	#Predicted Data
E=[]																	#Error

def Sd(X,m):
	'''Finding Standard Deviation'''
	z=0
	for i in X:
		z=z+i**2
	t=(z/len(X))-(m**2)
	return t**(0.5) 

def R(X,Y):
	'''Finding Correlation Coefficient'''
	s=0
	n=len(X)
	Xm=sum(X)/n 														#Mean of H indices
	Ym=sum(Y)/n 														#Mean of Impact Factors
	Xs=Sd(X,Xm) 														#Standard Deviation Of H indices
	Ys=Sd(Y,Ym)															#Standard Deviation of Impact Factors
	for i in range(n):
		s=s+(X[i]-Xm)*(Y[i]-Ym)
	r=s/(n*Xs*Ys)
	return r,Xm,Ym,Xs,Ys

def PIF(x,a,b):
	y=a*x+b 															#Regression Line
	return y 
									
r,Xm,Ym,Xs,Ys=R(T['H index'].values,T['Impact Factor'].values)			#Correlation Coefficient of Test Data
rt,Xmt,Ymt,Xst,Yst=R(J['H index'].values,J['Impact Factor'].values)

a=(r*(Ys/Xs))
b=(Ym-a*Xm)							

print("Correlation Coefficient Of Test Data:",r)
print("Correlation Coefficient Of Entire DataSet:",rt)
print("Regression Line:","y =",a,"x +",b)

#Working on Test Data
for i in range(len(S)):
	x=S['H index'].values[i]
	p=PIF(x,a,b)														#Predicting Data
	y=S['Impact Factor'].values[i]
	P.append(p)
	E.append(p-y)														#Calculating Error

s=0
for e in E:	
	s=s+(e)**2															#Sum Of Squares Of Errors	
Ems = (s/len(E))		
print("Mean Square Error:",Ems)

#Predicting Impact Factors Of Conferences
C = pandas.read_csv('Conferences-H-Index-Raw.csv',sep=';',usecols=['Title','H index'])
C.index=C.index + 1
CIF=['' for i in range(len(C))]
nc = pandas.DataFrame({'Impact Factor':CIF})
C = C.merge(nc, left_index = True, right_index = True)
C.to_csv('Conferences-H-Index.csv') 									

for i in range(len(C)):
	Xc=C['H index'].values[i]
	Yc=PIF(Xc,a,b)
	C['Impact Factor'].values[i]=Yc
	
C.to_csv('Conferences-HI-IF.csv')										#Writing the Impact Factor data onto file

#Plotting Journal Data and Regression Line
plt.xlabel("H index")
plt.ylabel("Impact Factor")
plt.title("Journals")
plt.plot(T['H index'].values,T['Impact Factor'].values,'g.',label="Training Data")
plt.plot(S['H index'].values,S['Impact Factor'].values,'b.',label="Test Data")
plt.plot(J['H index'].values,[PIF(x,a,b) for x in J['H index'].values],'r',label="Regression Line")
plt.legend()
plt.show()

#END

'''
#Merging H index and Impact Factor Data
I = pandas.read_csv('Journals-HI-IF.txt',names=["Name Of Journal","Impact Factor"],sep=";",usecols=[0,2])
H = pandas.read_csv('Journals-H-Index-Raw.csv',sep=";",usecols=["Title","H index"])

#Cleaning Data
I["Name Of Journal"]=I["Name Of Journal"].str.strip()
I["Name Of Journal"]=I["Name Of Journal"].str.replace(" ","")
I["Name Of Journal"]=I["Name Of Journal"].str.replace(",","")
I["Name Of Journal"]=I["Name Of Journal"].str.lower()
H["Title"]=H["Title"].str.strip()
H["Title"]=H["Title"].str.replace(" ","")
H["Title"]=H["Title"].str.replace(",","")
H["Title"]=H["Title"].str.lower()
H = H.rename(columns={"Title":"Name Of Journal"})
J=pandas.merge(H,I,on="Name Of Journal")	

L=len(J.index)
T=J.head(int(L*0.8))													#Training Data
S=J.tail(L-int(L*0.8))													#Test Data
r=T.corr(method='pearson').iat[0,1]	
'''
