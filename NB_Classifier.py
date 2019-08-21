#Anuneet Anand
#2018022
#A-6
#Assignment-3 : Naive Bayes Classifier

import csv

def chk(X,Y,Z): 												#To Count Favourable Cases
	for i in range(len(X)):
		if X[i]==Y[i] or X[i]=="*" or Y[i]=="*":
			Z[i]=Z[i]+1	
	return Z

#TIC-TAC-TOE
def A():
	'''
	Classes: Positive , Negative
	Number of Instances : 958
	Number Of Attributes : 9
	Training & Testing : Leave One Out Cross Validation
	'''
	Attributes=[]
	Class=[]
	Ac=0
	p=0

	with open("tic-tac-toe.data.txt",'r') as TTT: 				#Reading Data
		reader=TTT.readlines()
		t=0
		for lines in reader:
			lines=lines[:-1]
			data=lines.split(",")
			P=data[:9]
			Attributes.append(P)
			Class.append(data[9])
			if data[9]=="positive":
				p=p+1
			t=t+1
	TTT.close()

	for i in range(958):										
		Test=Attributes[i]
		Result=Class[i]
		Pos_Match=[0 for k in range(9)]
		Neg_Match=[0 for k in range(9)]
		w=p
		if Result=="positive":
			w=w-1
		j=0
		while j<958:											#Training & Testing
			if j!=i:
				if Class[j]=="positive":
					Pos_Match=chk(Attributes[j],Test,Pos_Match)
				else:
					Neg_Match=chk(Attributes[j],Test,Neg_Match)
			j=j+1

		Pos_Prob=w/957											#Predicting
		Neg_Prob=(957-w)/957
		for m in Pos_Match:
			Pos_Prob=Pos_Prob*(m/w)
		for m in Neg_Match:
			Neg_Prob=Neg_Prob*(m/(957-w))

		if Pos_Prob>Neg_Prob:
			Output="positive"
		else:
			Output="negative"

		if Output==Result:										
			Ac=Ac+1

	Ac=(Ac*100)/958
	return Ac

#SPECT HEART
def B():
	'''
	Classes: 0 , 1
	Number of Instances : 267
	Number Of Attributes : 22
	Training & Testing : Specified (Training-80,Testing-187)
	'''
	Attributes_TR=[]
	Class_TR=[]
	Attributes_TS=[]
	Class_TS=[]
	Ac=0
	pTR=0
	pTS=0

	with open("SPECT.train",'r') as STR: 						#Reading Training Data
		reader=STR.readlines()
		t=0
		for lines in reader:
			lines=lines[:-1]
			data=lines.split(",")
			P=data[1:23]
			Attributes_TR.append(P)
			Class_TR.append(data[0])
			if data[0]=="1":
				pTR=pTR+1
			t=t+1
	STR.close()	

	with open("SPECT.test",'r') as STS: 						#Reading Testing Data
		reader=STS.readlines()
		t=0
		for lines in reader:
			lines=lines[:-1]
			data=lines.split(",")
			P=data[1:23]
			Attributes_TS.append(P)
			Class_TS.append(data[0])
			if data[0]=="1":
				pTS=pTS+1
			t=t+1
	STS.close()	

	lTR=len(Class_TR)
	lTS=len(Class_TS)

	for i in range(lTS):
		Test=Attributes_TS[i]
		Result=Class_TS[i]
		Yes=[0 for k in range(22)]
		No=[0 for k in range(22)]

		j=0 													#Training & Testing
		while j<lTR:
			if Class_TR[j]=="1":
				Yes=chk(Attributes_TR[j],Test,Yes)
			else:
				No=chk(Attributes_TR[j],Test,No)
			j=j+1
		
		Yes_Prob=pTR/lTR 										#Predicting
		No_Prob=(lTR-pTR)/lTR

		for m in Yes:
			Yes_Prob=Yes_Prob*(m/pTR)
		for m in No:
			No_Prob=No_Prob*(m/(lTR-pTR))

		if Yes_Prob>No_Prob:
			Output="1"
		else:
			Output="0"

		if Output==Result:
			Ac=Ac+1

	Ac=(Ac*100)/lTS
	return Ac

#SOYBEAN (Small)
def C():
	'''
	Classes: D1,D2,D3,D4
	Number of Instances : 47
	Number Of Attributes : 35
	Training & Testing : Leave One Out Cross Validation
	'''
	Attributes=[]
	Class=[]
	Ac=0
	p1=0;p2=0;p3=0;p4=0;

	with open("soybean-small.data",'r') as SS: 					#Reading Data
		reader=SS.readlines()
		t=0
		for lines in reader:
			lines=lines[:-1]
			data=lines.split(",")
			P=data[:35]
			Attributes.append(P)
			Class.append(data[35])
			if data[35]=="D1":
				p1=p1+1
			if data[35]=="D2":
				p2=p2+1
			if data[35]=="D3":
				p3=p3+1
			if data[35]=="D4":
				p4=p4+1
			t=t+1
	SS.close()

	for i in range(47):
		Test=Attributes[i]
		Result=Class[i]
		D1_Match=[0 for k in range(35)]
		D2_Match=[0 for k in range(35)]
		D3_Match=[0 for k in range(35)]
		D4_Match=[0 for k in range(35)]

		j=0
		while j<47: 											#Training & Testing
			if j!=i:
				if Class[j]=="D1":
					D1_Match=chk(Attributes[j],Test,D1_Match)
				if Class[j]=="D2":
					D2_Match=chk(Attributes[j],Test,D2_Match)
				if Class[j]=="D3":
					D3_Match=chk(Attributes[j],Test,D3_Match)
				if Class[j]=="D4":
					D4_Match=chk(Attributes[j],Test,D4_Match)
			j=j+1

		D1_Prob=p1/47
		D2_Prob=p2/47
		D3_Prob=p3/47
		D4_Prob=p4/47
		for m in D1_Match: 										#Predicting
			D1_Prob=D1_Prob*(m/p1)
		for m in D2_Match:
			D2_Prob=D2_Prob*(m/p2)
		for m in D3_Match:
			D3_Prob=D3_Prob*(m/p3)
		for m in D4_Match:
			D4_Prob=D4_Prob*(m/p4)
		D={D1_Prob:"D1",D2_Prob:"D2",D3_Prob:"D3",D4_Prob:"D4"}
		M=max(D.keys())
		Output=D[M]
		#A check is placed for the case when all probabilities are equal.
		if Output==Result and not(D1_Prob==D2_Prob==D3_Prob==D4_Prob):
			Ac=Ac+1

	Ac=(Ac*100)/47
	return Ac

#SHUTTLE-LANDING
def D():
	'''
	Classes: No_Auto(1),Auto(2)
	Number Of Instance:15
	Number Of Attributes:6
	Training & Testing : Leave One Out Cross Validation
	'''
	Attributes=[]
	Class=[]
	Ac=0
	p=0

	with open("shuttle-landing-control.data.txt",'r') as SLC:	#Reading Data
		reader=SLC.readlines()
		t=0
		for lines in reader:
			lines=lines[:-1]
			data=lines.split(",")
			P=data[1:7]
			Attributes.append(P)
			Class.append(data[0])
			if data[0]=="1":
				p=p+1
			t=t+1
	SLC.close()

	for i in range(15):
		Test=Attributes[i]
		Result=Class[i]
		No_Auto=[0 for k in range(6)]
		Auto=[0 for k in range(6)]
		w=p
		if Result=="1":
			w=w-1
		j=0
		while j<15: 											#Training & Testing
			if j!=i:
				if Class[j]=="1":
					No_Auto=chk(Attributes[j],Test,No_Auto)
				else:
					Auto=chk(Attributes[j],Test,Auto)
			j=j+1

		No_Auto_Prob=w/14 										#Predicting
		Auto_Prob=(14-w)/14
		for m in No_Auto:
			No_Auto_Prob=No_Auto_Prob*(m/w)
		for m in Auto:
				Auto_Prob=Auto_Prob*(m/(14-w))

		if No_Auto_Prob>Auto_Prob:
			Output="1"
		else:
			Output="2"
		#A check is placed for the case when all probabilities are equal.
		if Output==Result and No_Auto_Prob!=Auto_Prob:
			Ac=Ac+1

	Ac=(Ac*100)/15
	return Ac

#MONK's PROBLEM
def E(monkstrain,monkstest):
	'''
	Classes:0,1
	Number Of Instances:432
	Number Of Attributes:7
	Training & Testing: 3 Sets
	'''
	Attributes_TR=[]
	Class_TR=[]
	Attributes_TS=[]
	Class_TS=[]
	Ac=0
	pTR=0
	pTS=0

	with open(monkstrain,'r') as MTR:						#Reading Training Data
		reader=MTR.readlines()
		t=0
		for lines in reader:
			lines=lines[1:-1]
			data=lines.split(" ")
			P=data[1:7]
			Attributes_TR.append(P)
			Class_TR.append(data[0])
			if data[0]=="1":
				pTR=pTR+1
			t=t+1
	MTR.close()	

	with open(monkstest,'r') as MTS:						#Reading Testing Data
		reader=MTS.readlines()
		t=0
		for lines in reader:
			lines=lines[1:-1]
			data=lines.split(" ")
			P=data[1:7]
			Attributes_TS.append(P)
			Class_TS.append(data[0])
			if data[0]=="1":
				pTS=pTS+1
			t=t+1
	MTS.close()	

	lTR=len(Class_TR)
	lTS=len(Class_TS)

	for i in range(lTS):
		Test=Attributes_TS[i]
		Result=Class_TS[i]
		Yes=[0 for k in range(6)]
		No=[0 for k in range(6)]

		j=0
		while j<lTR:										#Training & Testing
			if Class_TR[j]=="1":
				Yes=chk(Attributes_TR[j],Test,Yes)
			else:
				No=chk(Attributes_TR[j],Test,No)
			j=j+1
		
		Yes_Prob=pTR/lTR 									#Predicting
		No_Prob=(lTR-pTR)/lTR

		for m in Yes:
			Yes_Prob=Yes_Prob*(m/pTR)
		for m in No:
			No_Prob=No_Prob*(m/(lTR-pTR))

		if Yes_Prob>No_Prob:
			Output="1"
		else:
			Output="0"

		if Output==Result:
			Ac=Ac+1

	Ac=(Ac*100)/lTS
	return Ac

#OUTPUT:-

print("Tic-Tac-Toe Endgame Data Set Accuracy :",A())
print("SPECT Heart Data Set Accuracy :",B())
print("Soybean (Small) Data Set Accuracy :",C())
print("Shuttle Landing Control Data Set Accuracy :",D())
print("Monk's Problem Data Set 1 Accuracy :",E("monks-1.train","monks-1.test"))
print("Monk's Problem Data Set 2 Accuracy :",E("monks-2.train","monks-2.test"))
print("Monk's Problem Data Set 3 Accuracy :",E("monks-3.train","monks-3.test"))
print("Monk's Problem Data Set Average Accuracy :",(E("monks-1.train","monks-1.test")+E("monks-2.train","monks-2.test")+E("monks-3.train","monks-3.test"))/3)

with open("Result.csv",'w') as csvFile:
	writer = csv.writer(csvFile)
	writer.writerow(["DataSet","Accuracy"])
	writer.writerow(["Tic-Tac-Toe Endgame Data Set ",A()])
	writer.writerow(["SPECT Heart Data Set ",B()])
	writer.writerow(["Soybean (Small) Data Set ",C()])
	writer.writerow(["Shuttle Landing Control Data Set ",D()])
	writer.writerow(["Monk's Problems Data Set ",(E("monks-1.train","monks-1.test")+E("monks-2.train","monks-2.test")+E("monks-3.train","monks-3.test"))/3])
csvFile.close()