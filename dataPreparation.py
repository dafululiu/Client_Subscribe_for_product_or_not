import numpy as np
import csv
from sklearn.model_selection import train_test_split
class MarketingData:
	'''
	Marketing Data class
	'''
	labels=0
	data=0
	def oneHot(self,arr):
		'''
		Changing the string values to corresponding integer(s)
		'''
		#print arr.shape
		#print "SHAPE FOR ONE-HOT: ",arr.shape[1]
		colHavingNumbers=[0,5,9,11,12,13,14]
		for i in colHavingNumbers:	
			#print "**"
			#print i
			#extractedData = arr[:,i]
			#v1= set(extractedData)
			#COL No.0,5,9,11,12,13,14
			#print "Current Val of ",i
			for j in range(arr.shape[0]):
				#print j,i
				#print arr[:,i][j]
				arr[:,i][j] =  arr[:,i][j].astype(np.int)
				
		colNotHavingNumbers=[1,2,3,4,6,7,8,10,15,16]	
		for i in colNotHavingNumbers:
			#print "//"
			#print i
			#print arr[:,i]
			#for j in range(arr.shape[0]):
			values=list(set(arr[:,i]))
			#print values
			val=np.arange(len(values))
			#print val
			#print arr[:,i][0]
			#print type(values[0])
			for j in range(arr.shape[0]):
				for k in range(len(values)):
					#print "K :",k
					if(arr[:,i][j] == values[k]):
						#print 
						arr[:,i][j]=val[k]
						arr[:,i][j] =  arr[:,i][j].astype(np.int)
						#pass
		"""
		for k in range(arr.shape[0]):
			if(arr[:,13][k].astype(np.int) == -1):
				arr[:,13][k] = 0
			if(arr[:,5][k].astype(np.int) < 0):
				arr[:,5][k] = abs(arr[:,5][k].astype(np.int))
		"""		
		#print "OVER"
		#print arr
		return arr

	def dataPrep(self,filename):
		'''
		Read a csv file having labels in last column and data in remaining columns
		'''
		self.callDash(100);
		#print "Reading file",filename
		arr=np.genfromtxt(filename,delimiter=',',dtype=None)
		arr=arr[1:]
		#Before sending change string to one-hot values

		changedArr=self.oneHot(arr)
		data=changedArr[:,:-1]
		labels=changedArr[:,-1]
		print "Loaded :",len(labels), "rows of data from ",filename
		self.callDash(100);
		return data,labels

	def __init__(self):
		'''
		Marketing Data Class constructor
		'''
		#print "Inside Marketing Data"
		pass

	def shuffleAndSplit(self,data,labels):
		'''
		Shuffle the dataset and split the data with 80%,20% for Train and Test data
		'''
		self.callDash(100)
		x_train, x_test, y_train, y_test = train_test_split(data,labels, test_size=0.2)
		#x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)
		print "::::Splitted with size::::"
		print "TRAIN: ",round(len(y_train)/float(len(data))*100 )," %"
		print "TEST: ",round(len(y_test)/float(len(data))*100 )," %"
		#print "VALIDATION: ",int(len(y_val)/float(len(data))*100 )," %"
		self.callDash(100)
		return x_train, x_test,  y_train,y_test

	def callDash(self,value):
		'''
		Dummy function to print ----------------
		'''
		print "-"*value

