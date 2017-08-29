import dataPreparation as dP
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.externals import joblib
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def readcsv(filename):
	'''
	Function to read a csv file
	'''
	objectOfMarketingData=dP.MarketingData()
	data,labels=objectOfMarketingData.dataPrep(filename)
	#print data,labels
	return data,labels

def shuffleAndSplit(data,labels):
	'''
	Function is to Shuffle and Split the dataset
	'''
        objectOfMarketingData=dP.MarketingData()
	x_train, x_test, y_train,y_test=objectOfMarketingData.shuffleAndSplit(data,labels)
	#print x_train, x_test, y_train,y_test
	return x_train.astype(np.int), x_test.astype(np.int),  y_train.astype(np.int),y_test.astype(np.int)




def testSingle(X,Y):

	'''
	Function to test a Single Random pre-processed Value
	'''

	#print X,Y," RECEIVED FOR REAL TIME TESTING..."
	clf = joblib.load('modelSVM_fullWorking.pkl')
	predicted=clf.predict(X.reshape(1,-1))
	#print "Predicted value: ",int(predicted[0])
	#print "Actual value : ",Y
	if(int(Y[0])==1):
		actual="No"
	elif(int(Y[0])==0):
		actual="Yes"
	else:
		actual="invalid"
	if(int(predicted[0]) == 0):
		#print "Predicted as Yes"
		predict="Yes"
	elif(int(predicted[0])  == 1):
		#print "Predicted as No"
		predict="No"
	else:
		predict="invalid"
		#print "Sorry Cant determine"
	print "Actual value is  :",actual
	print "And predicted as :",predict

	#return predict,actual

def testFull(X,Y):
	'''
	Function to test the Test data set
	'''
	#print X,Y," RECEIVED FOR REAL TIME TESTING..."
	clf = joblib.load('modelSVM_fullWorking.pkl')
	predicted=clf.predict(X)
	acc= accuracy_score(Y, predicted)
	print "Testing Accuracy is ",acc*100," %"
	results = confusion_matrix(Y, predicted)
	print "Confusion Matrix "	
	print results

if __name__=="__main__":
	'''
	Main Function goes here
	'''
	data,labels=readcsv("marketing-data.csv") #Passing input file to read

	#writeTocsv(data,labels)	#Optional for data Visualization after converting data to one-hot values

	x_train, x_test, y_train,y_test=shuffleAndSplit(data,labels)	#Shuffle and Split the dataset

	#train(x_train, x_test, y_train,y_test)		#Train with data

	testSingle(data[84],labels[84])	#Test with Single data

	testFull(x_test,y_test)		#Test with whole dataset


