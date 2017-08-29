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

def writeTocsv(data,labels):
	'''
	Function to write data to csv file
	'''
	#print data.shape,labels.shape
	lbl=np.reshape(labels , (data.shape[0],1))
	#print data.shape,lbl.shape
	finalData=np.c_[data,lbl]
	#print finalData.shape
	import pandas as pd 
	df = pd.DataFrame(finalData)
	df.to_csv("processed-marketing-data.csv")


def train(x_train, x_test, y_train,y_test):

	'''
	Function for Training the model using SVM Classifier
	'''

	print "Training ... Please wait ..."
	#print x_train.shape,y_train.shape
	#print x_test.shape
	
	"""
	#To Take only few data eg.80,20 for train and test respectively
	trainDataValue=80,testDataValue=20
	x_train_new= x_train[:trainDataValue].astype(np.int)
	y_train_new= y_train[:trainDataValue].astype(np.int)
	x_test_new= x_test[:testDataValue].astype(np.int)
	y_test_new=y_test[:testDataValue].astype(np.int)
	"""

	#Take Whole data
	x_train_new= x_train.astype(np.int)
	y_train_new= y_train.astype(np.int)
	x_test_new= x_test.astype(np.int)
	y_test_new=y_test.astype(np.int)
	
	"""
	#FEATURE SELECTION PHASE
	print "Feature Selection :"
	print "Importance of featues: "
	model = ExtraTreesClassifier()
	model.fit(x_train_new, y_train_new)
	print model.feature_importances_
	"""
	np.delete(x_train_new, 4, 1)	#deleting less important feature
	np.delete(x_test_new, 4, 1)	#deleting less important feature

	#print x_train_new.shape,y_train_new.shape,x_test_new.shape
	model = svm.SVC(kernel='linear').fit(x_train_new, y_train_new)
	model.score(x_train_new, y_train_new)
	#Predict Output
	predicted= model.predict(x_train_new)
	#print predicted
	#print y_train_new,y_test_new
	#print "Training with "
	#print np.count_nonzero(y_train_new == 1)," No data"
	#print np.count_nonzero(y_train_new == 0)," Yes data"
	#print "Testing with "
	#print np.count_nonzero(y_test_new == 1)," No data"
	#print np.count_nonzero(y_test_new == 0)," Yes data"
	#y_train_converted= y_train_new.astype(np.int)
	#print y_train_Ven.dtype

	joblib.dump(model, 'modelSVM.pkl') 
	print "Model Created..."


if __name__=="__main__":
	'''
	Main Function goes here
	'''
	data,labels=readcsv("marketing-data.csv") #Passing input file to read

	#writeTocsv(data,labels)	#Optional for data Visualization after converting data to one-hot values

	x_train, x_test, y_train,y_test=shuffleAndSplit(data,labels)	#Shuffle and Split the dataset

	train(x_train, x_test, y_train,y_test)		#Train with data

	


