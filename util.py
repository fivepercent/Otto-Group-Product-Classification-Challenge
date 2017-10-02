import pandas as pd
import numpy as np

import random

def loadTrainData():
	print('Loading training data...')
	train_df=pd.read_csv('data/train.csv')
	train_data=train_df.values
	randomIndex = [i for i in range(len(train_data))]
	random.shuffle(randomIndex)
	train_data_shuffle=train_data[randomIndex]
	x=train_data_shuffle[:,1:-1]
	y=train_data_shuffle[:,-1]
	return x,y

def loadTestData():
	print('Loading test data...')
	test_df=pd.read_csv('data/test.csv')
	test_data=test_df.values
	x=test_data[:,1:]
	return x

def evaluation(y_predict,y):
	logLoss=0
	m=y.shape[0]
	for i in range(m):
		cur_p=max(min(y_predict[i][y[i]-1], 1-10**(-15)), 10**(-15))
		logLoss -= np.log(cur_p)
	return logLoss/m

def writeOut(y_test_predict):
	print('Is writting result....')
	pd.DataFrame({"id": range(1, y_test_predict.shape[0]+1),
              "Class_1": y_test_predict[:,0],
              "Class_2": y_test_predict[:,1],
              "Class_3": y_test_predict[:,2],
              "Class_4": y_test_predict[:,3],
              "Class_5": y_test_predict[:,4],
              "Class_6": y_test_predict[:,5],
              "Class_7": y_test_predict[:,6],
              "Class_8": y_test_predict[:,7],
              "Class_9": y_test_predict[:,8],})[['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']].to_csv('RandomForest.csv', index=False, header=True)