# put this python source code on the main folder of the dataset
# command line scripts: "python3 thisfilename.py <modelType> <kernel> <gamma>"
# constraints (1): <modelType> should be either "SVC", "SVR", "NuSVC", "NuSVR", "LinearSVC", "LinearSVR"
# constraints (2): <kernel> should be either "linear", "poly", "rbf" or "sigmoid"
# constraints (3): <gamma> should be either "auto" or "scale"
# <kernel> and <gamma> will be ignored (still being checked) when using LinearSVC

from sys import argv
import os, time, sys, psutil
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from glob import glob

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# logs initialization
if not 'logs/' in glob('*/'):
	os.mkdir("logs/")
logname = "logs/logs-" + str(int(time.time() // 1))
for arg in argv:
	logname = logname + '~' + arg
logname = logname + '.txt'
global logfile; logfile = None

# redefine "print" function to write in both stdout and logs
def write(str, end='\n', flush=False):
	print(str, end=end, flush=flush)
	logfile.write(str+end)

# exception handling
def filteringException():
	if len(argv) != 4:
		# incorrect arguments count
		write('Incorrect format!')
		write('Valid format: "python3 thisfilename.py <modelType> <kernel> <gamma>"')
		sys.exit(-1)
	else:
		# <modelType>
		if argv[1] != 'SVC' and argv[1] != 'NuSVC' and argv[1] != 'LinearSVC' and argv[1] != 'SVR' and argv[1] != 'NuSVR' and argv[1] != 'LinearSVR':
			write('<modelType> should be either "SVC", "SVR", "NuSVC", "NuSVR", "LinearSVC", "LinearSVR"!')
			sys.exit(-41)
		# <kernel>
		if argv[2] != 'linear' and argv[2] != 'poly' and argv[2] != 'rbf' and argv[2] != 'sigmoid':
			write('<kernel> should be either "linear", "poly", "rbf" or "sigmoid"!')
			sys.exit(-42)
		# <gamma>
		if argv[3] != 'auto' and argv[3] != 'scale':
			write('<gamma> should be either "auto" or "scale"!')
			sys.exit(-43)

# processing arguments after surpassed all exception tests
def processArguments():
	modelType = argv[1]
	kernel = argv[2]; gamma = argv[3]

	return modelType, kernel, gamma

def readData():
	# initialize
	write('Begin loading from csv file...', flush=True)
	totalcnt = 0
	dataList = []; labelList = []
	startTime = time.time()

	# iterate all subfolders
	data = pd.read_csv(
		"train.csv",
		header=None,
		sep=",",
		quotechar="'"
	)
	for index, row in data.iterrows():
		totalcnt += 1
		featuresList = []
		label = -1
		for colLabel in data.columns:
			if colLabel == 39:
				label = int(row[colLabel])
			elif colLabel == 11 or colLabel == 13 or colLabel == 36 or colLabel == 37:
				featuresList.append(float(row[colLabel]))
			else: featuresList.append(int(row[colLabel]))
		dataList.append(featuresList)
		labelList.append(label)

	
	# finalize and return value
	endTime = time.time()
	write('Successfully loaded ' + str(totalcnt) + ' records.', flush=True)
	write('Elapsed time: ' + str(endTime - startTime) + ' seconds.', flush=True)
	del totalcnt, startTime, endTime
	return dataList, labelList

def train_and_predict(process, dataList, labelList, trainRatio):
	# initialize module
	data_train, data_test, label_train, label_test = train_test_split(dataList, labelList, train_size = trainRatio)

	startTime = time.time()
	SV_Module = None

	# parsing module by arguments
	if modelType == 'SVC':
		SV_Module = svm.SVC(kernel=kernel, gamma=gamma)
	elif modelType == 'NuSVC':
		SV_Module = svm.NuSVC(kernel=kernel, gamma=gamma)
	elif modelType == 'LinearSVC':
		SV_Module = svm.LinearSVC()

	# training modules
	SV_Module.fit(data_train, label_train)

	# perform prediction by built-in predict() function
	predictedLabel = SV_Module.predict(data_test)

	acc = accuracy_score(label_test, predictedLabel)

	del predictedLabel
	del data_train, label_train, data_test, label_test
	return acc

def findAccuracy(process, dataList, labelList, trainRatio):
	finalAcc = 0; Iterations = 8
	for i in range(Iterations):
		finalAcc += train_and_predict(process, dataList, labelList, trainRatio) / Iterations
	return finalAcc

# main function of this source code
def mainFunction(process, modelType, kernel, gamma):
	# reading training data
	data, labels = readData()

	# perform prediction
	acclist = []; ratiolist = []
	acclist.append(findAccuracy(process, data, labels, 1 / 2))
	ratiolist.append(1 / 2)
	for i in range(3, 8):
		acclist.append(findAccuracy(process, data, labels, 1 / i))
		ratiolist.append(1 / i)
		acclist.append(findAccuracy(process, data, labels, (i - 1) / i))
		ratiolist.append((i - 1) / i)
	acclist = np.array(acclist)
	avg = np.average(acclist)
	Min = np.min(acclist); MinArg = np.argmin(acclist)
	Max = np.max(acclist); MaxArg = np.argmax(acclist)
	write('Average accuracy = ' + str(avg) + '.', flush=True)
	write('Min accuracy = ' + str(Min) + ' at TrainRatio = ' + str(ratiolist[MinArg]) + '.', flush=True)
	write('Max accuracy = ' + str(Max) + ' at TrainRatio = ' + str(ratiolist[MaxArg]) + '.', flush=True)

if __name__ == "__main__":
	# initialize memory monitor
	this_process = psutil.Process(os.getpid())

	# initialize logfiles
	logfile = open(logname, 'w')
	logfile.write('Command line: python3 ')
	for arg in argv: logfile.write(arg + ' ')
	logfile.write('\n\n')

	# handling exceptions and arguments
	filteringException()
	modelType, kernel, gamma = processArguments()
	
	# main training
	mainFunction(this_process, modelType, kernel, gamma)

	# finish logging
	logfile.close()
	print('Logs saved into ' + logname + '.')