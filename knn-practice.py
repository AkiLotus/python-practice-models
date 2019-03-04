# put this python source code on the main folder of the dataset
# modify mainPath, trainingFolder and testdataFolder to suit your own dataset's location
# modify L and R to limit the range of k in kNN modules being tested on

import time
L = 1; R = 1

mainPath = '/home/akilotus/AIF_Training/mnist_png/'
trainingFolder = 'training/'
testdataFolder = 'test_data/'

# reading training data from training directories
def readImages_Training():
    # initialize
    import cv2
    import os, os.path, time
    primalPath = mainPath + trainingFolder
    print('Begin loading from ' + primalPath + ' ...', flush=True)
    cntimg = 0
    imgList = []; labelList = []
    startTime = time.time()

    # iterate
    for id in range(10):
        path = primalPath + str(id) + '/'
        print('Begin loading from ' + path + ' ...', flush=True)
        cntimg = 0
        for filename in os.listdir(path):
            imgList.append(cv2.imread(path + filename, 0).flatten())
            labelList.append(id)
            cntimg += 1
            print('Loading test image #' + str(cntimg) + ' from folder #' + str(id) + '...', flush=True)
    
    # finalize and return value
    endTime = time.time()
    print('Successfully loaded ' + str(cntimg) + ' images.', flush=True)
    print('Elapsed time: ' + str(endTime - startTime) + ' seconds.', flush=True)
    return imgList, labelList

# reading test data from test directories
def readImages_TestData():
    # initialize
    import cv2
    import os, os.path
    path = mainPath + testdataFolder
    print('Begin loading from ' + path + ' ...', flush=True)
    cntimg = 0
    tmpList = []; fnameList = []
    startTime = time.time()

    # iterate
    for filename in os.listdir(path):
        tmpList.append(cv2.imread(path + filename, 0).flatten())
        fnameList.append(filename)
        cntimg += 1
        print('Loading test image #' + str(cntimg) + '...', flush=True)
    
    # finalize and return value
    endTime = time.time()
    print('Successfully loaded ' + str(cntimg) + ' images.', flush=True)
    print('Elapsed time: ' + str(endTime - startTime) + ' seconds.', flush=True)
    return tmpList, fnameList

# perform predictions with given modules, target csv file and list of images to be predicted
def prediction(knnModule, csvFileName, testList, fnameList):
    # initialize
    import cv2
    csvOutput = open(csvFileName, 'w')
    csvOutput.write('ImageID,Label\n')
    cntimg = len(testList)
    startTime = time.time()

    # perform prediction by built-in predict() function
    print('Begin predicting...', flush=True)
    print('Writing target: ' + csvFileName + ' ...', flush=True)
    labelList = knnModule.predict(testList)

    # writing prediction results into csv
    for i in range(cntimg):
        csvOutput.write(fnameList[i] + ',' + str(labelList[i]) + '\n')
    
    # finalize and close file output stream
    endTime = time.time()
    print('Successfully predicted ' + str(cntimg) + ' images.', flush=True)
    print('Elapsed time: ' + str(endTime - startTime) + ' seconds.', flush=True)
    csvOutput.close()

# reading training data
imgs, labels = readImages_Training()

# reading test data
testimgs, testnames = readImages_TestData()

# each iteration is a different k used in respective kNN training module
for k in range(L, R+1):
    # initialize module
    print('Begin working with k = ' + str(k) + '.', flush=True)
    startTime = time.time()
    from sklearn.neighbors import KNeighborsClassifier
    KNN_Module = KNeighborsClassifier(n_neighbors=k, algorithm="ball_tree")
    KNN_Module.fit(imgs, labels)
    endTime = time.time()
    print('Successfully fitted ' + str(len(imgs)) + ' images.', flush=True)
    print('Elapsed time: ' + str(endTime - startTime) + ' seconds.', flush=True)

    # initialize output csv
    csvResult = 'result'
    if k < 10:
        csvResult += '0'
    csvResult += str(k)
    csvResult += '.csv'

    #perform prediction
    prediction(KNN_Module, csvResult, testimgs, testnames)