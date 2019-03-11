# put this python source code on the main folder of the dataset
# command line scripts: "python3 thisfilename.py [trainingFolder] [testdataFolder]"
# constraints (2+3): [trainingFolder] and [testdataFolder] must exist

# data would be distributed as following:
# a "training" folder, consists of labelled images
# "training" folder has subfolders, each contains images of the same label, and the folder itself is named after that label
# a "testdata" folder, consists of unlabelled images, used to test the accuracy of the modules

from sys import argv
import os, time, sys

# exception handling
def filteringException():
    if len(argv) != 3:
        # incorrect arguments count
        print('Incorrect format!')
        print('Valid format: "python3 thisfilename.py [trainingFolder] [testdataFolder]"')
        sys.exit(-1)
    else:
        # folders not found
        if not os.path.isdir(argv[1]):
            print('Training folder "{}" not found!'.format(argv[1]))
            sys.exit(-4041)
        if not os.path.isdir(argv[2]):
            print('Testdata folder "{}" not found!'.format(argv[2]))
            sys.exit(-4042)

# processing arguments after surpassed all exception tests
def processArguments():
    # defining paths for data
    # would be glad if paths being of any OS' but Windows :)
    trainingFolder = argv[1] + '/'
    testdataFolder = argv[2] + '/'

    return trainingFolder, testdataFolder

# reading training data from training directories
def readImages_Training(trainingFolder):
    # initialize
    import cv2
    from glob import glob
    primalPath = trainingFolder
    subfolderList = sorted(glob(primalPath + '*/'))
    print('Begin loading from ' + primalPath + ' ...', flush=True)
    cntimg = 0; totalcnt = 0
    imgList = []; labelList = []
    startTime = time.time()

    # iterate all subfolders
    for path in subfolderList:
        id = path.replace(primalPath, '').replace('/', '')
        print('Begin loading from ' + path + ' ...', flush=True)
        cntimg = 0
        # iterate all images within subfolders
        for filename in os.listdir(path):
            imgList.append(cv2.imread(path + filename, 0).flatten())
            labelList.append(id)
            cntimg += 1; totalcnt += 1
            print('Loading sample image #' + str(cntimg) + ' from folder #' + str(id) + '...', flush=True)
    
    # finalize and return value
    endTime = time.time()
    print('Successfully loaded ' + str(totalcnt) + ' images.', flush=True)
    print('Elapsed time: ' + str(endTime - startTime) + ' seconds.', flush=True)
    return imgList, labelList

# reading test data from test directories
def readImages_TestData(testdataFolder):
    # initialize
    import cv2
    path = testdataFolder
    print('Begin loading from ' + path + ' ...', flush=True)
    cntimg = 0
    tmpList = []; fnameList = []
    startTime = time.time()

    # iterate all images
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
def prediction(module, csvFileName, testList, fnameList):
    # initialize
    import cv2
    csvOutput = open(csvFileName, 'w')
    csvOutput.write('ImageID,Label\n')
    cntimg = len(testList)
    startTime = time.time()

    # perform prediction by built-in predict() function
    print('Begin predicting...', flush=True)
    print('Writing target: ' + csvFileName + ' ...', flush=True)
    labelList = module.predict(testList)

    # writing prediction results into csv
    for i in range(cntimg):
        csvOutput.write(fnameList[i] + ',' + str(labelList[i]) + '\n')
    
    # finalize and close file output stream
    endTime = time.time()
    print('Successfully predicted ' + str(cntimg) + ' images.', flush=True)
    print('Elapsed time: ' + str(endTime - startTime) + ' seconds.', flush=True)
    csvOutput.close()

# main function of this source code
def mainFunction(trainingFolder, testdataFolder):
    # reading training data
    imgs, labels = readImages_Training(trainingFolder)

    # reading test data
    testimgs, testnames = readImages_TestData(testdataFolder)

    # initialize module
    print('Begin training using ' + str(len(imgs)) + ' images...', flush=True)
    startTime = time.time()
    from sklearn.tree import DecisionTreeClassifier
    DT_Module = DecisionTreeClassifier(criterion='entropy', splitter='best')
    DT_Module.fit(imgs, labels)
    endTime = time.time()
    print('Successfully fitted ' + str(len(imgs)) + ' images.', flush=True)
    print('Elapsed time: ' + str(endTime - startTime) + ' seconds.', flush=True)

    # initialize output csv
    csvResult = 'result.csv'

    #perform prediction
    prediction(DT_Module, csvResult, testimgs, testnames)

if __name__ == "__main__":
    filteringException()
    trainingFolder, testdataFolder = processArguments()
    mainFunction(trainingFolder, testdataFolder)
