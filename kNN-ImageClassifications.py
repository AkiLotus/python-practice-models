# put this python source code on the main folder of the dataset
# command line scripts: "python3 thisfilename.py [trainingFolder] [testdataFolder] [minNeighbors] [maxNeighbors]"
# constraints (2+3): [trainingFolder] and [testdataFolder] must exist
# constraints (4+5): [minNeighbors] and [maxNeighbors] are integers. 1 <= minNeighbors <= maxNeighbors <= 100

# data would be distributed as following:
# a "training" folder, consists of labelled images
# "training" folder has subfolders, each contains images of the same label, and the folder itself is named after that label
# a "testdata" folder, consists of unlabelled images, used to test the accuracy of the modules

from sys import argv
import os, time, sys

# exception handling
def filteringException():
    if len(argv) != 5:
        # incorrect arguments count
        print('Incorrect format!')
        print('Valid format: "python3 thisfilename.py [trainingFolder] [testdataFolder] [minNeighbors] [maxNeighbors]"')
        sys.exit(-1)
    else:
        # folders not found
        if not os.path.isdir(argv[1]):
            print('Training folder "{}" not found!'.format(argv[1]))
            sys.exit(-4041)
        if not os.path.isdir(argv[2]):
            print('Testdata folder "{}" not found!'.format(argv[2]))
            sys.exit(-4042)

        # non-integers arguments at argv[3] and argv[4]
        try: int(argv[3])
        except ValueError as ex3:
            print('{} cannot be parsed into int: {}', argv[3], ex3)
            sys.exit(-43)
        try: int(argv[4])
        except ValueError as ex4:
            print('{} cannot be parsed into int: {}', argv[4], ex4)
            sys.exit(-44)
        
        # illegal integer limits
        Lf = int(argv[3]); Rt = int(argv[4])
        if Lf > Rt:
            print('Invalid limit: Lower bound {} exceeded upper bound {}!'.format(Lf, Rt))
            sys.exit(-4)
        if Lf < 1 or Lf > 100:
            print('Invalid lower bound {}: integers must be within range [1, 100]!'.format(Lf))
            sys.exit(-23)
        if Rt < 1 or Rt > 100:
            print('Invalid upper bound {}: integers must be within range [1, 100]!'.format(Rt))
            sys.exit(-24)

# processing arguments after surpassed all exception tests
def processArguments():
    # defining paths for data
    # would be glad if paths being of any OS' but Windows :)
    trainingFolder = argv[1] + '/'
    testdataFolder = argv[2] + '/'

    # defining limits for k being tested
    L = int(argv[3])
    R = int(argv[4])

    return trainingFolder, testdataFolder, L, R

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

# main function of this source code
def mainFunction(trainingFolder, testdataFolder, L, R):
    # reading training data
    imgs, labels = readImages_Training(trainingFolder)

    # reading test data
    testimgs, testnames = readImages_TestData(testdataFolder)

    # each iteration is a different k used in respective kNN training module
    for k in range(L, R+1):
        # initialize module
        print('Begin working with k = ' + str(k) + '.', flush=True)
        startTime = time.time()
        from sklearn.neighbors import KNeighborsClassifier
        KNN_Module = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='ball_tree')
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

if __name__ == "__main__":
    filteringException()
    trainingFolder, testdataFolder, L, R = processArguments()
    mainFunction(trainingFolder, testdataFolder, L, R)
