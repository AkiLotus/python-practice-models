# put this python source code on the main folder of the dataset
# command line scripts: "python3 thisfilename.py [trainingFolder] [testdataFolder] [csvoutputPrefix] [minPow] [maxPow] [iterationType]"
# constraints (1+2): [trainingFolder] and [testdataFolder] must exist
# constraints (3): [csvoutputPrefix] must make sure any generated files didn't already exist
# constraints (4+5): [minPow] and [maxPow] are integers. MinRange <= minPow <= maxPow <= MaxRange
# constraints (4+5)(cont): MinRange and MaxRange depend on [iterationType]
# constraints (6): [iterationType] must be either "expo" (exponential) or "rnge" (arithmetic progression)
# constraints (6)(cont): if "expo", (MinRange, MaxRange) = (0, 13); else (MinRange, MaxRange) = (1, 9999)

# data would be distributed as following:
# a "training" folder, consists of labelled images
# "training" folder has subfolders, each contains images of the same label, and the folder itself is named after that label
# a "testdata" folder, consists of unlabelled images, used to test the accuracy of the modules

from sys import argv
import os, time, sys, psutil
from sklearn.ensemble import RandomForestClassifier
import cv2
from glob import glob

# exception handling
def filteringException():
    if len(argv) != 7:
        # incorrect arguments count
        print('Incorrect format!')
        print('Valid format: "python3 thisfilename.py [trainingFolder] [testdataFolder] [csvoutputPrefix] [iterationType]"')
        sys.exit(-1)
    else:
        # folders not found
        if not os.path.isdir(argv[1]):
            print('Training folder "{}" not found!'.format(argv[1]))
            sys.exit(-4041)
        if not os.path.isdir(argv[2]):
            print('Testdata folder "{}" not found!'.format(argv[2]))
            sys.exit(-4042)

        # non-integers arguments at argv[4] and argv[5]
        try: int(argv[4])
        except ValueError as ex4:
            print('{} cannot be parsed into int: {}', argv[4], ex4)
            sys.exit(-44)
        try: int(argv[5])
        except ValueError as ex5:
            print('{} cannot be parsed into int: {}', argv[5], ex5)
            sys.exit(-45)

        # illegal iterationType (argv[6])
        if argv[6] != "expo" and argv[6] != "rnge":
            print('Invalid argv[6]: [iterationType] can only be "expo" or "rnge"!')
            sys.exit(-16)
        
        # illegal integer limits
        Lf = int(argv[4]); Rt = int(argv[5])
        if Lf > Rt:
            print('Invalid limit: Lower bound {} exceeded upper bound {}!'.format(Lf, Rt))
            sys.exit(-4)
        if argv[6] == "expo" and (Lf < 0 or Lf > 13):
            print('Invalid lower bound {}: integers must be within range [0, 13]!'.format(Lf))
            sys.exit(-24)
        if argv[6] == "expo" and (Rt < 0 or Rt > 13):
            print('Invalid upper bound {}: integers must be within range [0, 13]!'.format(Rt))
            sys.exit(-25)
        if argv[6] == "rnge" and (Lf < 1 or Lf > 9999):
            print('Invalid lower bound {}: integers must be within range [1, 9999]!'.format(Lf))
            sys.exit(-24)
        if argv[6] == "rnge" and (Rt < 1 or Rt > 9999):
            print('Invalid upper bound {}: integers must be within range [1, 9999]!'.format(Rt))
            sys.exit(-25)

# processing arguments after surpassed all exception tests
def processArguments():
    # defining paths for data
    # would be glad if paths being of any OS' but Windows :)
    trainingFolder = argv[1] + '/'
    testdataFolder = argv[2] + '/'
    csvoutputPrefix = argv[3]

    # defining limits for k being tested
    L = int(argv[4])
    R = int(argv[5])

    # iterationType initialization
    iterationType = argv[6]

    return trainingFolder, testdataFolder, csvoutputPrefix, L, R, iterationType

# reading training data from training directories
def readImages_Training(trainingFolder):
    # initialize
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
            print('Loading sample image #' + str(cntimg) + ' from folder #' + str(id) + '...\r', end='', flush=True)
        print('', sep='\n')
    
    # finalize and return value
    endTime = time.time()
    print('Successfully loaded ' + str(totalcnt) + ' images.', flush=True)
    print('Elapsed time: ' + str(endTime - startTime) + ' seconds.', flush=True)
    del primalPath, subfolderList, cntimg, totalcnt, startTime, endTime
    return imgList, labelList

# reading test data from test directories
def readImages_TestData(testdataFolder):
    # initialize
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
        print('Loading test image #' + str(cntimg) + '...\r', end='', flush=True)
    
    # finalize and return value
    endTime = time.time()
    print('\nSuccessfully loaded ' + str(cntimg) + ' images.', flush=True)
    print('Elapsed time: ' + str(endTime - startTime) + ' seconds.', flush=True)
    del path, cntimg, startTime, endTime
    return tmpList, fnameList

# perform predictions with given modules, target csv file and list of images to be predicted
def prediction(module, csvFileName, testList, fnameList):
    # initialize
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
    del csvOutput, cntimg, startTime, endTime, labelList

# printing logs for consumed memories
def displayMemory(MemBefore, MemAfter):
    memUsage = MemAfter - MemBefore
    print('Memory usage: %.2f MiB || %.2f KiB.' % (memUsage / 1048576, memUsage / 1024))

# main function of this source code
def mainFunction(process, trainingFolder, testdataFolder, csvPrefix, L, R, iterationType):
    # reading training data
    imgs, labels = readImages_Training(trainingFolder)

    # reading test data
    testimgs, testnames = readImages_TestData(testdataFolder)

    # each iteration is a different k used in respective Random Forest training module
    # to be more precise, for each k, the amount of trees in the forest is 2^k
    for k in range(L, R+1):
        # initialize the number of trees, based on the iteration
        # and the iterationType declared from command line
        treeCount = 0
        if iterationType == "expo":
            treeCount = (2 ** k)
        else: treeCount = k

        # initialize output csv
        csvResult = csvPrefix + '-'
        if treeCount < 1000: csvResult += '0'
        if treeCount < 100: csvResult += '0'
        if treeCount < 10: csvResult += '0'
        csvResult += str(treeCount)
        csvResult += '.csv'

        # terminate if output csv file exists
        if os.path.isfile(csvResult):
            print('Error, file {} already exists!'.format(csvResult))
            sys.exit(-4096)

        # initialize module
        print('\nBegin working with n_estimators = ' + str(treeCount) + '.', flush=True)
        print('Begin training using ' + str(len(imgs)) + ' images...', flush=True)
        MemBefore = process.memory_info().rss
        startTime = time.time()
        RF_Module = RandomForestClassifier(n_estimators=treeCount, criterion='gini', warm_start=False)
        RF_Module.fit(imgs, labels)
        endTime = time.time()
        MemAfter = process.memory_info().rss
        print('Successfully fitted ' + str(len(imgs)) + ' images.', flush=True)
        print('Elapsed time: ' + str(endTime - startTime) + ' seconds.', flush=True)
        displayMemory(MemBefore, MemAfter)

        # perform prediction
        prediction(RF_Module, csvResult, testimgs, testnames)
        del RF_Module, startTime, endTime, MemBefore, MemAfter

if __name__ == "__main__":
    this_process = psutil.Process(os.getpid())

    filteringException()
    trainingFolder, testdataFolder, csvPrefix, L, R, iterationType = processArguments()
    mainFunction(this_process, trainingFolder, testdataFolder, csvPrefix, L, R, iterationType)
