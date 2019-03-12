# put this python source code on the main folder of the dataset
# command line scripts: "python3 thisfilename.py <trainingFolder> <testdataFolder> <csvoutputPrefix> <minNeighbors> <maxNeighbors>"
# constraints (1+2): <trainingFolder> and <testdataFolder> must exist
# constraints (3): <csvoutputPrefix> must make sure any generated files didn't already exist
# constraints (4+5): <minNeighbors> and <maxNeighbors> are integers. 1 <= minNeighbors <= maxNeighbors <= 100

# data would be distributed as following:
# a "training" folder, consists of labelled images
# "training" folder has subfolders, each contains images of the same label, and the folder itself is named after that label
# a "testdata" folder, consists of unlabelled images, used to test the accuracy of the modules

from sys import argv
import os, time, sys, psutil
from sklearn.neighbors import KNeighborsClassifier
import cv2
from glob import glob

# logs initialization
if not 'logs/' in glob('*/'):
    os.mkdir("logs/")
logname = "logs/logs-" + str(int(time.time() // 1)) + '.txt'
global logfile; logfile = None

# redefine "print" function to write in both stdout and logs
def write(str, end='\n', flush=False):
    print(str, end=end, flush=flush)
    logfile.write(str+end)

# exception handling
def filteringException():
    if len(argv) != 6:
        # incorrect arguments count
        write('Incorrect format!')
        write('Valid format: "python3 thisfilename.py <trainingFolder> <testdataFolder> <csvoutputPrefix> <minNeighbors> <maxNeighbors>"')
        sys.exit(-1)
    else:
        # folders not found
        if not os.path.isdir(argv[1]):
            write('Training folder "{}" not found!'.format(argv[1]))
            sys.exit(-4041)
        if not os.path.isdir(argv[2]):
            write('Testdata folder "{}" not found!'.format(argv[2]))
            sys.exit(-4042)

        # non-integers arguments at argv[4] and argv[5]
        try: int(argv[4])
        except ValueError as ex4:
            write('{} cannot be parsed into int: {}', argv[4], ex4)
            sys.exit(-44)
        try: int(argv[5])
        except ValueError as ex5:
            write('{} cannot be parsed into int: {}', argv[5], ex5)
            sys.exit(-45)
        
        # illegal integer limits
        Lf = int(argv[4]); Rt = int(argv[5])
        if Lf > Rt:
            write('Invalid limit: Lower bound {} exceeded upper bound {}!'.format(Lf, Rt))
            sys.exit(-4)
        if Lf < 1 or Lf > 100:
            write('Invalid lower bound {}: integers must be within range [1, 100]!'.format(Lf))
            sys.exit(-24)
        if Rt < 1 or Rt > 100:
            write('Invalid upper bound {}: integers must be within range [1, 100]!'.format(Rt))
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

    return trainingFolder, testdataFolder, csvoutputPrefix, L, R

# reading training data from training directories
def readImages_Training(trainingFolder):
    # initialize
    primalPath = trainingFolder
    subfolderList = sorted(glob(primalPath + '*/'))
    write('Begin loading from ' + primalPath + ' ...', flush=True)
    cntimg = 0; totalcnt = 0
    imgList = []; labelList = []
    startTime = time.time()

    # iterate all subfolders
    for path in subfolderList:
        id = path.replace(primalPath, '').replace('/', '')
        write('Begin loading from ' + path + ' ...', flush=True)
        cntimg = 0
        # iterate all images within subfolders
        for filename in os.listdir(path):
            imgList.append(cv2.imread(path + filename, 0).flatten())
            labelList.append(id)
            cntimg += 1; totalcnt += 1
            write('Loading sample image #' + str(cntimg) + ' from folder #' + str(id) + '...\r', end='', flush=True)
        write('', end='\n')
    
    # finalize and return value
    endTime = time.time()
    write('Successfully loaded ' + str(totalcnt) + ' images.', flush=True)
    write('Elapsed time: ' + str(endTime - startTime) + ' seconds.', flush=True)
    del primalPath, subfolderList, cntimg, totalcnt, startTime, endTime
    return imgList, labelList

# reading test data from test directories
def readImages_TestData(testdataFolder):
    # initialize
    path = testdataFolder
    write('Begin loading from ' + path + ' ...', flush=True)
    cntimg = 0
    tmpList = []; fnameList = []
    startTime = time.time()

    # iterate all images
    for filename in os.listdir(path):
        tmpList.append(cv2.imread(path + filename, 0).flatten())
        fnameList.append(filename)
        cntimg += 1
        write('Loading test image #' + str(cntimg) + '...\r', end='', flush=True)
    
    # finalize and return value
    endTime = time.time()
    write('\nSuccessfully loaded ' + str(cntimg) + ' images.', flush=True)
    write('Elapsed time: ' + str(endTime - startTime) + ' seconds.', flush=True)
    del path, cntimg, startTime, endTime
    return tmpList, fnameList

# perform predictions with given modules, target csv file and list of images to be predicted
def prediction(knnModule, csvFileName, testList, fnameList):
    # initialize
    csvOutput = open(csvFileName, 'w')
    csvOutput.write('ImageID,Label\n')
    cntimg = len(testList)
    startTime = time.time()

    # perform prediction by built-in predict() function
    write('Begin predicting...', flush=True)
    write('Writing target: ' + csvFileName + ' ...', flush=True)
    labelList = knnModule.predict(testList)

    # writing prediction results into csv
    for i in range(cntimg):
        csvOutput.write(fnameList[i] + ',' + str(labelList[i]) + '\n')
    
    # finalize and close file output stream
    endTime = time.time()
    write('Successfully predicted ' + str(cntimg) + ' images.', flush=True)
    write('Elapsed time: ' + str(endTime - startTime) + ' seconds.', flush=True)
    csvOutput.close()
    del csvOutput, cntimg, startTime, endTime, labelList

# printing logs for consumed memories
def displayMemory(MemBefore, MemAfter):
    memUsage = MemAfter - MemBefore
    write('Memory usage: %.2f MiB || %.2f KiB.' % (memUsage / 1048576, memUsage / 1024))

# main function of this source code
def mainFunction(process, trainingFolder, testdataFolder, csvPrefix, L, R):
    # reading training data
    imgs, labels = readImages_Training(trainingFolder)

    # reading test data
    testimgs, testnames = readImages_TestData(testdataFolder)

    # each iteration is a different k used in respective kNN training module
    for k in range(L, R+1):
        # initialize output csv
        csvResult = csvPrefix + '-'
        if k < 10:
            csvResult += '0'
        csvResult += str(k)
        csvResult += '.csv'

        # terminate if output csv file exists
        if os.path.isfile(csvResult):
            write('Error, file {} already exists!'.format(csvResult))
            sys.exit(-4096)

        # initialize module
        write('\nBegin working with k = ' + str(k) + '.', flush=True)
        write('Begin training using ' + str(len(imgs)) + ' images...', flush=True)
        MemBefore = process.memory_info().rss
        startTime = time.time()
        KNN_Module = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='ball_tree')
        KNN_Module.fit(imgs, labels)
        endTime = time.time()
        MemAfter = process.memory_info().rss
        write('Successfully fitted ' + str(len(imgs)) + ' images.', flush=True)
        write('Elapsed time: ' + str(endTime - startTime) + ' seconds.', flush=True)
        displayMemory(MemBefore, MemAfter)

        # perform prediction
        prediction(KNN_Module, csvResult, testimgs, testnames)
        del KNN_Module, startTime, endTime, MemBefore, MemAfter

if __name__ == "__main__":
    # initialize memory monitor
    this_process = psutil.Process(os.getpid())

    # handling exceptions and arguments
    filteringException()
    trainingFolder, testdataFolder, csvPrefix, L, R = processArguments()

    # initialize logfiles
    logfile = open(logname, 'w')
    logfile.write('Command line: python3 ')
    for arg in argv: logfile.write(arg + ' ')
    logfile.write('\n\n')
    
    # main training
    mainFunction(this_process, trainingFolder, testdataFolder, csvPrefix, L, R)

    # finish logging
    logfile.close()
    print('Logs saved into ' + logname + '.')
