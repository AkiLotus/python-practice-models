# put this python source code on the main folder of the dataset
# command line scripts: "python3 thisfilename.py <trainingFolder> <testdataFolder> <csvoutputPrefix> <iterationType> <min> <max> [<step>]"
# constraints (1+2): <trainingFolder> and <testdataFolder> must exist
# constraints (3): <csvoutputPrefix> must make sure any generated files didn't already exist
# constraints (4): <iterationType> must be either "expo" (exponential) or "rnge" (arithmetic progression)
# constraints (4)(cont): if "expo", (MinRange, MaxRange) = (0, 13); else (MinRange, MaxRange) = (1, 9999)
# constraints (5+6): <min> and <max> are integers. MinRange <= min <= max <= MaxRange
# constraints (5+6)(cont): MinRange and MaxRange depend on <iterationType>
# constraints (7): <step> is optional (1 by default), yet if defined, must be a positive integer

# the (min, max, step) tuple works exactly as how Python's range works

# data would be distributed as following:
# a "training" folder, consists of labelled images
# "training" folder has subfolders, each contains images of the same label, and the folder itself is named after that label
# a "testdata" folder, consists of unlabelled images, used to test the accuracy of the modules

from sys import argv
import os, time, sys, psutil
from sklearn.ensemble import RandomForestClassifier
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
    if len(argv) != 7 and len(argv) != 8:
        # incorrect arguments count
        print('Incorrect format!')
        print('Valid format: "python3 thisfilename.py <trainingFolder> <testdataFolder> <csvoutputPrefix> <iterationType> <min> <max> [<step>]"')
        sys.exit(-1)
    else:
        # folders not found
        if not os.path.isdir(argv[1]):
            print('Training folder "{}" not found!'.format(argv[1]))
            sys.exit(-4041)
        if not os.path.isdir(argv[2]):
            print('Testdata folder "{}" not found!'.format(argv[2]))
            sys.exit(-4042)

        # illegal iterationType (argv[4])
        if argv[6] != "expo" and argv[4] != "rnge":
            print('Invalid argv[4]: <iterationType> can only be "expo" or "rnge"!')
            sys.exit(-14)

        # non-integers arguments at argv[5] and argv[6]
        try: int(argv[5])
        except ValueError as ex5:
            print('{} cannot be parsed into int: {}', argv[5], ex5)
            sys.exit(-45)
        try: int(argv[6])
        except ValueError as ex6:
            print('{} cannot be parsed into int: {}', argv[6], ex6)
            sys.exit(-46)
        
        # illegal integer limits
        Lf = int(argv[5]); Rt = int(argv[6])
        if Lf > Rt:
            print('Invalid limit: Lower bound {} exceeded upper bound {}!'.format(Lf, Rt))
            sys.exit(-4)
        if argv[4] == "expo" and (Lf < 0 or Lf > 13):
            print('Invalid lower bound {}: integers must be within range [0, 13]!'.format(Lf))
            sys.exit(-25)
        if argv[4] == "expo" and (Rt < 0 or Rt > 13):
            print('Invalid upper bound {}: integers must be within range [0, 13]!'.format(Rt))
            sys.exit(-26)
        if argv[4] == "rnge" and (Lf < 1 or Lf > 9999):
            print('Invalid lower bound {}: integers must be within range [1, 9999]!'.format(Lf))
            sys.exit(-25)
        if argv[4] == "rnge" and (Rt < 1 or Rt > 9999):
            print('Invalid upper bound {}: integers must be within range [1, 9999]!'.format(Rt))
            sys.exit(-26)

        if len(argv) == 8:
            # non-integers arguments at argv[7]
            try: int(argv[7])
            except ValueError as ex5:
                print('{} cannot be parsed into int: {}', argv[5], ex5)
                sys.exit(-47)

            # non-positive arguments at argv[7]
            if (int(argv[7]) <= 0):
                print('Invalid argv[7]: <step> must be a positive integer!'.format(argv[7]))
                sys.exit(-447)

# processing arguments after surpassed all exception tests
def processArguments():
    # defining paths for data
    # would be glad if paths being of any OS' but Windows :)
    trainingFolder = argv[1] + '/'
    testdataFolder = argv[2] + '/'
    csvoutputPrefix = argv[3]

    # iterationType initialization
    iterationType = argv[4]

    # defining limits for k being tested
    L = int(argv[5])
    R = int(argv[6])
    step = 1
    if len(argv) == 8: step = int(argv[7])

    return trainingFolder, testdataFolder, csvoutputPrefix, iterationType, L, R, step

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
def prediction(module, csvFileName, testList, fnameList):
    # initialize
    csvOutput = open(csvFileName, 'w')
    csvOutput.write('ImageID,Label\n')
    cntimg = len(testList)
    startTime = time.time()

    # perform prediction by built-in predict() function
    write('Begin predicting...', flush=True)
    write('Writing target: ' + csvFileName + ' ...', flush=True)
    labelList = module.predict(testList)

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
def mainFunction(process, trainingFolder, testdataFolder, csvPrefix, iterationType, L, R, step):
    # reading training data
    imgs, labels = readImages_Training(trainingFolder)

    # reading test data
    testimgs, testnames = readImages_TestData(testdataFolder)

    # each iteration is a different k used in respective Random Forest training module
    # to be more precise, for each k, the amount of trees in the forest is 2^k
    for k in range(L, R+1, step):
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
            write('Error, file {} already exists!'.format(csvResult))
            sys.exit(-4096)

        # initialize module
        write('\nBegin working with n_estimators = ' + str(treeCount) + '.', flush=True)
        write('Begin training using ' + str(len(imgs)) + ' images...', flush=True)
        MemBefore = process.memory_info().rss
        startTime = time.time()
        RF_Module = RandomForestClassifier(n_estimators=treeCount, criterion='gini', warm_start=False)
        RF_Module.fit(imgs, labels)
        endTime = time.time()
        MemAfter = process.memory_info().rss
        write('Successfully fitted ' + str(len(imgs)) + ' images.', flush=True)
        write('Elapsed time: ' + str(endTime - startTime) + ' seconds.', flush=True)
        displayMemory(MemBefore, MemAfter)

        # perform prediction
        prediction(RF_Module, csvResult, testimgs, testnames)
        del RF_Module, startTime, endTime, MemBefore, MemAfter

if __name__ == "__main__":
    # initialize memory monitor
    this_process = psutil.Process(os.getpid())

    # handling exceptions and arguments
    filteringException()
    trainingFolder, testdataFolder, csvPrefix, iterationType, L, R, step = processArguments()

    # initialize logfiles
    logfile = open(logname, 'w')
    logfile.write('Command line: python3 ')
    for arg in argv: logfile.write(arg + ' ')
    logfile.write('\n\n')
    
    # main training
    mainFunction(this_process, trainingFolder, testdataFolder, csvPrefix, iterationType, L, R, step)

    # finish logging
    logfile.close()
    print('Logs saved into ' + logname + '.')
