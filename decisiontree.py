# this is a script used to generate a decision tree from scratch
# currently works with the infamous "iris" dataset
# average producing time at about 97 seconds

import numpy as np
import pandas as pd
import sys, time

# recursion limit, 1000 by default
# in case the tree is large, this is to avoid error by maximum recursion depth reached
sys.setrecursionlimit(1000)

# initialization of the tree
decisiontree = []
adjacencyList = []

# list of iris species, for convenience
irisSpecies = ["setosa", "versicolor", "virginica"]

# calculating entropy of a discrete dataset
def calcEntropy(dataset):
    # empty dataset, entropy = 0
    if dataset.size == 0:
        return 0.0
    
    # initialize
    specCount = [0, 0, 0]
    totalRow = 0
    entropy = 0.0

    # counting species row-by-row
    for index, row in dataset.iterrows():
        totalRow += 1
        for enum in range(3):
            if row["species"] == irisSpecies[enum]:
                specCount[enum] += 1
    
    # calculating information given by each specie in the dataset
    for enum in range(3):
        if specCount[enum] == 0:
            continue
        prob = specCount[enum] / totalRow
        entropy -= prob * np.log2(prob)

    return entropy

# calculating information of a dataset, if applying criteria: colLabel < lbound
def calcGain(dataset, colLabel, lbound):
    # initialize with the whole dataset's entropy
    infoGain = calcEntropy(dataset)

    # intialize
    totalRow = 0; trueCount = 0; falseCount = 0

    # counting True/False row-by-row
    for index, row in dataset.iterrows():
        totalRow += 1
        if float(row[colLabel]) < lbound:
            trueCount += 1
        else: falseCount += 1
    
    # calculating information given by True/False
    infoGain -= (trueCount / totalRow) * calcEntropy(dataset.loc[dataset[colLabel] < lbound])
    infoGain -= (falseCount / totalRow) * calcEntropy(dataset.loc[dataset[colLabel] >= lbound])

    return infoGain

# creating nodes of the tree: current node is which with id 'index' and parent 'parent'
# bound array used to maintain the bound for each attribute
def DFS(dataset, bound, index, parent):
    # check if the dataset consists of a single specie
    # if so, terminate: this is one of the leaf nodes
    for spec in irisSpecies:
        if dataset.loc[dataset["species"] == spec].size == dataset.size:
            adjacencyList.append([])
            decisiontree.append(((spec, -1.0), parent))
            return

    # initialize to find the max information gain
    maxinfoGain = -4
    bestCriteria = ("N/A", -1, -1.0)
    colid = 0

    # traversing each column label (except 'species', of course)
    for colLabel in dataset.columns:
        if colLabel == "species":
            continue
        # traversing each possible lower bound
        # bound array used to only loop through existing values
        for lbound in np.arange(bound[colid][0], bound[colid][1], 0.1):
            infoGain = calcGain(dataset, colLabel, float(lbound))
            if infoGain > maxinfoGain:
                maxinfoGain = infoGain
                bestCriteria = (colLabel, colid, float(lbound))
        colid += 1
    
    # add the current node with the most optimal criteria to split dataset
    adjacencyList.append([])
    decisiontree.append(((bestCriteria[0], bestCriteria[2]), parent))

    # add left node: the node consist of the dataset returning True for the criteria
    # recurse to the next node asap
    adjacencyList[index].append(len(decisiontree))
    subset1 = dataset.loc[dataset[bestCriteria[0]] < bestCriteria[2]]
    newbound = bound.copy(); newbound[bestCriteria[1]] = (newbound[bestCriteria[1]][0], bestCriteria[2] - 0.1)
    DFS(subset1, newbound, len(decisiontree), index)

    # add right node: the node consist of the dataset returning False for the criteria
    # recurse to the next node asap
    adjacencyList[index].append(len(decisiontree))
    subset2 = dataset.loc[dataset[bestCriteria[0]] >= bestCriteria[2]]
    newbound = bound.copy(); newbound[bestCriteria[1]] = (bestCriteria[2], newbound[bestCriteria[1]][1])
    DFS(subset2, newbound, len(decisiontree), index)

# print the tree: each instance calls to the node 'index' with depth 0
def displayTree(index, depth):
    # illustrating depth by space indentation
    for i in range(2*depth):
        print(" ", end="")
    
    # printing mandatory data (the criteria required, or the specie for leaf node)
    NodeData = decisiontree[index][0]
    label, lbound = NodeData[0], NodeData[1]
    if lbound == -1.0:
        print("[Specified: {}]".format(label))
    else:
        print("[{} < ".format(label) + "{0:.1f}".format(lbound) + "]")
    
    # moving down to the children (of course, leaf node won't have any)
    for id in adjacencyList[index]:
        displayTree(id, depth+1)

if __name__ == "__main__":
    StartTime = time.time()

    # reading iris data from csv file, data retrieved goes into a DataFrame
    # read pandas' documentation for details
    data = pd.read_csv(
        "iris.csv",
        sep=",",
        quotechar="'",
        usecols=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
        dtype={
            "sepal_length": float,
            "sepal_width": float,
            "petal_length": float,
            "petal_width": float,
            "species": str
        }
    )

    # add "bound" for each attributes
    # this saves traversal time for unneccesary loop
    bound = [(0.0, 10.0) for i in range(4)]

    # building tree, starting from root
    # root node has index = 0
    DFS(data, bound, 0, -1)

    # display the tree, starting from root
    displayTree(0, 0)

    EndTime = time.time()

    print('Tree produced in {} seconds.'.format(EndTime - StartTime))

"""
[petal_length < 1.9]
  [Specified: setosa]
  [petal_width < 1.7]
    [petal_length < 5.0]
      [sepal_length < 5.0]
        [sepal_width < 2.4]
          [Specified: versicolor]
          [Specified: virginica]
        [Specified: versicolor]
      [sepal_width < 2.7]
        [Specified: virginica]
        [sepal_length < 6.1]
          [Specified: versicolor]
          [petal_length < 5.1]
            [Specified: versicolor]
            [Specified: virginica]
    [petal_length < 4.9]
      [sepal_length < 6.0]
        [Specified: versicolor]
        [Specified: virginica]
      [Specified: virginica]
"""
