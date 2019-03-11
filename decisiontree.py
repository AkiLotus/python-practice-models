# this is a script used to generate a decision tree from scratch
# currently works with the infamous "iris" dataset
# average producing time at about 97 seconds

import numpy as np
import pandas as pd
import sys, time

sys.setrecursionlimit(1000)

decisiontree = []
adjacencyList = []

irisSpecies = ["setosa", "versicolor", "virginica"]

def calcEntropy(dataset):
    if dataset.size == 0:
        return 0.0
    specCount = [0, 0, 0]
    totalRow = 0
    entropy = 0.0
    for index, row in dataset.iterrows():
        totalRow += 1
        for enum in range(3):
            if row["species"] == irisSpecies[enum]:
                specCount[enum] += 1
    for enum in range(3):
        if specCount[enum] == 0:
            continue
        prob = specCount[enum] / totalRow
        entropy -= prob * np.log2(prob)
    return entropy

def calcGain(dataset, colLabel, lbound):
    infoGain = calcEntropy(dataset)
    totalRow = 0; trueCount = 0; falseCount = 0
    for index, row in dataset.iterrows():
        totalRow += 1
        if float(row[colLabel]) < lbound:
            trueCount += 1
        else: falseCount += 1
    infoGain -= (trueCount / totalRow) * calcEntropy(dataset.loc[dataset[colLabel] < lbound])
    infoGain -= (falseCount / totalRow) * calcEntropy(dataset.loc[dataset[colLabel] >= lbound])
    return infoGain

def DFS(dataset, bound, index, parent):
    for spec in irisSpecies:
        if dataset.loc[dataset["species"] == spec].size == dataset.size:
            adjacencyList.append([])
            decisiontree.append(((spec, -1.0), parent))
            return

    maxinfoGain = -4
    bestCriteria = ("N/A", -1, -1.0)
    colid = 0
    for colLabel in dataset.columns:
        if colLabel == "species":
            continue
        for lbound in np.arange(bound[colid][0], bound[colid][1], 0.1):
            infoGain = calcGain(dataset, colLabel, float(lbound))
            if infoGain > maxinfoGain:
                maxinfoGain = infoGain
                bestCriteria = (colLabel, colid, float(lbound))
        colid += 1
    
    adjacencyList.append([])
    decisiontree.append(((bestCriteria[0], bestCriteria[2]), parent))

    adjacencyList[index].append(len(decisiontree))
    subset1 = dataset.loc[dataset[bestCriteria[0]] < bestCriteria[2]]
    newbound = bound.copy(); newbound[bestCriteria[1]] = (newbound[bestCriteria[1]][0], bestCriteria[2] - 0.1)
    DFS(subset1, newbound, len(decisiontree), index)

    adjacencyList[index].append(len(decisiontree))
    subset2 = dataset.loc[dataset[bestCriteria[0]] >= bestCriteria[2]]
    newbound = bound.copy(); newbound[bestCriteria[1]] = (bestCriteria[2], newbound[bestCriteria[1]][1])
    DFS(subset2, newbound, len(decisiontree), index)

def displayTree(index, depth):
    for i in range(2*depth):
        print(" ", end="")
    NodeData = decisiontree[index][0]
    label, lbound = NodeData[0], NodeData[1]
    if lbound == -1.0:
        print("[Specified: {}]".format(label))
    else:
        print("[{} < ".format(label) + "{0:.1f}".format(lbound) + "]")
    for id in adjacencyList[index]:
        displayTree(id, depth+1)

StartTime = time.time()

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
bound = [(0.0, 10.0) for i in range(4)]
DFS(data, bound, 0, -1)
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
