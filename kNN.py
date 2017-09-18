from numpy import *
import operator
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import locale
import codecs
from os import listdir

def createDataSet():
		group = np.array([
                    [1.0,1.1],
                    [1.0,1.0],
                    [0,0],
                    [0,0.1]
                    ])
		labels = ['A','A','B','B']
		return group,labels

def classify0(inX,dataX,labels,k):    
    dataSize = dataX.shape[0]  #行数
    diffMat = np.tile(inX,(dataSize,1))-dataX #intX 复制4行，形成矩阵，并计算距离差
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)#按行相加
    distance = sqDistance**0.5 #开根号，distance是array
    sortedDistIndicies = distance.argsort()#返回排序后的下标
    classCount={}#字典 key id label, val is count
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]#排名第i的label
        classCount[voteIlabel] = classCount.get(voteIlabel,0)
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #根据iteritems的第1个元素排序，即字典中的val，第0个是key
    #reverse定义为True时将按降序排列
    return sortedClassCount[0][0] #返回排序后的字典的前一个(从0开始)

def file2matrix(filename):
    with open('D:\ML\kNN\datingTestSet.txt','r') as fr:
        arrayOLines = fr.readlines()
        numberOfLine = len(arrayOLines)
        returnMat = np.zeros((numberOfLine,3))
        classLabelVector=[]
        index = 0
        for line in arrayOLines:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index,:] = listFromLine[0:3]
            labels = {'didntLike':1,'smallDoses':2,'largeDoses':3}
            classLabelVector.append(labels[listFromLine[-1]]) 
            index += 1
        return returnMat,classLabelVector

def autoNorm(dataX):
    #归一化公式 newVal=(oldVal-min)/(max-min)
    minVals = dataX.min(axis=0)
    maxVals = dataX.max(0)
    ranges = maxVals - minVals
    rows = dataX.shape[0]
    newVal = dataX - tile(minVals,(rows,1)) #(oldVal-min)
    ## tile(minVals,(row,1))复制row行,列数为minVals列数的一倍
    newVal = newVal/tile(ranges,(rows,1))#(oldVal-min)/(max-min)
    return newVal,ranges,minVals

def datingClassTest():
    hoRatio = 0.10#10% of data as test
    #读入数据
    filename = 'D:\ML\kNN\datingTestSet2.txt'
    dataX,labels = file2matrix('D:\ML\kNN\datingTestSet2.txt')
    #归一化
    normMat,ranges,minVals = autoNorm(dataX)
    m = dataX.shape[0]#number of rows
    numTestVecs = int(m*hoRatio)#number of test
    errorcount = 0;#initialize number of errors
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],labels[numTestVecs:m],5)
        print("the classifier predicted %d, the real answer is :%d" %((classifierResult),labels[i]))
        if(classifierResult != labels[i]):
            errorcount+=1
    print("error rate :%f" %((errorcount)/(numTestVecs)))

def classifyPerson():
    resultList = ["第一类","第二类","第三类"] #output lables
    percentTats = float(input("玩游戏消耗的时间"))
    ffilm = float(input("每年获得的飞行里程数"))
    iceCream = float(input("每周消耗冰激凌"))
    #读入数据
    filename = 'D:\ML\kNN\datingTestSet2.txt'
    dataX,labels = file2matrix('D:\ML\kNN\datingTestSet2.txt') 
    #归一化
    normMat,ranges,minVals = autoNorm(dataX)
    test_list = np.array([percentTats,ffilm,iceCream])
    classifierResult = classify0(test_list,dataX,labels,3)
    print("你喜欢的类别是:"+resultList[classifierResult])

def img2vector(filename):
    returnVect = zeros((1,1024))
    with open(r'D:\ML\kNN\digits\testDigits\test.txt','rt') as fr:#字符串前加r，表示的是禁止字符串转义
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir(r'D:\ML\kNN\digits\trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir(r'D:\ML\kNN\digits\testDigits')
    errorcount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        #print("the classifier came back with %d, the real answer is %d" %(classifierResult,classNumStr))
        if(classifierResult != classNumStr):
            errorcount+=1.0
    print("\nthe total number of errors is %d" %errorcount)
    print("\nthe total error rate is %f" %float((errorcount/mTest)))
            

        
