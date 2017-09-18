from numpy import *
import operator
import numpy as np
import os

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
