from scipy import sparse
from scipy.stats import uniform
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from numpy import count_nonzero
import math
import datetime
from sklearn.preprocessing import normalize

#------------------------------------------------------------------
# Training functions


#below function accepts category and document and returns 1 if document category is same as given category 
def deltaFunction(category,document):
    try:
        if compressedTraining[document-1,totalColumnsTraining-2] == category:
            return 1
        else:
            return 0
    
    except IndexError as e:
        print(str(e))

#below function helps to create delta matrix by iterating through classes and examples
def createDeltaMatrix():
    global deltaMatrix
    for i in range(20):
        tempList=[]
        for j in range(numberOfDocumentsTraining): #number od examples
            tempList.append(deltaFunction(i+1,j+1))
        deltaMatrix.append(tempList)
    
# below code is to generate txt file
# createDeltaMatrix()
# print("delta matrix done")

# with open('deltaMatrix(train).txt', 'w') as filehandle:
#     for listitem in deltaMatrix:
#         filehandle.write('%s\n' % listitem)
#     print("deltamatrix stored")

# below function used to find X matrix and Y matrix
def exMatrixAndYMatrix():
    global exMatrixT
   
    for i in range(numberOfDocumentsTesting):
        print(i)
        tempListEx= [1]   # first attributes values are set as 1 to make columns n+1
        
        for j in range(totalColumnsTesting):  # done for texting file
            
            try:
                tempListEx.append(compressedTesting[i,j])
            except IndexError:
                tempListEx.append(0)
        exMatrixT.append(tempListEx)
   
    print("=================")

# below code is to generate txt file

# exMatrixAndYMatrix()
# print("x matrix test done")

# with open('exMatrixTest.txt', 'w') as filehandle:
#     for listitem in exMatrixT:
#         filehandle.write('%s\n' % listitem)
#     print("exMatrixT test stored")

# with open('yMatrix(train).txt', 'w') as filehandle:
#     for listitem in yMatrix:
#         filehandle.write('%s\n' % listitem)
#     print("yMatrix stored")

#below function helps to initialize W as null matrix
def initializeWMatrix(N,M):
    global W
    W = [[ 0 for i in range(M)] for j in range(N) ] 

#below function helps to find exp (W. Xt) and normalize its value to value / sum of the column
def getNormalizedMatrix():
   
    x = (np.dot(WM,exMatrixM.T))  #find (W.Xt)

    a = x[:-1,:]
    length = len(x[0])
    listOfStrings1 = [ 1 ] * length  #remove last row and fill with 1's
    b = np.array(listOfStrings1)

    result = np.vstack((a,b))

    df = pd.DataFrame(result)    
    a = df.div(df.sum(axis=0), axis=1)  #normalize using dataframe 
    dataN = a.values.tolist()


    return np.expm1(dataN)  #Exp of above noramalized matrix

def updateW(numberOfIteration):
    global WM
    normMatrix = np.array([])
    
   
    for i in range(numberOfIteration):

        print(i)
        normMatrix = getNormalizedMatrix() 
        deltaP = np.subtract(deltaMatrix,normMatrix) # delta - norm matrix
        deltaPX = np.matmul(deltaP,exMatrixM) # delta * X
        penaltyW = penaltyRate * WM   
        final = np.subtract(deltaPX,penaltyW)
        finalEta = learningRate * final

        WM = WM.__add__(finalEta)   # final W matrix
        

#----------------------------------------------------------------------------------------------------------

# Testing functions
#below is the function which finds exp(W*Xt) for testing data

def getMatrixForPrediction():
    x = (np.dot(WM,exMatrixMT.T))

    a = x[:-1,:]
    length = len(x[0])
    listOfStrings1 = [ 1 ] * length
    b = np.array(listOfStrings1)

    result = np.vstack((a,b))

    df = pd.DataFrame(result)    
    a = df.div(df.sum(axis=0), axis=1)  # normalized 
    dataN = a.values.tolist()


    return np.expm1(dataN)


# this function predicts class for each example by using argmax for above normalized matrix
def predict():
    predictionMatrix = getMatrixForPrediction()
    classesPredicted = np.argmax(predictionMatrix, axis = 0)

    
    finalOutput = list()
    print("formatting output file")
    # this makes list of list having id and class
    for i in range(0,len(classesPredicted)):
        tempList = list()
        tempList.append(IDList[i])
        tempList.append(classesPredicted[i]+1)
        finalOutput.append(tempList)
    return finalOutput

# convert list of list to csv file
def listOfListToCSV():
    finalOutputList = predict()
    final =pd.DataFrame(finalOutputList,columns=['id','class'])
    final.to_csv('finalOutput(0.001,0.001,1500).csv', index = False)
    print("file is out ")



# main function : it has all data import , sparse matrix and function class
if __name__ == '__main__':

    yMatrix = list()
    exMatrix = list()
    deltaMatrix = list()
    exMatrixT = list()
    # to extract X matrix of training
    with open(r'exMatrix(train).txt', 'r') as filehandle:
        for line in filehandle:
            temp=[]
            currentPlace = line[:-1]
            line = currentPlace[1:-1].split(",")
            for i in range(len(line)):
                temp.append(int(line[i]))
            exMatrix.append(temp)
    print("exMatrix got")


    # to extract X matrix of testing

    with open(r'exMatrixTest.txt', 'r') as filehandle:
        for line in filehandle:
            temp=[]
            currentPlace = line[:-1]
            line = currentPlace[1:-1].split(",")
            for i in range(len(line)-1):
                temp.append(int(line[i]))
            exMatrixT.append(temp)
    print("exMatrixT got")





    yo = np.array(exMatrixT)

    # to extract Y matrix of testing

    with open(r'yMatrix(train).txt', 'r') as filehandle:
        for line in filehandle:
            temp = []
            currentPlace = line[:-1]
            temp.append(int(currentPlace[1:-1]))
            yMatrix.append(temp)


     # to extract Y matrix of testing

    with open(r'deltaMatrix(train).txt', 'r') as filehandle:
        for line in filehandle:
            temp=[]
            currentPlace = line[:-1]
            line = currentPlace[1:-1].split(",")
            for i in range(len(line)):
                temp.append(int(line[i]))
            deltaMatrix.append(temp)
        
    print("deltaMatrix got")

    # read csv files
    trainingSet = pd.read_csv(r'training.csv', squeeze=True, header=None,error_bad_lines=False)
    testingSet = pd.read_csv (r'testing.csv', squeeze=True, header=None,error_bad_lines=False)

    numberOfDocumentsTraining = len(trainingSet)  # only docs
    totalColumnsTraining = len(trainingSet.columns) # 0-15 =====16

    filteredTrainingSet = trainingSet.iloc[:, 1:]
    sparseMatrixTraining = filteredTrainingSet.values
    compressedTraining = csr_matrix(sparseMatrixTraining)
    W = []

    learningRate = 0.001
    penaltyRate = 0.001
    classesPredicted = []


    numberOfDocumentsTesting = len(testingSet) # only docs
    totalRowsTesting = len(testingSet.index) 
    totalColumnsTesting = len(testingSet.columns)
    print("----------")
    filteredTesting = testingSet.iloc[:, 1: ]

    sparseMatrixTesting = filteredTesting.values
    compressedTesting = csr_matrix(sparseMatrixTesting)
    IDList = (testingSet[0]).tolist()

    print("data set up")

    initializeWMatrix(20,61189 )   #totalColumnsTraining  rows and columns size totalColumnsTraining -1

    print("W initialized")

    exMatrixM = np.array(exMatrix)
    deltaMatrixM = np.array(deltaMatrix)
    WM = np.array(W)

    t = exMatrixM.T

    updateW(3000)   # update w for n iteration
    print("W done")

    exMatrixMT = np.array(exMatrixT)

    listOfListToCSV()

