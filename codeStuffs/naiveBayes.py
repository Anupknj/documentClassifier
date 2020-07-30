from scipy import sparse
from scipy.stats import uniform
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from numpy import count_nonzero
import math
import datetime

#-------------------------------------------------------------------------------------------------------------------
# functions used for training:

# this functions takes category and wordID as arguments and returns total count of respective wordId belonging 
# to that category
def countWordOccurenceOfCategory(category,word):
    occurence = 0
    for i in range(0,numberOfDocumentsSplitTraining):
        try:
            if (compressedSplitTraining[i,word-1]!=0 and compressedSplitTraining[i,totalColumnsTraining-2]==category):
                occurence+=compressedSplitTraining[i,word-1]
        except IndexError as error:
            print(error)
    return occurence

# this calls countWordOccurenceOfCategory function for every category
def getWordOccurenceForAllCategory():
    mainList = list()
    for i in range (1,21):
        print(i)
        DocWordIds = list()
        for j in range(1,totalColumnsTraining-1):
            DocWordIds.append(countWordOccurenceOfCategory(i,j))
        mainList.append(DocWordIds)

    # mainList is stored in an external text file named wordOccurence
    with open('wordOccurence.txt', 'w') as filehandle:
        for listitem in mainList:
            filehandle.write('%s\n' % listitem)
        # print("word Occurence stored")   

# this function helps to return list stored in text file
def countWordOccurenceOfCategoryList(cat,word):
    listBa = places[cat][1:-1]
    listLelo= listBa.split(",")
    return int(listLelo[word])

def countWordProb(word):
    listBa = wordCount[1:-1]
    # listLelo= listBa.split(",")
    return int(listBa[word])

# this function accepts category and returns total number of words in that category
def totalNumberOfWordsOfCat(category):
    totalNumbers = 0 
    rows, cols =compressedTraining.nonzero()
    for i in range(0,len(rows)):
        if compressedTraining[rows[i],totalColumnsTraining-2] == category:
            if cols[i] == totalColumnsTraining-2:
                continue
            totalNumbers+= compressedTraining[rows[i],cols[i]]
    return totalNumbers

# this function accepts category and returns total number of documents belonging to that category
def totalDocsOfACat(category):
    docCounter = 0
    for i in range(0,numberOfDocumentsTraining):
        if  compressedTraining[i,totalColumnsTraining-2] == category:
            docCounter+=1
    return docCounter

#-----------------------------------------------------------------------------------------------------------
#Below functions help to extract testing document and calculate the probability for each of the class and predict 

# this function recieves words list and document id of the test data and predicts the class
def calculateProbability(wordsList,docId):
    maxValue =  float("-inf") # value which is used to predict the class (it is initialized to -ve infinity)
    classPredicted = 0

    for i in range(1,21):
        totalMAP = 0
        for j in wordsList:
            MAP = (countWordOccurenceOfCategoryList(i-1,j-1) + (alpha - 1)) / (totalCountsCat[i] + ((alpha-1) * lengthOfVocab))
            totalMAP += (compressedTesting[docId,j] * math.log2(MAP))
        value = math.log2(MLE[i]) + totalMAP
        # value is calculated for each of the class and for each of the word IDs using MLE and MAP

        if value > maxValue:  # if the value is higher, then it is stored
            maxValue = value
            classPredicted = i # class is the index 
    print(classPredicted)
    classesPredicted.append(classPredicted) # finally class predicted is stored in list classesPredicted
    
# this function accepts datasets and it returns document id and its words list
def findWordsInTesting(a):
    rows, cols =a.nonzero()  # gets rows and columns seperately from compressed data
    rowCounter = 0  # keeps count of document id
    wordsList =list()
    for i in range(0,len(rows)): 
        if cols[i] == 0:
            continue
        if rows[i] != rowCounter: # gets column id's (word id) till row changes  ie, it gets all wordIds belonging to single row (doc id)
            print(rowCounter)   
            calculateProbability(wordsList,rowCounter)  # class is predicted for that document
            rowCounter = rows[i]
            wordsList.clear()
        
        if a[rows[i],cols[i]] != 0:
            wordsList.append(cols[i])
    print(rowCounter)
   

    
    calculateProbability(wordsList,rowCounter) # last document
  

#  this functions finds weights for each word ID
def findWeightOfWords():
    classP = 0
    for j in range(61185):
        print(j)
        if classP == 0:
            for i in range(1,21):
                classP += (MLE[i] * math.log2(MLE[i])) # overall entropty

        posP = 0
        negP = 0
        IG = 0 

        for i in range(1,21):
            probW = countWordProb(j) / 2905656 # probability of the word
            MAP = (countWordOccurenceOfCategoryList(i-1,j) + (alpha - 1)) / (totalCountsCat[i] + ((alpha-1) * lengthOfVocab))
            posP += (MAP * math.log2(MAP)) # entropty for positive case
            negP += ((1-MAP) * math.log2(1-MAP)) # entropty of negative case
        IG = -(classP) + probW * posP + (1-probW) * negP # information gain
        print(IG)
        totalIG.append(IG)
        
        
#  find probability of the each word    
def wordProb():
    count = 0
    global wordCount
    for j in range(61188):
        print(j)
        for i in range(1,21):
            
            count+= countWordOccurenceOfCategoryList(i-1,j)

        wordCount.append(count)

# finds top 100 values
def findTop100Weights():

    N = 100

    final_list = [] 
  
    for i in range(0,N):  
        print(i)
        max1 = 0
          
        for j in range(len(wordWeights)):      
            if float(wordWeights[j]) > max1: 
                index1 = (wordWeights.index(wordWeights[j]))+1
                max1 = float(wordWeights[j])
        wordWeights.remove(str(max1))
        
        final_list.append(index1) 
          
    print(final_list)

# this finds confusion matrix for the given matrix
def findConfusionMatrix(x):
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			if i == j:
				x[i, j] = 0
	
	confusionClassList = []
	x = np.transpose(x)
	for i in range(x.shape[0]):
		confusionClassList.append(sum(x[i]))
	return confusionClassList    


#------------------------------------------------------------
# functions for the output of the classes predicted in the required format

# this function makes a list of list containing ID and its predicted class
def outputFormat():
    finalOutput = list()
    print("formatting output file")

    for i in range(0,len(classesPredicted)):
        tempList = list()
        tempList.append(IDList[i])
        tempList.append(classesPredicted[i])
        finalOutput.append(tempList)
    return finalOutput

# this function converts  above list to CSV file
def listOfListToCSV():
    finalOutputList = outputFormat()
    final =pd.DataFrame(finalOutputList,columns=['id','class'])
    final.to_csv('finalOutput(0.0001).csv', index = False)
    print("file is out ")

def plotMap():
    plt.semilogx([0.00001,0.0001,0.001,0.01, 0.1], [0.86713,0.86802,0.86772,0.85975,0.82993])
    plt.xlabel('B-values')
    plt.ylabel('accuracy')
    plt.title('graph comparing beta values and accuracy')

    plt.show()
#----------------------------------------------------------
# MAIN function which contains main flow

if __name__ == '__main__':

    
    # This follows data set ups

    # retrieving trained word occurence file which contains word counts category wise
    places = [] 
    with open(r'wordOccurenceCategory.txt', 'r') as filehandle:
        for line in filehandle:
            currentPlace = line[:-1]
            places.append(currentPlace)
        print("word occurence list retrieved")
    
    wordWeights = [] 
    # with open(r'D:\University\Academics\_ML\project\P2\IGList.txt', 'r') as filehandle:
    #     for line in filehandle:
    #         currentPlace = line[:-1]
    #         wordWeights.append(currentPlace)
    #     print("IG list retrieved")
    
    # findTop100Weights()

    wordCount = []
    # wordProb()
    # with open('IGList.txt', 'w') as filehandle:
    #     for listitem in wordCount:
    #         filehandle.write('%s\n' % listitem)
    #     print("IGList  stored")

    
    # with open(r'D:\University\Academics\_ML\project\P2\wordIDProb.txt', 'r') as filehandle:
    #     for line in filehandle:
    #         currentPlace = line[:-1]
    #         wordCount.append(currentPlace)
    #     print("word occurence list retrieved")
    # wordProb()
    # print(countWordProb(1))
    #imports file for training and testing set ups


    trainingSet = pd.read_csv(r'training.csv', squeeze=True, header=None,error_bad_lines=False)

    testingSet = pd.read_csv (r'testing.csv', squeeze=True, header=None,error_bad_lines=False)
    
    # List MLE list contains MLE for each category ie, number of docs Yk/ total number of docs
    MLE = [0,0.04025,0.052,0.05183,0.053583,0.05016,0.0525,0.0515,0.05116,0.054083,0.0523,0.05383,0.05325,0.05216,0.05166,0.05308,0.05425,0.0483,0.04941,0.03891,0.035583]
    # List contains total words in each of the category
    totalCountsCat = [0,154382,138499,116141,103535,90456,144656,64094,107399,110928,124537,143087,191242,97924,158750,162521,236747,172257,280067,172670,135764]


    lengthOfVocab = 61188       # total unique words
    beta =  1/ lengthOfVocab   
    alpha = 1 +beta
    classesPredicted = list()   # list containing predicted classes

    # training data set up
    numberOfDocumentsTraining = len(trainingSet)  # total number   docs
    totalColumnsTraining = len(trainingSet.columns) #  total number of columns  0-15 =====16
    filteredTrainingSet = trainingSet.iloc[:, 1:]   
    sparseMatrixTraining = filteredTrainingSet.values   # getting sparse matrix
    compressedTraining = csr_matrix(sparseMatrixTraining) # getting compressed data from sparse matrix

    # test data split 

    splitOfTheData = round(numberOfDocumentsTraining * 0.7)

    splitTrain = trainingSet.iloc[:splitOfTheData, 1: ]
    splitTest = trainingSet.iloc[splitOfTheData+1:, 1 :]

    sparseMatrixSplitTrain = splitTrain.values
    sparseMatrixSplitTest = splitTest.values

    compressedSplitTraining  = csr_matrix(sparseMatrixSplitTrain)
    compressedSplitTesting = csr_matrix(sparseMatrixSplitTest)

    numberOfDocumentsSplitTraining = len(splitTrain)
    numberOfDocumentsTesting = len(testingSet) 
    totalRowsTesting = len(testingSet.index) 

    # testing data set up

    filteredTesting = testingSet.iloc[:, : ]

    sparseMatrixTesting = filteredTesting.values
    compressedTesting = csr_matrix(sparseMatrixTesting) # getting compressed data out of sparse matrix
    IDList = (testingSet[0]).tolist()

    totalIG = []
    # findWeightOfWords()
    # with open('IGList.txt', 'w') as filehandle:
    #     for listitem in totalIG:
    #         filehandle.write('%s\n' % listitem)
    #     print("IG List  stored")
    # compressedTesting document is predicted
    findWordsInTesting(compressedTesting) 

    # output file is generated
    listOfListToCSV()

    # plotMap()
