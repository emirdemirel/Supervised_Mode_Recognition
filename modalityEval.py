#from compmusic import dunya
import os,sys, json, pickle, csv
import numpy as np
import essentia.standard as ess
import matplotlib.pyplot as plt
import itertools
from pandas import DataFrame, read_csv
import pandas as pd
import warnings

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score

from modalityUtils import AnalysisParams

def computeHPCP_Regional(fileData, region):
    '''
    INPUT :
    
    fileData (dict): Dictionary that contains all the necessary information of the audio for classification
    
    region (float) : local region of the song
    '''
    
    fileData['mean_hpcp_vector_' + str(region)] = []
    fileData['std_hpcp_vector_' + str(region)] = []

    for j in range(fileData['numBins']):
        hpcps = [];
        hpcplen = int(len(fileData['hpcp']) * region)
        for i in range(hpcplen):
            hpcps.append(fileData['hpcp'][i][j])

        fileData['mean_hpcp_vector_' + str(region)].append(np.mean(hpcps))
        fileData['std_hpcp_vector_' + str(region)].append(np.std(hpcps))


def readDataset(dataDir,filename, modality, region):
    
    '''
    INPUT:
    
    dataDir(str) : Directory of the Pickle files.
    
    filename(str) : Name of the Pickle file which contains the features and ground truth
    
    modality (str) : Modality type present in the music tradition of analysis
    
    region (float) : The first 'x' portion of songs to be analyzed. Default = 0.3
    
    computeRegional(boolean) : If 1, local features will be added to the feature set. If 0, include only the global features
    
    OUTPUT : 
    
    dataList (list of dicts) : The data structure which has all the necessary information for automatic classification
    
    modalitySet (set) : List of 
    
    '''
    
    with open(os.path.join(dataDir,filename) , 'rb') as f:
        dataFiles = pickle.load(f)  # !!!!!!!!!!!!!

    modalitySet = []
    dataList = [];
    
    if modality == 'makam':
    
        for datafile in dataFiles:  # control for empty dataStructures (files with no HPCP)
            if len(datafile['hpcp']) != 0:
                dataList.append(datafile)
                modalitySet.append(datafile[modality])
                
        params = AnalysisParams(200, 100, 'hann', 2048, 44100)
        
        for i in range(len(dataList)):
            computeHPCP_Regional(dataList[i], region)
        print('PART_' + str(region) + '__COMPLETE \n')        
            
    if modality == 'tab':
        for datafile in dataFiles:
            for section in datafile['section']:
                dataList.append(datafile)
                modalitySet.append(section[modality])
            
            
    modalitySet = set(modalitySet)        
    
    return dataList, modalitySet


def generateCSV(targetDir, data, region, featureSet, modality, combined):
    '''
    INPUT : 
    
    targetDir (str) : directory of the workspace
    
    data : The list of dictionaries that contains all the data for classification
    
    region (float) : The first 'x' portion of songs to be analyzed. Default = 0.3 (ONLY APPLICABLE FOR MAKAM TRADITION)
    
    featureSet (str) : Set of features to be included for classification ('mean', 'std', 'all')
    
    modality (str) : Modality type present in the music tradition of analysis
    
    combined (boolean) : Perform classification using the combination of local and global features. (ONLY APPLICABLE FOR MAKAM TRADITION)
    
    OUTPUT : 
    
    CSV file with the proper format for MachineLearning steps
    
    '''

    numBin = data[0]['numBins'] ### TODO - writer better, fileData[numbin]
    
    if modality == 'makam':
    
        if combined == 0 :    

            fieldnames = ['name']
            if featureSet == 'mean' or featureSet == 'all':
                for i in range(numBin):
                    ind = str(i)
                    fieldnames.append('hpcp_mean_' + ind)
            if featureSet == 'std' or featureSet == 'all':
                for i in range(numBin):
                    ind = str(i)
                    fieldnames.append('hpcp_std_' + ind)
            modalityType = modality + 'Type'        
            fieldnames.append(modalityType)

            datasList = []
            datasList.append(fieldnames)
            for index in range(len(data)):  ##navigate in the dictionary of features
                tempList = []  # temporary List to put attributes for each audio slice (data-point)
                dataname = data[index]['fileName']
                tempList.append(dataname)
                if featureSet == 'mean' or featureSet == 'all':
                    for i in range(numBin):
                        tempList.append(data[index]['mean_hpcp_vector'][i])
                if featureSet == 'std' or featureSet == 'all':
                    for i in range(numBin):
                        tempList.append(data[index]['std_hpcp_vector'][i])

                tempList.append(data[index][modality])  # append modality types for classification  
                datasList.append(tempList)

            with open(targetDir + 'DataCSVforstage' + '_' + str(numBin) + 'bins_' + featureSet + '.csv',
                              'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(datasList)


        if combined == 1 :

            '''
            running this option will provide csv files that contains the combinations of local & global features

            The local region is defined by the parameter 'region'

            '''
            iterations = ['meanLocal+stdGlobal', 'stdLocal+meanGlobal','meanLocal+meanGlobal','stdLocal+stdGlobal']

            for iteration in iterations:
                fieldnames = ['name']
                for i in range(len(data[0]['mean_hpcp_vector'])):
                    ind = str(i)
                    fieldnames.append('hpcpmean_' + ind)
                for i in range(len(data[0]['mean_hpcp_vector'])):
                    ind = str(i)
                    fieldnames.append('hpcpstd_' + ind)

                fieldnames.append('makamType')
                datasList = []
                datasList.append(fieldnames)

                for index in range(len(data)):  ##navigate in dictionary
                    tempList = []  # temporary List to put attributes for each audio slice (data-point)
                    dataname = data[index]['fileName']
                    tempList.append(dataname)

                    if iteration == 'stdLocal+meanGlobal':
                        for i in range(len(data[0]['mean_hpcp_vector'])):
                            tempList.append(data[index]['mean_hpcp_vector'][i])
                        for i in range(len(data[0]['std_hpcp_vector'])):
                            tempList.append(data[index]['std_hpcp_vectorQ_'+str(region)][i])

                    elif iteration == 'meanLocal+stdGlobal':
                        for i in range(len(data[0]['mean_hpcp_vector'])):
                            tempList.append(data[index]['mean_hpcp_vectorQ_'+str(region)][i])
                        for i in range(len(data[0]['std_hpcp_vector'])):
                            tempList.append(data[index]['std_hpcp_vector'][i])

                    elif iteration == 'meanLocal+meanGlobal':
                        for i in range(len(data[0]['mean_hpcp_vector'])):
                            tempList.append(data[index]['mean_hpcp_vectorQ_'+str(region)][i])
                        for i in range(len(data[0]['mean_hpcp_vector'])):
                            tempList.append(data[index]['mean_hpcp_vector'][i])

                    elif iteration == 'stdLocal+stdGlobal':
                        for i in range(len(data[0]['std_hpcp_vector'])):
                            tempList.append(data[index]['std_hpcp_vectorQ_'+str(region)][i])
                        for i in range(len(data[0]['std_hpcp_vector'])):
                            tempList.append(data[index]['std_hpcp_vector'][i])

                    tempList.append(data[index][modality])  # append scales for classification

                    datasList.append(tempList)
                with open(targetDir + 'DataCSVforstage' + '_' + str(numBin) + 'bins_' + iteration + '.csv',
                          'w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(datasList)
                    
    elif modality == 'tab' : 
        
        fieldnames = ['name']
        if featureSet == 'mean' or featureSet == 'all':
            for i in range(numBin):
                ind = str(i)
                fieldnames.append('hpcp_mean_' + ind)
        if featureSet == 'std' or featureSet == 'all':
            for i in range(numBin):
                ind = str(i)
                fieldnames.append('hpcp_std_' + ind)
        modalityType = modality + 'Type'        
        fieldnames.append(modalityType)

        datasList = []
        datasList.append(fieldnames)
        for index in range(len(data)):  ##navigate in the dictionary of features
            for section in data[index]['section']:
                
                tempList = []  # temporary List to put attributes for each audio slice (data-point)
                dataname = data[index]['fileName']
                tempList.append(dataname)
                if featureSet == 'mean' or featureSet == 'all':
                    for i in range(numBin):
                        tempList.append(section['mean_hpcp_vector'][i])
                if featureSet == 'std' or featureSet == 'all':
                    for i in range(numBin):
                        tempList.append(section['std_hpcp_vector'][i])

                tempList.append(section[modality])  # append modality types for classification  
                datasList.append(tempList)

        with open(targetDir + 'DataCSVforstage' + '_' + str(numBin) + 'bins_' + featureSet + '.csv','w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(datasList)


def hyperparameterOptimization(X_train, Y_train, nfolds):
    
    ##HYPERPARAMETER OPTIMIZATION USING GRID SEARCH WITH 10-fold CROSS VALIDATION

    ##HYPERPARAMETER SET:
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    ## APPLY CROSS N-FOLD CROSS VALIDATION, ITERATING OVER PARAMETER GRIDS

    grid_search = GridSearchCV(SVC(), param_grid, cv=nfolds, scoring='accuracy')
    grid_search.fit(X_train, Y_train)
    ## RETURN PARAMETERS WITH MOST ACCURACATE CROSS-VALIDATION SCORES
    return grid_search.best_params_


def machineLearning(targetDir, X, Y, attribute, numBin):
    f_measures = []
    accuracies = []
    Y_total = [];
    Y_pred_total = [];
    ## TO INCREASE GENERALIZATION POWER, THE TRAIN-VALIDATE-TEST PROCEDURE IS PERFORMED
    ## OVER 10 RANDOM INSTANCES.
    for randomseed in range(30, 50, 2):
        ## SPLIT DATASET INTO TRAIN_SET & TEST_SET, IMPORTANT ---> STRATIFY
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=randomseed)

        ##OPTIMIZE PARAMETERS ON TRAIN SET.
        ##IMPORTANT --> TEST SET CANNOT INFLUENCE THE MODEL. IF SO -> RISK OF OVERFITTING
        param = hyperparameterOptimization(X_train, Y_train, 10)

        ## CREATE A PIPELINE
        estimators = []
        ##CREATE THE MODEL WITH OPTIMIZED PARAMETERS
        model1 = SVC(C=param['C'], gamma=param['gamma'])
        estimators.append(('classify', model1))
        model = Pipeline(estimators)
        ## TRAIN MODEL WITH TRAINING SET & PREDICT USING X_TEST_FEATURES
        Y_pred = model.fit(X_train, Y_train).predict(X_test)

        ##EVALUATION
        ## TEST PREDICTED VALUES WITH GROUND TRUTH (Y_TEST)
        accscore = accuracy_score(Y_test, Y_pred)
        #print('Accuracy Measure = ')
        #print(accscore)

        f_measure = f1_score(Y_test, Y_pred, average='weighted')
        #print('F Measure = ')
        #print(f_measure)

        f_measures.append(f_measure)
        accuracies.append(accscore)
        for i in Y_test:
            Y_total.append(i)
        for i in Y_pred:
            Y_pred_total.append(i)

    ##AVERAGE ALL RANDOM SEED ITERATIONS FOR GENERALIZATION
    print('F-measure (mean,std) --- FINAL')
    f = round(np.mean(f_measures) ,2)
    fstd = np.std(f_measures)
    print(f,fstd )
    print('Accuracy (mean,std) FINAL')
    ac = round(np.mean(accuracies), 2)
    accstd=np.std(accuracies)
    print(ac,accstd)
    cm = confusion_matrix(Y_total, Y_pred_total)

    ## -------------------------------------------------------------

    with open(targetDir + 'scores' + attribute + '_' + str(numBin) + '.txt', 'w') as scorefile:
        scorefile.write(str(f))
        scorefile.write(str(ac))
    return cm, f_measures, accuracies


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    plt.figure(figsize=(9, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title, fontsize=26)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.tick_params(labelsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", size=20)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=24)
    plt.xlabel('Predicted label', fontsize=24)


