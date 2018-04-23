import sys
import json, os, pickle, csv, itertools, argparse
import numpy as np
import essentia.standard as ess
import matplotlib.pyplot as plt
import pandas as pd

from pandas import DataFrame, read_csv
from scipy.stats import kurtosis
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import IPython.display as ipd
from compmusic import dunya
import essentia
essentia.log.warningActive=False

########### ARGUMENT PARSING ###########

def get_args_FeatureExtraction():
    parser = argparse.ArgumentParser(
        description='A tool for Chroma (HPCP) Feature Extraction using Essentia library. (FEATURE EXTRACTION)')
    parser.add_argument(
        '-t', '--tradition', type=str ,help='Input music tradition to perform the mode classification task', action = 'store', required = True)
    parser.add_argument(
        '-n', '--numberofBins', type = int, help='Input number of bins per octave in chroma vectors', required = True, action='store')
    parser.add_argument(
        '-o', '--output_directory', type=str, help='Output directory for the pickle file that contains the dataset with the extracted features', required = True)

    args = parser.parse_args()
    musicTradition = args.tradition
    numBins = args.numberofBins
    outDir = args.output_directory

    return musicTradition, numBins, outDir


def get_args_pickle2CSV():
    
    parser = argparse.ArgumentParser(
        description='A tool for converting the feature data into proper format for the machine learning pipeline')
    parser.add_argument(
        '-i', '--input_filename', type=str, help='Name of the input pickle file which is the output file of ChromaFeatureExtraction.py', required = True)  
    parser.add_argument(
        '-r', '--region', type=float ,help='Input region of audios (from the beginning) for extracting local features. values = [0,1] (region = 1 is equivalent of using the Global Features)', action = 'store', required = True)        
    parser.add_argument(
        '-c', '--combined', type = bool, help='SET combined = 1 FOR PERFORMING CLASSIFICATION USING THE COMBINATION OF LOCAL AND GLOBAL FEATURES; combined = 0 FOR USING ONLY THE LOCAL FEATURES (except the case region = 1, which ALREADY corresponds to global features)', required = True, action='store')
    
    args = parser.parse_args()
    filename = args.input_filename
    region = args.region
    combined = args.combined
    
    return filename, region, combined


def get_args_Classification():
    
    parser = argparse.ArgumentParser(
        description='A tool for Chroma (HPCP) Feature Extraction using Essentia library.(CLASSIFICATION)')
    parser.add_argument(
        '-i', '--input_filename', type=str, help='Name of the input CSV file which is the output file of DataFormatting.py', required = True)
    parser.add_argument(
        '-r', '--region', type=float ,help='Input region of audios (from the beginning) for extracting local features \n, SELECT THE REGION OF THE SONG TO BE ANALYZED LOCALLY. (region = 1 is equivalent of using the Global Features)', action = 'store', required = True)        
    parser.add_argument(
        '-m', '--mode', type=str ,help='Type of mode specific for the music tradition of analysis.', action = 'store', required = True)
    
    args = parser.parse_args()
    filename = args.input_filename
    region = args.region
    mode = args.mode

    return filename, region, mode

############# DOWNLOAD FUNCTIONS ###############

def downloadDataset(annotationsFile, dataDir):
    '''
    annotationsFile (.json) = JSON file with tonic and mode annotations, and recording mbids
    targetDir (str) = target output directory for the dataset
    '''

    if os.path.exists(dataDir)!=1:         #In case of the directory already exists
        os.mkdir(dataDir)
    
    with open(dataDir+annotationsFile) as json_data: ##ninemakams.json is specific for the case
        collectionFiles = json.load(json_data)
    #print(collectionFiles)    
    modes=set()
    for file in collectionFiles:
        if '/' in file['mode']:                     #### for the case when there exists two different names for the same mode type            
            file['mode'] = file['mode'].split('/')[1]
            modes.add(file['mode'])
        else:
            modes.add(file['mode'])
    print(modes)       
    #Create directories for makams and download one recording for each
    for mode in modes:
        if os.path.exists(dataDir+mode)!=1:         #In case of the directory already exists
            os.mkdir(dataDir+mode)
        for file in collectionFiles:
            if(file['mode']==mode):
                musicbrainzid=file['mbid'].split('https://musicbrainz.org/recording/')[-1]
                fileset = []
                for file in set(os.listdir(dataDir+mode)):
                    fileset.append(file.split('.')[0])
                fileset=set(fileset)
                #print('Downloading recording : ')
                if not musicbrainzid in fileset:
                    contents = dunya.docserver.get_mp3(musicbrainzid)
                    name = "%s.mp3" % (musicbrainzid)
                    path = os.path.join(dataDir+mode+'/', name)
                    print(name)
                    open(path, "wb").write(contents)            

    print('Dataset downloaded and created in ' + dataDir + ' folder')    

############ DATA HANDLING FOR FEATURE EXTRACTION STEP #############

# Container for analysis parameters
class AnalysisParams:
    def __init__(self, windowSize, hopSize, windowFunction, fftN, fs):
        '''
        windowSize: milliseconds,
        hopSize: milliseconds,
        windowFunction: str ('blackman','hanning',...)
        fftN: int
        '''
        self.windowSize = windowSize
        self.hopSize = hopSize
        self.windowFunction = windowFunction
        self.fftN = fftN
        self.fs = fs


def initiateData4File(file, root):
    '''Forming the data structure for file
    Parameters
    ----------
    file,root : str
        File name and path info

    Returns
    -------
    fileData : dict
        Dictionary containing all info and data for the file
    '''
    fileData = dict();
    # Necessary info about data
    fileData['path'] = root;
    fileData['mbid'] = file;fileData['artist'] = [];
    fileData['fileName'] = [];  # fileData['classType'] = [];
    # Tonal features
    fileData['hpcp'] = [];
    fileData['mean_hpcp_vector'] = [];  # global Mean of HPCP vectors
    fileData['std_hpcp_vector'] = [];  # global standard deviations of HPCP vectors
    fileData['numBins'] = [];  # number of Bins of HPCP vectors
    # data from annotations    
    fileData['tonic'] = [];
    
    return fileData

def getRecordingMetaData(mbid):
    title = dunya.conn._dunya_query_json("api/makam/recording/" + mbid)['title']
    artist = dunya.conn._dunya_query_json("api/makam/recording/" + mbid)['artists'][0]['name']

    return title, artist

def createDataStructure(dataDir,collectionFiles,numBins,mode):
    
    modeCategories = []    
    dataList = []
            
    for index in range(len(collectionFiles)):            
        filename = collectionFiles[index]['mbid'].split('recording/')[1]
        fileData = initiateData4File(filename,dataDir)    
        fileData['tonic'] = collectionFiles[index]['tonic']                
        
        fileData['numBins']=numBins
        
        if '/' in collectionFiles[index]['mode']:       #### for the case when there exists two different names for the same mode type            
            fileData['mode'] = collectionFiles[index]['mode'].split('/')[1]            
        else:
            fileData['mode'] = collectionFiles[index]['mode']
            
        fileData['path'] = fileData['path']+fileData['mode']+'/'
        
        modeCategories.append(fileData['mode'])
        dataList.append(fileData)
    '''
    if mode == 'tab':
        
        for index in range(len(collectionFiles)):
                
            filename = collectionFiles[index]['mbid']
            fileDir = dataDir+'data/documents/'+filename+'/'
            fileData = initiateData4File(filename,fileDir)
            
            fileData['nawba'] =collectionFiles[index]['nawba']
            fileData['section'] = []
            for section in collectionFiles[index]['section']:
                fileData['section'].append(section)
                modeCategories.append(section['mode']) 
            fileData['path'] = fileData['path']
            fileData['numBins']=numBins  
            
            dataList.append(fileData)
    '''
    modeCategories = set(modeCategories)
    
    return dataList, modeCategories

############# FEATURE EXTRACTION FUNCTIONS ###########

def computeHPCP(x, windowSize, hopSize, params, tonic, numBin):
    # Initializing lists for features
    hpcp = [];
    # Main windowing and feature extraction loop
    for frame in ess.FrameGenerator(x, frameSize=windowSize, hopSize=hopSize, startFromZero=True):
        frame = ess.Windowing(size=windowSize, type=params.windowFunction)(frame)
        mX = ess.Spectrum(size=windowSize)(frame)
        mX[mX < np.finfo(float).eps] = np.finfo(float).eps

        freq, mag = ess.SpectralPeaks()(mX)  # extract frequency and magnitude information by finding the spectral peaks
        tunefreq, tunecents = ess.TuningFrequency()(freq, mag)
        # harmonic pitch-class profiles
        hpcp.append(ess.HPCP(normalized='unitSum', referenceFrequency=tonic, size=numBin, windowSize=12 / numBin)(freq,mag))   

    return hpcp

def computeHPCPFeatures(fileData, params, numBin,mode): ###change name!!!!!!!!!
    # Reading the wave file
    fs = params.fs

    x = ess.MonoLoader(filename=os.path.join(fileData['path'], fileData['fileName']), sampleRate=fs)()
    x = ess.DCRemoval()(x)  ##preprocessing / apply DC removal for noisy regions
    x = ess.EqualLoudness()(x)
    # Windowing (first converting from msec to number of samples)
    windowSize = round(fs * params.windowSize / 1000);
    windowSize = int(windowSize / 2) * 2  # assuring window size is even
    hopSize = round(fs * params.hopSize / 1000);
    hopSize = int(hopSize / 2) * 2  # assuring hopSize is even
    #print(len(x)/fs) #duration
    
    
    tonic = fileData['tonic']
    HPCPs = computeHPCP(x, windowSize, hopSize, params, tonic, numBin)
    fileData['hpcp'] = np.array(HPCPs)
    
    '''    
    elif mode == 'tab':
        for section in fileData['section']:
            startSample = section['start_time']*fs
            endSample = section['end_time']*fs
            x_section = x[startSample:endSample]
            tonic = section['tonic']
            HPCPs = computeHPCP(x, windowSize, hopSize, params, tonic, numBin)
            section['hpcp'] = np.array(HPCPs)
    '''
    
def computeGlobHPCP(fileData,numBins):    
    '''
    INPUT :
    
    fileData (dict): Dictionary that contains all the necessary information of the audio for classification
    
    numBins (int) : number of Bins per octave in HPCP vectors
    
    region (float) : local region of the song
    '''
   
    fileData['mean_hpcp_vector'] = [];  # global Mean of HPCP vectors
    fileData['std_hpcp_vector'] = [];  # global standard deviation of HPCP vectors
    
    for j in range(numBins):
        hpcps = [];
        for i in range(len(fileData['hpcp'])):
            hpcps.append(fileData['hpcp'][i][j]) 

        fileData['mean_hpcp_vector'].append(np.mean(hpcps))
        fileData['std_hpcp_vector'].append(np.std(hpcps))
              
def FeatureExtraction(dataDir,outDir,dataList, mode):
    
    params = AnalysisParams(200,100,'hann',2048,44100)
    numBins = dataList[0]['numBins']
    i = 1
    fs = params.fs
    print('Feature Extraction in Process. This might take a while...')    
           
    for ind in range(len(dataList)):
        if dataList[ind]['mode']!=dataList[ind-1]['mode']:
            print('extracting Features for ' + mode +' : ',dataList[ind]['mode'])  
        songname = dataList[ind]['mbid']  
        dataList[ind]['fileName'] = songname+'.mp3'
           
        computeHPCPFeatures(dataList[ind],params,numBins,mode)
        computeGlobHPCP(dataList[ind],numBins)
    '''
    if mode == 'tab':        
        for ind in range(len(dataList)):            
            songname = dataList[ind]['mbid']  
            dataList[ind]['fileName'] = songname+'.mp3'
            computeHPCPFeatures(dataList[ind],params,numBins,mode)
            for section in dataList[ind]['section']:
                computeGlobHPCP(section,numBins)         
                print('Section ',i)
                i = i + 1
    '''
    
    ### Saving all results in a pickle file
    pickleProtocol=1 #choosen for backward compatibility
    with open(outDir+'extractedFeatures_for'+mode+'tradition('+str(numBins)+'bins).pkl' , 'wb') as f:
        pickle.dump(dataList, f, pickleProtocol)
    print('Features are extracted and saved in a pickle file located in '+outDir+' directory')
    
    return dataList 

####### --- REGIONAL FEATURE COMPUTATION FOR CLASSIFICATION STEP #######

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


############# DATA HANDLING FOR CLASSIFICATION STEP ###################
        
def readDataset(dataDir,filename, mode, region):
    
    '''
    INPUT:
    
    dataDir(str) : Directory of the Pickle files.
    
    filename(str) : Name of the Pickle file which contains the features and ground truth
    
    mode (str) : mode type present in the music tradition of analysis
    
    region (float) : The first 'x' portion of songs to be analyzed. Default = 0.3
    
    computeRegional(boolean) : If 1, local features will be added to the feature set. If 0, include only the global features
    
    OUTPUT : 
    
    dataList (list of dicts) : The data structure which has all the necessary information for automatic classification
    
    modeSet (set) : List of 
    
    '''
    
    with open(os.path.join(dataDir,filename) , 'rb') as f:
        dataFiles = pickle.load(f)  # !!!!!!!!!!!!!

    modeSet = []
    dataList = [];
    
    if mode == 'Makam' or mode == 'Rag' or mode == 'Raaga':
    
        for datafile in dataFiles:  # control for empty dataStructures (files with no HPCP)
            if len(datafile['hpcp']) != 0:
                dataList.append(datafile)
                modeSet.append(datafile['mode'])
                
        params = AnalysisParams(200, 100, 'hann', 2048, 44100)     
        
        for i in range(len(dataList)):
            computeHPCP_Regional(dataList[i], region)
            
        print('computing Local Features for the first' + str(region) + 'region of the audio files is COMPLETE \n')
            
    elif mode == 'Tab':
        for datafile in dataFiles:
            for section in datafile['section']:
                dataList.append(datafile)
                modeSet.append(section['mode'])
                        
    modeSet = set(modeSet)        
    
    return dataList, modeSet


def generateCSV(targetDir, data, region, mode, combined):
    '''
    INPUT : 
    
    targetDir (str) : directory for CSV files to be generated
    
    data : The list of dictionaries that contains all the data for classification
    
    region (float) : The first 'x' portion of songs to be analyzed. Default = 0.3 (ONLY APPLICABLE FOR MAKAM TRADITION)
    
    mode (str) : mode type present in the music tradition of analysis
    
    combined (boolean) : Perform classification using the combination of local and global features. (ONLY APPLICABLE FOR MAKAM TRADITION)
    
    OUTPUT : 
    
    CSV file with the proper format for MachineLearning steps
    
    '''
    
    numBin = data[0]['numBins'] ### TODO - writer better, fileData[numbin]
    
    if mode == 'Makam' or mode == 'Rag' or mode == 'Raaga':
    
        if combined == False :
            
            featureSets = ['mean','std','all']
            for featureSet in featureSets:
                fieldnames = ['name']
                if featureSet == 'mean' or featureSet == 'all':
                    for i in range(numBin):
                        ind = str(i)
                        fieldnames.append('hpcp_mean_' + ind)
                if featureSet == 'std' or featureSet == 'all':
                    for i in range(numBin):
                        ind = str(i)
                        fieldnames.append('hpcp_std_' + ind)
                modeType = mode + 'Type'        
                fieldnames.append(modeType)

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

                    tempList.append(data[index]['mode'])  # append mode types for classification  
                    datasList.append(tempList)

                with open(targetDir + 'DataCSVforstage' + '_' + str(numBin) + 'bins_' + featureSet + '.csv',
                                  'w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(datasList)

                print('Generating CSV file for the features ' + featureSet + ' is COMPLETE')


        if combined == True :

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

                modeType = mode + 'Type'        
                fieldnames.append(modeType)
                
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
                            tempList.append(data[index]['std_hpcp_vector_'+str(region)][i])

                    elif iteration == 'meanLocal+stdGlobal':
                        for i in range(len(data[0]['mean_hpcp_vector'])):
                            tempList.append(data[index]['mean_hpcp_vector_'+str(region)][i])
                        for i in range(len(data[0]['std_hpcp_vector'])):
                            tempList.append(data[index]['std_hpcp_vector'][i])

                    elif iteration == 'meanLocal+meanGlobal':
                        for i in range(len(data[0]['mean_hpcp_vector'])):
                            tempList.append(data[index]['mean_hpcp_vector_'+str(region)][i])
                        for i in range(len(data[0]['mean_hpcp_vector'])):
                            tempList.append(data[index]['mean_hpcp_vector'][i])

                    elif iteration == 'stdLocal+stdGlobal':
                        for i in range(len(data[0]['std_hpcp_vector'])):
                            tempList.append(data[index]['std_hpcp_vector_'+str(region)][i])
                        for i in range(len(data[0]['std_hpcp_vector'])):
                            tempList.append(data[index]['std_hpcp_vector'][i])

                    tempList.append(data[index]['mode'])  # append scales for classification

                    datasList.append(tempList)
                with open(targetDir + 'DataCSVforstage' + '_' + str(numBin) + 'bins_' + iteration + '.csv',
                          'w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(datasList)
                
                print('Generating CSV file for the features ' + iteration + ' is COMPLETE')
            
            
    elif mode == 'Tab' : 
        
        fieldnames = ['name']
        if featureSet == 'mean' or featureSet == 'all':
            for i in range(numBin):
                ind = str(i)
                fieldnames.append('hpcp_mean_' + ind)
        if featureSet == 'std' or featureSet == 'all':
            for i in range(numBin):
                ind = str(i)
                fieldnames.append('hpcp_std_' + ind)
        modeType = mode + 'Type'        
        fieldnames.append(modeType)

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

                tempList.append(section['mode'])  # append mode types for classification  
                datasList.append(tempList)

        with open(targetDir + 'DataCSVforstage' + '_' + str(numBin) + 'bins_' + featureSet + '.csv','w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(datasList)
            
############## MACHINE LEARNING & CLASSIFICATION ###############

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

    print('Accuracy score for the Feature Set ' + attribute + ' : ')
    
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
    plt.show()
    
#-------------------------------------------- END OF UTILITIES ---------------------------------------------#
