import sys, json, os, pickle, csv, itertools, argparse
import numpy as np
import essentia.standard as ess
import matplotlib.pyplot as plt
import pandas as pd

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

from external_utilities.pitchfilter import PitchFilter
from external_utilities.toniclastnote import TonicLastNote
from external_utilities.predominantmelodymakam import PredominantMelodyMakam


def FeatureExtraction_Recording(recording,params):
    
    numBins = params.numbins
    fs = params.fs
    # LOAD Audio file
    Audio = ess.MonoLoader(filename = recording.path, sampleRate = fs)()
    Audio = ess.DCRemoval()(Audio)  # PREPROCESSING / DC removal 
    Audio = ess.EqualLoudness()(Audio) # PREPROCESSING - Equal Loudness Filter
    
    # Windowing Parameters (first converting from msec to number of samples)
    # assuring windowSize and hopSize are even
    windowSize = round(fs * params.windowSize / 1000); windowSize = int(windowSize / 2) * 2
    hopSize = round(fs * params.hopSize / 1000); hopSize = int(hopSize / 2) * 2         
    
    tonic = float(recording.tonic)    

    # FRAME-BASED Spectral Analysis
    hpcp = [];
    for frame in ess.FrameGenerator(Audio, frameSize=windowSize, hopSize=hopSize, startFromZero=True):
        frame = ess.Windowing(size=windowSize, type=params.windowFunction)(frame)        
        mX = ess.Spectrum(size=windowSize)(frame)
        mX[mX < np.finfo(float).eps] = np.finfo(float).eps
         # EXTRACT frequency and magnitude information of the harmonic spectral peaks
        freq, mag = ess.SpectralPeaks()(mX) 
        # harmonic pitch-class profiles
        hpcp.append(ess.HPCP(normalized='unitSum', referenceFrequency=tonic, size=numBins, windowSize=12 / numBins)(freq,mag))  
    recording.chroma_framebased = np.array(hpcp)
    
    # FEATURE SUMMARIZATION
    mean_chroma = [];  # global Mean of HPCP vectors
    std_chroma = [];  # global standard deviation of HPCP vectors    
    for j in range(numBins):
        tmp = [];
        for i in range(len(recording.chroma_framebased)):
            tmp.append(recording.chroma_framebased[i][j])           
        mean_chroma.append(np.mean(tmp)); std_chroma.append(np.std(tmp))       
    recording.chroma_mean = mean_chroma; recording.chroma_std = std_chroma 
    
def Compute_LocalFeatures(recording, params):    
    '''
    INPUT :
    
    fileData (dict): Dictionary that contains all the necessary information of the audio for classification
    
    region (float) : local region of the song
    '''       
    
    mean = []; std = []
    len_local = int(len(recording.chroma_framebased) * float(params.local_region))
    for j in range(params.numbins):
        local_chromas = []        
        for i in range(len_local):
            local_chromas.append(recording.chroma_framebased[i][j])
        mean.append(np.mean(local_chromas)); std.append(np.std(local_chromas))
        
    setattr(recording,'chroma_mean_local', mean)
    setattr(recording,'chroma_std_local', std)
    
############

def GenerateCSVFile(Collection, featureSet):
    
    #FEATURE_SETS = ['chroma_mean', 'chroma_std']
    numBins = Collection.analysis_parameters.numbins

    DataList4CSV = []        
    #FIELDNAMES
    fieldnames = ['MBIDs']
    for features in featureSet:
        for i in range(numBins):
            fieldnames.append(features + '_Bin_#' + str(i))
    classLabel = Collection.mode + 'Type(Class)'        
    fieldnames.append(classLabel)
    DataList4CSV.append(fieldnames)
        
    for recording in Collection.recordings: 
        recording_data = [recording.mbid] 
        if 'chroma_mean' in featureSet:
            for i in range(numBins):
                recording_data.append(recording.chroma_mean[i])
        if 'chroma_std' in featureSet:
            for i in range(numBins):
                recording_data.append(recording.chroma_std[i])
        if 'chroma_mean_local' in featureSet:
            for i in range(numBins):
                recording_data.append(recording.chroma_mean_local[i])
        if 'chroma_std_local' in featureSet:
            for i in range(numBins):
                recording_data.append(recording.chroma_std_local[i])        
                
        recording_data.append(recording.modeClass)  # append mode types for classification  
        DataList4CSV.append(recording_data)
    
    with open(Collection.path + 'FeatureData_w_' + str(numBins) + 'ChromaBins.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(DataList4CSV)
        
    setattr(Collection,'csv_filename','FeatureData_w_' + str(numBins) + 'ChromaBins.csv')    
            
def ReadCSVFile(Collection):
    
    df = pd.read_csv(os.path.join(Collection.path,Collection.csv_filename))
    df.pop('MBIDs'); data_class=df.pop(Collection.mode + 'Type(Class)')
    X = df; y = data_class
    return X, y

############


def hyperparameterOptimization(X_train, Y_train, nfolds):
    
    # HYPERPARAMETER OPTIMIZATION USING GRID SEARCH WITH 10-fold CROSS VALIDATION

    # PARAMETER SET:
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    # APPLY CROSS N-FOLD CROSS VALIDATION, ITERATING OVER PARAMETER GRIDS
    grid_search = GridSearchCV(SVC(), param_grid, cv=nfolds, scoring='accuracy')
    
    grid_search.fit(X_train, Y_train)
    # RETURN PARAMETERS WITH MOST ACCURACATE CROSS-VALIDATION SCORES
    return grid_search.best_params_

def MachineLearningPipeline(SupervisedLearning):

    N_FOLDS = 10
    
    X = SupervisedLearning.features; y = SupervisedLearning.classes
    
    f_measures = []; accuracies = []
    y_total = []; y_pred_total = []
    # TO INCREASE GENERALIZATION POWER, THE TRAIN-VALIDATE-TEST PROCEDURE IS PERFORMED
    # OVER 10 RANDOM INSTANCES.
    for randomseed in range(30, 50, 2):
        ## SPLIT DATASET INTO TRAIN_SET & TEST_SET, IMPORTANT ---> STRATIFY
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=randomseed)
        # OPTIMIZE parameters on the train set
        param = hyperparameterOptimization(X_train, y_train, N_FOLDS)
        # CREATE a Machine Learning Pipeline
        pipeline = []
        # CREATE the classifier with optimized parameters
        Classifier = SVC(C=param['C'], gamma=param['gamma'])
        # APPEND the classifier to the pipeline               
        pipeline.append(('classify', Classifier))
        model = Pipeline(pipeline)                       
        # TRAIN MODEL WITH TRAINING SET & PREDICT USING X_TEST_FEATURES
        y_pred = model.fit(X_train, y_train).predict(X_test)
        # TEST PREDICTED VALUES WITH GROUND TRUTH (Y_TEST)
        accscore = accuracy_score(y_test, y_pred)
        f_measure = f1_score(y_test, y_pred, average='weighted')
        f_measures.append(f_measure); accuracies.append(accscore)
        
        # APPEND to the containers for confusion matrix
        for elem in y_test:
            y_total.append(elem)
        for elem in y_pred:
            y_pred_total.append(elem)
            
    SupervisedLearning.classifier = Classifier
    
    # AVERAGE ALL RANDOM SEED ITERATIONS FOR GENERALIZATION
    f_score = round(np.mean(f_measures) ,2); f_score_std = np.std(f_measures)    
    accuracy = round(np.mean(accuracies), 2); accuracy_std=np.std(accuracies)
    
    SupervisedLearning.scores_fMeasure = (f_score, f_score_std)
    SupervisedLearning.scores_accuracy = (accuracy, accuracy_std)
    
    SupervisedLearning.confusion_matrix = confusion_matrix(y_total, y_pred_total)
  
    
    return True  

###############
             
def TrainClassifier(SupervisedLearning):    
    
    N_FOLDS = 10
    X = SupervisedLearning.features; y = SupervisedLearning.classes
    
    
    param = hyperparameterOptimization(X, y, N_FOLDS)
    pipeline = []
    # CREATE the classifier with optimized parameters
    Classifier = SVC(C=param['C'], gamma=param['gamma'])
    # APPEND the classifier to the pipeline               
    pipeline.append(('classify', Classifier))
    model = Pipeline(pipeline)                       
    # TRAIN MODEL WITH TRAINING SET & PREDICT USING X_TEST_FEATURES
    trained_model = model.fit(X, y)
    
    return trained_model

def DetectTonicLastNote(Recording):
    
    extractor = PredominantMelodyMakam(filter_pitch=False)
    pitch_hz = extractor.run(Recording.mbid + '.mp3')['pitch']
    pitch_filter = PitchFilter()
    pitch_hz = pitch_filter.run(pitch_hz)

    tonic_identifier = TonicLastNote()
    tonic, _, _, _ = tonic_identifier.identify(pitch_hz)
    Recording.tonic = tonic['value']

def PredictMakam_Recording(SupervisedLearning, Recording,Params):
    
    Recording.extract_features(Params)
    Recording.extract_local_features(Params)
    
    FeaturesRecording = Recording.chroma_std
    FeaturesRecording.extend(Recording.chroma_std_local)
    FeaturesRecording = np.array(FeaturesRecording)
    return SupervisedLearning.trained_model.predict(FeaturesRecording.reshape(1, -1))