import sys, json, os, pickle, csv, itertools
import numpy as np
import essentia.standard as ess

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

import modeUtils.utils as U


class AnalysisParams:
    def __init__(self, windowSize, hopSize, windowFunction, fftN, fs, numbins, regions, local_region):
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
        self.numbins = numbins
        self.regions = []
        self.local_region = local_region
        
    def __repr__(self):
        return 'AnalysisParameters (windowSize={}, hopSize={}, windowFunction={},fftN = {}\
                fs = {}, numbins = {},regions = {}, local_region = {}' \
                .format(self.windowSize, self.hopSize,self.windowFunction, \
                        self.fftN,self.fs, self.numbins, self.regions, self.local_region)


class Collection(object):
    """Object for representing the dataset collection.
    
       Attributes
       ----------

       
       """
    def __init__(self,
                 path = None,
                 mode = None,
                 mode_categories = None,
                 list_recordings = [],
                 analysis_parameters = AnalysisParams,
                 dunya_token = None,
                 recordings = dict()):
        self.path = path
        self.mode = mode
        self.mode_categories = mode_categories
        self.list_recordings = list_recordings
        self.analysis_parameters = analysis_parameters       
        self.dunya_token = dunya_token
        self.recordings = recordings
        
    def __repr__(self):
        return 'Collection (path={}, mode={}, mode_categories={},\
                list_recordings = {}, analysis_parameters = {}' \
                .format(self.path, self.mode,self.mode_categories, len(self.list_recordings), \
                self.analysis_parameters)
      
    def set_dunya_token(self, token):
        self.dunya_token = token
        dunya.set_token(self.dunya_token) 
        
    def set_mode_type(self, mode_type):
        self.mode = mode_type
        
    def set_parameters(self, parameters = AnalysisParams):         
        self.analysis_parameters = parameters
        
    def get_recording(self,mbid):
        for recording in self.recordings:
            if mbid == recording.mbid:
                return recording    
                
    def extract_chroma_features(self):          
        for recording in self.recordings:
            recording.extract_features(self.analysis_parameters)           
        # SAVE all results in a pickle file
        pickleProtocol=1 #choosen for backward compatibility
        with open(self.path +'extractedFeatures_for' + self.mode + \
                  'tradition('+str(self.analysis_parameters.numbins)+'bins).pkl' , 'wb') as f:
            pickle.dump(self.recordings, f, pickleProtocol)
        print('Features are extracted and saved in a pickle file.')    
        
    def extract_local_chroma_features(self):    
        for recording in self.recordings:
            recording.extract_local_features(self.analysis_parameters)
    
    def load_features(self,filename):
        with open(filename , 'rb') as f:
            self.recordings = pickle.load(f)  
    
    def download_collection(self, annotation_file):
        
        '''
        annotationsFile (.json) = JSON file with tonic and mode annotations, and recording mbids
        targetDir (str) = target output directory for the dataset
        '''       
        dataDir = annotation_file.split('annotation')[0]
        self.path = dataDir
        if 'Turkish' in annotation_file:
            self.mode = 'Makam'
        elif 'Hindustani' in annotation_file:
            self.mode = 'Rag'
        elif 'Carnatic' in annotation_file:
            self.mode = 'Raaga'
        #In case of the directory already exists
        if os.path.exists(dataDir)!=1:              
            os.mkdir(dataDir)

        with open(annotation_file) as json_data: 
            collectionFiles = json.load(json_data)
        #print(collectionFiles)
        
        modes=set()
        for file in collectionFiles:
            # for the case when there exists two different names for the same mode type
            if '/' in file['mode']:                                 
                file['mode'] = file['mode'].split('/')[1]
                modes.add(file['mode'])
            else:
                modes.add(file['mode'])
        #print(modes)   
        mbids = []
        #Create directories for makams and download one recording for each        
        for mode in modes:            
            if os.path.exists(dataDir+mode)!=1:         
                os.mkdir(dataDir+mode)
                
            for file in collectionFiles:
                if(file['mode']==mode):                                                            
                    # CREATE Recording Class object for each recording
                    rec = Recording()
                    # GET MBIDs of each recording
                    rec.mbid = file['mbid'].split('https://musicbrainz.org/recording/')[-1]                        
                    filename = "%s.mp3" % (rec.mbid)
                    # PARSE information from annotations
                    rec.tonic = file['tonic']
                    rec.modeClass = file['mode']
                    
                    path = os.path.join(dataDir+mode,filename)
                    rec.path = path
                    
                    fileset = []
                    for file in set(os.listdir(dataDir+mode)):
                        fileset.append(file.split('.')[0])
                    fileset=set(fileset)                    
                    if not rec.mbid in fileset:
                        contents = dunya.docserver.get_mp3(rec.mbid)                        
                        #print('Downloading recording : ', mbid)
                        open(os.path.join(path,filename), "wb").write(contents) 
                        
                    self.recordings[rec] = rec    
                    mbids.append(rec.mbid)        

        self.mode_categories = modes
        self.list_recordings = mbids
        
        print('Dataset downloaded and created in ' + dataDir + ' folder')  
        
        return True    
    
class Recording(object):
    """Object for representing a recording.
        This object comprises features and related properties (transcription/segments etc)
       Attributes
       ----------
      
       
    """
    def __init__(self,
                 path = None,
                 mbid = None, 
                 title = None,
                 artist = None,
                 mode = 'Makam',
                 modeClass = None,
                 tonic = None,
                 chroma_framebased = None, 
                 chroma_mean = None,
                 chroma_std = None, 
                 predicted_modeClass = None):
        self.path = path
        self.mbid = mbid
        self.title = title
        self.artist = artist
        self.mode = mode
        self.modeClass = modeClass
        self.tonic = tonic
        self.chroma_framebased = chroma_framebased
        self.chroma_mean = chroma_mean
        self.chroma_std = chroma_std
        self.predicted_modeClass = predicted_modeClass
        
    def __repr__(self):
        return 'Recording (path = {}, mbid,={}, mode={}, modeClass={},tonic = {},\
                chroma_framebased = {}, chroma_mean = {},chroma_std = {},predicted_modeClass = {}'\
                .format(self.path, self.mbid, self.mode, self.modeClass, self.tonic, self.num_chroma_bins, \
                 self.chroma_framebased, self.chroma_mean, self.chroma_std, self.predicted_modeClass)
        
    def get_recording_metadata(self):
        self.title = dunya.conn._dunya_query_json("api/makam/recording/" + self.mbid)['title']
        self.artist = dunya.conn._dunya_query_json("api/makam/recording/" + self.mbid)['artists'][0]['name']
                            
    def extract_features(self, params):  
        U.FeatureExtraction_Recording(self,params)
    
    def extract_local_features(self, params):
        U.Compute_LocalFeatures(self, params)
        
    def load_recording(self, fileName, params):
        fs = params.fs
        # LOAD Audio file
        self.path = fileName
        Audio = ess.MonoLoader(filename = self.path, sampleRate = fs)()
        Audio = ess.DCRemoval()(Audio)  # PREPROCESSING / DC removal 
        Audio = ess.EqualLoudness()(Audio) # PREPROCESSING - Equal Loudness Filter
        self.tonic = ess.TonicIndianArtMusic()(Audio)
        
        
class SupervisedLearning(object):
    def __init__(self,
                 classifier = None,
                 feature_set = None,
                 features = None,
                 classes = None,
                 classTypes = None,
                 scores_fMeasure = None,
                 scores_accuracy = None,
                 confusion_matrix = None):
        self.classifier = classifier
        self.feature_set = feature_set
        self.features = features
        self.classes = classes
        self.classTypes = classTypes
        self.scores_fMeasure = scores_fMeasure
        self.scores_accuracy = scores_accuracy
        self.confusion_matrix = confusion_matrix
        
    def __repr__(self):
        return 'Recording (classifier = {}, feature_set,={}, features={}, classes={},\
                scores_fMeasure = {}, scores_accuracy = {},confusion_matrix = {}'\
                .format(self.classifier, self.feature_set, self.features, self.classes, \
                 self.scores_fMeasure, self.scores_accuracy, self.confusion_matrix)    
        
    def feature_selection(self, features4classification):        
        self.feature_set = features4classification
    
    def create_dataframe(self, Collection):
        U.GenerateCSVFile(Collection, self.feature_set)
        
    def read_dataframe(self,Collection):
        self.features, self.classes = U.ReadCSVFile(Collection)
        self.classTypes = set(self.classes)
        
    def evaluate_classifier(self):
        U.MachineLearningPipeline(self)
        
    def train_model(self):
        TrainedModel = U.TrainClassifier(self)
        setattr(self,'trained_model', TrainedModel)        
    
    def predict_mode_recording(self,Recording,AnalysisParams):        
        Recording.predicted_modeClass = U.PredictMode_Recording(self,Recording, AnalysisParams)        
        return Recording.predicted_modeClass