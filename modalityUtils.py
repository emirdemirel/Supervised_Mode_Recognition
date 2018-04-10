import json
import os
#from compmusic import dunya
import sys
import pickle
import csv
import numpy as np
import essentia.standard as ess
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


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

def createDataStructure(dataDir,collectionFiles,numBins,modality):
    
    modalityCategories = []    
    dataList = []
    
    if modality == 'makam':
    
        for index in range(len(collectionFiles)):            
            filename = collectionFiles[index]['mbid'].split('recording/')[1]
            fileData = initiateData4File(filename,dataDir)    
            fileData['tonic'] = collectionFiles[index]['tonic']
            fileData[modality] = collectionFiles[index][modality]
            fileData['path'] = fileData['path']+fileData[modality]+'/'
            fileData['numBins']=numBins
            
            modalityCategories.append(fileData[modality])
            dataList.append(fileData)
    
    if modality == 'tab':
        
        for index in range(len(collectionFiles)):
                
            filename = collectionFiles[index]['mbid']
            fileDir = dataDir+'data/documents/'+filename+'/'
            fileData = initiateData4File(filename,fileDir)
            
            fileData['nawba'] =collectionFiles[index]['nawba']
            fileData['section'] = []
            for section in collectionFiles[index]['section']:
                fileData['section'].append(section)
                modalityCategories.append(section[modality]) 
            fileData['path'] = fileData['path']
            fileData['numBins']=numBins  
            
            dataList.append(fileData)
    
    modalityCategories = set(modalityCategories)
    
    return dataList, modalityCategories
#---------------------------------------------------

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


def computeHPCPFeatures(fileData, params, numBin,modality): ###change name!!!!!!!!!
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
    print(len(x)/fs)
    
    if modality == 'makam':
        tonic = fileData['tonic']
        HPCPs = computeHPCP(x, windowSize, hopSize, params, tonic, numBin)
        fileData['hpcp'] = np.array(HPCPs)
        
    elif modality == 'tab':
        for section in fileData['section']:
            startSample = section['start_time']*fs
            endSample = section['end_time']*fs
            x_section = x[startSample:endSample]
            tonic = section['tonic']
            HPCPs = computeHPCP(x, windowSize, hopSize, params, tonic, numBin)
            section['hpcp'] = np.array(HPCPs)



def computeGlobHPCP(fileData,numBins):
   
    features = list(fileData.keys())
    '''
    features.remove('path');
    features.remove('name');
    features.remove('mbid');
    features.remove('artist')
    '''
    fileData['mean_hpcp_vector'] = [];  # global Mean of HPCP vectors
    fileData['std_hpcp_vector'] = [];  # global standard deviations of HPCP vectors
    
    for feature in features:
        if feature == 'hpcp':
            for j in range(numBins):
                hpcps = [];
                for i in range(len(fileData['hpcp'])):
                    hpcps.append(fileData['hpcp'][i][j]) 

                fileData['mean_hpcp_vector'].append(np.mean(hpcps))
                fileData['std_hpcp_vector'].append(np.std(hpcps))


                
def FeatureExtraction(dataDir,dataList, modality):
    
    params = AnalysisParams(200,100,'hann',2048,44100)
    numBins = dataList[0]['numBins']
    i = 1
    fs = params.fs
    print('Tracks in the dataset')
    
    if modality == 'makam':
        
        for ind in range(len(dataList)):

            if dataList[ind][modality]!=dataList[ind-1][modality]:
                    print('extracting Features for modality : ',dataList[ind][modality])  

            songname = dataList[ind]['mbid']  
            dataList[ind]['fileName'] = songname+'.mp3'
           
            computeHPCPFeatures(dataList[ind],params,numBins,modality)
            computeGlobHPCP(dataList[ind],numBins)

            print('Track ',i)
            i = i + 1
    
    if modality == 'tab':
        
        for ind in range(len(dataList)):
            
            songname = dataList[ind]['mbid']  
            dataList[ind]['fileName'] = songname+'.mp3'
            computeHPCPFeatures(dataList[ind],params,numBins,modality)
            for section in dataList[ind]['section']:
                computeGlobHPCP(section,numBins)         
                print('Section ',i)
                i = i + 1
                
               
            
            
    ### Saving all results in a pickle file
    pickleProtocol=1 #choosen for backward compatibility
    with open(dataDir+'extractedFeatures_for'+modality+'tradition('+str(numBins)+'bins).pkl' , 'wb') as f:
        pickle.dump(dataList, f, pickleProtocol)
            
    return dataList        

#---------------------------------------------------------------------------------------------------



