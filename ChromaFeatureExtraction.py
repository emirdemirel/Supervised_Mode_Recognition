import json, os, sys, pickle, csv
import modalityUtils as mu
import argparse


####################################################################################################
### MAIN FUNCTION ###

MusicTradition, numBins, outputDir = mu.get_args()

if MusicTradition == 'TurkishClassicalMusic':
    
    ### THE MODALITY CONCEPT IN TURKISH CLASSICAL MUSIC TRADITION
    modality = 'makam'
    
    ### dataDir :THE DIRECTORY OF THE AUDIO FILES
    dataDir = '../data/compMusicDatasets/turkishMakam/'
    annotationsFile = 'annotations.json'
    
elif MusicTradition == 'HindustaniClassicalMusic':
    
    ### THE MODALITY CONCEPT IN HINDUSTANI CLASSICAL MUSIC TRADITION
    modality = 'rag'
        
elif MusicTradition == 'CarnaticClassicalMusic':
    
    ### THE MODALITY CONCEPT IN CARNATIC CLASSICAL MUSIC TRADITION
    modality = 'raaga'    
    
elif MusicTradition == 'Jazz':
    
    ### THE MODALITY CONCEPT IN JAZZ TRADITION
    modality = 'chordscale'
    
elif MusicTradition == 'ArabAndalusianMusic':
    
    ### THE MODALITY CONCEPT IN ARAB-ANDALUSIAN MUSIC TRADITION
    modality = 'tab'   
    annotationsFile = 'dataset_77_tab_tonic.json'
    ### dataDir :THE DIRECTORY OF THE AUDIO FILES

    dataDir = '../andalusian-corpora-notebooks/'
    
print('Analysis on '+ MusicTradition + ' Tradition.\n')
print('Number of bins per octave in the Chroma Vectors : ' + str(numBins))

'''   
### LOAD THE JSON FILE THAT HAS THE ANNOTATIONS AND MBIDS
with open(dataDir + annotationsFile) as json_data:
    collectionFiles = json.load(json_data)

dataList, modalityCategories = mu.createDataStructure(dataDir, collectionFiles, numBins, modality)

print('Modality categories in the dataset : \n')
print(modalityCategories, '\n')

print('Number of Categories in the dataset')
print(len(modalityCategories))
      
dataslist = mu.FeatureExtraction(outputDir,dataDir,dataList,modality)
'''