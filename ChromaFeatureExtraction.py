import json, os, sys
from modeUtils.utilities import *

### MAIN FUNCTION ###

def main(MusicTradition, numBins, dataDir):
    
    if MusicTradition == 'TurkishClassicalMusic':

        ### THE mode CONCEPT IN TURKISH CLASSICAL MUSIC TRADITION
        mode = 'Makam'
     
        annotationsFile = 'annotations.json'

    elif MusicTradition == 'HindustaniClassicalMusic':

        ### THE mode CONCEPT IN HINDUSTANI CLASSICAL MUSIC TRADITION
        mode = 'Rag'
        annotationsFile = 'annotations_hindustani.json'

    elif MusicTradition == 'CarnaticClassicalMusic':

        ### THE mode CONCEPT IN CARNATIC CLASSICAL MUSIC TRADITION
        mode = 'Raaga'    
        annotationsFile = 'annotations_carnatic.json'

    elif MusicTradition == 'Jazz':

        ### THE mode CONCEPT IN JAZZ TRADITION
        mode = 'ChordScale'

    elif MusicTradition == 'ArabAndalusianMusic':

        ### THE mode CONCEPT IN ARAB-ANDALUSIAN MUSIC TRADITION
        mode = 'Tab'   
        annotationsFile = 'dataset_77_tab_tonic.json'
        ### dataDir :THE DIRECTORY OF THE AUDIO FILES

    print('Analysis on '+ MusicTradition + ' Tradition.\n')
    print('Number of bins per octave in the Chroma Vectors : ' + str(numBins))
      
    ### LOAD THE JSON FILE THAT HAS THE ANNOTATIONS AND MBIDS
    with open(dataDir + annotationsFile) as json_data:
        collectionFiles = json.load(json_data)

    dataList, modeCategories = createDataStructure(dataDir, collectionFiles, numBins, mode)
    
    print('mode categories in the dataset : \n')
    print(modeCategories, '\n')

    print('Number of Categories in the dataset')
    print(len(modeCategories))
    
    dataslist = FeatureExtraction(outputDir,dataDir,dataList,mode)
    
#####################################################
    
if __name__ == "__main__":
    
    MusicTradition, numBins, outputDir = get_args_FeatureExtraction()
    main(MusicTradition, numBins, outputDir)