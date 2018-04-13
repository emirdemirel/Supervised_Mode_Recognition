from modeUtils.utilities import readDataset, generateCSV, get_args_pickle2CSV

### This function generates the CSV files with the proper format for the Machine Learning step. 
### ----> X = Features, Y = Class Labels (ground truth)

def main(filename, region, combined):
    
    dataDir = 'data/'
    modality = filename.split('trad')[0].split('for')[1]
    dataSet, modalitySet = readDataset(dataDir,filename, modality, region)   
    print('Modality types in this dataset:  \n')
    print(modalitySet) 
    
    generateCSV(dataDir,dataSet,region, modality, combined)

if __name__ == "__main__":
    
    filename, region, combined = get_args_pickle2CSV()    
    main(filename, region, combined)