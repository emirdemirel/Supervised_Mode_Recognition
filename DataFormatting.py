from modeUtils.utilities import readDataset, generateCSV, get_args_pickle2CSV

### This function generates the CSV files with the proper format for the Machine Learning step. 
### ----> X = Features, Y = Class Labels (ground truth)

def main(filename, region, combined):
    
    
    mode = filename.split('trad')[0].split('for')[1]
    print(mode)
    if mode == 'Makam':
        dataDir = 'data/Turkish/'
    elif mode == 'Rag':
        dataDir = 'data/Hindustani/'
    elif mode == 'Raaga':
        dataDir = 'data/Carnatic/'    
    
    
    dataSet, modeSet = readDataset(dataDir,filename, mode, region)
    
    print('Mode types in this dataset:  \n')
    print(modeSet) 
    
    generateCSV(dataDir,dataSet,region, mode, combined)

if __name__ == "__main__":
    
    filename, region, combined = get_args_pickle2CSV()    
    main(filename, region, combined)