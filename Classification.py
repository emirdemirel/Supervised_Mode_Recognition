import pandas as pd
from modeUtils.utilities import *

import warnings
warnings.filterwarnings("ignore")

def main(filename, region, mode):
    
    dataDir = 'data/'   
    numBins = filename.split('_')[1].split('bins')[0]
    attribute = filename.split('bins_')[1].split('.')[0]    
    modeType = mode + 'Type' 
    
    print('This process might take a while (5-10 min) \n CROSS-VALIDATION & TRAINING ') 
    list_accuracy=[]
    
    df = pd.read_csv(os.path.join(dataDir,filename))
    df.pop('name'); dataclass=df.pop(modeType)
    X=df; Y=dataclass
    cm,acc,f = machineLearning(dataDir,X,Y,attribute,numBins)
        
    list_accuracy.append([cm,acc])

    best_model = max(enumerate(list_accuracy))[1]
    modalitySet = sorted(modalitySet)
    plot_confusion_matrix(best_model[0],modalitySet,normalize=False)
    
    
if __name__ == "__main__":
    
    filename, region, mode = get_args_Classification()
    
    main(filename, region, mode)