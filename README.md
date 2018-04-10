# Supervised_Modality_Classification

# AUDIO CLASSIFICATION BASED ON 'MODALITY' USING CHROMA FEATURES",
    
#### This series of notebooks demonstrate a tutorial for classifying audio files based on the tonal concept of 'modality in music'. The concept of 'modality' exists in various tonal music traditions around the world. In this tutorial, we apply supervised learning based on chroma features.",
    
    There are 2 main steps for the audio classification: Feature Extraction and Classification. ,
   
   #### The first notebook shows the steps of Chroma Feature Extraction from an audio collection.
   #### The second notebook includes the manual Feature Selection step and the whole Pipeline of the Machine Learning process for Audio Classification based on 'Modality'.
    
   #### PARAMETERS:
   - numBins(int) : Number of bins in the Chroma Vectors. This is parametrized in consideration of possible microtonalities existing in non-Western music traditions (12,24,36,48, ...)\n",
   - modality(str) : Name of the modality type specific for the music tradition.\n",
         Classical Western Music / Jazz : Mode, chord-scale",
         Classical Turkish Music : Makam",
         Classical North Indian Music : Rag, Raaga",
         Classical Arab-Andalusian Music : Tab",
