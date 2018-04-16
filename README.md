# Chroma-based Supervised Mode Recognition in Multi-cultural Context

# 
    
#### This repository is a demonstration for classifying audio files based on the tonal concept of 'harmonic modes' in Musicology. The concept of 'harmonic modes' exists in various tonal music traditions around the world with distinct alterations. In this project, we apply chroma based supervised mode recognition in the multi-cultural context. The evaluation of algorithms are designed to be performed on the Ottoman-Turkish, Hindustani and Carnatic datasets that were built within the scope of CompMusic Project
    
    There are 2 main steps for the audio classification: Feature Extraction and Classification. 
   
  Installation
  ---------
  In order to use the tools and notebooks, you need to install 'docker' . Docker provides a virtual environment with all the desired dependencies and libraries already installed. In this toolbox for 'Chroma-based Supervised Mode Recognition in Multi-cultural Context', we have used the tools in 'MIR-Toolbox' which contains a set of tools installed and compiled, including 'Essentia' for several Music Information Retrieval applications. For more information regarding toolbox, please refer to https://github.com/MTG/MIR-toolbox-docker  :
  
   1) Install docker-compose
   Follow [instructions](https://docs.docker.com/compose/install/).

   #### Windows
    https://docs.docker.com/docker-for-windows/install/

   #### Mac
    https://docs.docker.com/docker-for-mac/install/

   #### Ubuntu
    https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-docker-ce


   2) Clone the repository into target directory.
   
    git clone https://github.com/emirdemirel/Supervised_Mode_Recognition.git
    
   3) Initiate the docker image using following command.
   
     docker-compose up
     
     
    

   
   Authors
   -------------
   Emir Demirel
   emir.demirel@upf.edu
    
 
   
