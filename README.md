   [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-ff69b4.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

# Chroma-based Supervised Mode Recognition in Multi-cultural Context

# 
    
 This repository is a demonstration for a audio classification methodology based on the tonal concept of 'harmonic modes' in Musicology. The concept of 'harmonic modes' exists in various tonal music traditions around the world with distinct alterations. In this project, we apply chroma based supervised mode recognition in the multi-cultural context. The evaluation of algorithms are designed to be performed on the Ottoman-Turkish, Hindustani and Carnatic datasets that were built within the scope of CompMusic Project.
 
  -  Please cite the reference below if you use the content of this repository in your work.
  
  >  [1] Demirel, E., Bozkurt, B., Serra, X. (2018). Automatic Makam Recognition using Chroma Features, in proceedings of  Folk Music Analysis 2018 (FMA)
 https://zenodo.org/record/1239435#.WulmO3VMSfg
    
   
  Installation
  ---------
  In order to use the tools and notebooks, you need to install 'docker' . Docker provides a virtual environment with all the desired dependencies and libraries already installed. In this toolbox for 'Chroma-based Supervised Mode Recognition in Multi-cultural Context', we have used the tools in 'MIR-Toolbox' which contains a set of tools (including *numpy,matplotlib, scikit-learn,'Essentia'* ) installed and compiled for several Music Information Retrieval applications. For more information regarding toolbox, please refer to https://github.com/MTG/MIR-toolbox-docker  :
  
   1) Install docker-compose
   Follow [instructions](https://docs.docker.com/compose/install/).

   **Windows**
    https://docs.docker.com/docker-for-windows/install/

   **Mac**
    https://docs.docker.com/docker-for-mac/install/

   **Ubuntu**
    https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-docker-ce


   2) Clone the repository into target directory.
   
    git clone https://github.com/emirdemirel/Supervised_Mode_Recognition.git
    
   3) Initiate the docker image using following command. You may need to access with superuser permission.
   
     docker-compose up
     
   Then access localhost:8888 on your browser and when asked for a password use 'mir'.
     
   4) Open the Jupyter notebook  'Chroma-based_ModeRecognition_in_Multi-culturalContext.ipynb' for the implementation of the work presented in paper [1].

   
   Authors
   -------------
   Emir Demirel
   emir.demirel@upf.edu
