############################################################
MULTIMODAL DEEP BELIEF NETS FOR PREDICTING RBP BINDING SITES
############################################################
This code trains a Multimodal DBN on the RNAseq dataset.
The implementation uses GPUs to accelerate training.

(1) GET DATA
  - Download and preprocessed the RNAseq data into a place with lots of disk space.
  —- Use RNAshapes and JAR3D to predict RNA 2D and 3D structure.
  —- Construct RNA 1D, 2D and 3D structural profiles for further training and testing. 

(2) TRAIN MULTIMODAL DBN
  - Change to the directory containing this file.
  - Edit paths in runall_dbn_RNAbP.sh
  - Edit model and training pbtxt files to be consistent with RNA structural information.
  - Train dbn
  $ ./runall_dbn_RNAbP.sh

If you have any trouble at installation or running the code, please feel free to contact the author:
zhangsai13@mails.tsinghua.edu.cn
