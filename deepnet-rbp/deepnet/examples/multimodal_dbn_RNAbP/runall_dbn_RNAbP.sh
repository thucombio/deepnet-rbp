#!/bin/bash
# This script trains a Multimodal DBN using deepnet.
# Before running this script download and extract data from
# http://www.bioinf.uni-freiburg.de/Software/GraphProt/

# Location of deepnet. EDIT this for your setup.
deepnet=/home/bioinformatics/deepnet-master/deepnet

numsplits=10

for i in `seq ${numsplits}`
do
echo "Iteration"${i}

# Location of the downloaded and preprocessed data. This is also the place where learned models
# and representations extracted from them will be written. Should have lots of
# space ~30G. EDIT this for your setup.
i=$(($i-1))
prefix=/home/bioinformatics/Documents/data/RNAbP/cv${i}

# Amount of gpu memory to be used for buffering data. Adjust this for your GPU.
# For a GPU with 6GB memory, this should be around 4GB.
# If you get 'out of memory' errors, try decreasing this.
gpu_mem=5G

# Amount of main memory to be used for buffering data. Adjust this according to
# your RAM. Having at least 16G is ideal.
main_mem=20G

trainer=${deepnet}/trainer.py
extract_rep=${deepnet}/extract_rbm_representation.py
model_output_dir=${prefix}/dbn_models
data_output_dir=${prefix}/dbn_reps
clobber=false

mkdir -p ${model_output_dir}
mkdir -p ${data_output_dir}

# Set up paths.
echo Setting up data
python setup_data.py ${prefix} ${model_output_dir} ${data_output_dir} \
  ${gpu_mem} ${main_mem} ${numsplits} || exit 1

# RNA1SEQ LAYER - 1.
#(
if ${clobber} || [ ! -e ${model_output_dir}/RNA1seq_rbm1_LAST ]; then
  echo "Training first layer RNA1seq RBM."
  python ${trainer} models/RNA1seq_rbm1.pbtxt \
    trainers/dbn/train_CD_RNA1seq_layer1.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/RNA1seq_rbm1_LAST \
    trainers/dbn/train_CD_RNA1seq_layer1.pbtxt RNA1seq_hidden1 \
    ${data_output_dir}/RNA1seq_rbm1_LAST ${gpu_mem} ${main_mem} || exit 1
fi

# RNA2SEQ LAYER - 1.
if ${clobber} || [ ! -e ${model_output_dir}/RNA2seq_rbm1_LAST ]; then
  echo "Training first layer RNA2seq RBM."
  python ${trainer} models/RNA2seq_rbm1.pbtxt \
    trainers/dbn/train_CD_RNA2seq_layer1.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/RNA2seq_rbm1_LAST \
    trainers/dbn/train_CD_RNA2seq_layer1.pbtxt RNA2seq_hidden1 \
    ${data_output_dir}/RNA2seq_rbm1_LAST ${gpu_mem} ${main_mem} || exit 1
fi

# MERGE 1D AND 2D DATA PBTXT FOR TRAINING JOINT RBM
if ${clobber} || [ ! -e ${data_output_dir}/joint_rbm_LAST/input_data.pbtxt ]; then
  mkdir -p ${data_output_dir}/joint_rbm_LAST
  python merge_dataset_pb.py \
    ${data_output_dir}/RNA1seq_rbm1_LAST/data.pbtxt \
    ${data_output_dir}/RNA2seq_rbm1_LAST/data.pbtxt \
    ${prefix}/RNAseq.pbtxt \
    ${data_output_dir}/joint_rbm_LAST/input_data.pbtxt \
	${prefix} || exit 1
fi

# TRAIN JOINT RBM
if ${clobber} || [ ! -e ${model_output_dir}/joint_rbm_LAST ]; then
  echo "Training joint layer RBM."
  python ${trainer} models/joint_rbm.pbtxt \
    trainers/dbn/train_CD_joint_layer.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/joint_rbm_LAST \
    trainers/dbn/train_CD_joint_layer.pbtxt joint_hidden1 \
    ${data_output_dir}/joint_rbm_LAST ${gpu_mem} ${main_mem} || exit 1
fi

# TRAIN JOINT RBM2
if ${clobber} || [ ! -e ${model_output_dir}/joint_rbm2_LAST ]; then
  echo "Training joint layer RBM2."
  python ${trainer} models/joint_rbm2.pbtxt \
    trainers/dbn/train_CD_joint_layer2.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/joint_rbm2_LAST \
    trainers/dbn/train_CD_joint_layer2.pbtxt joint_hidden2 \
    ${data_output_dir}/joint_rbm2_LAST ${gpu_mem} ${main_mem} || exit 1
fi

# ERROR BACK-PROPAGATION FOR THE TOP-LAYER
for layer in joint_hidden2
  do
    if ${clobber} || [ ! -e ${model_output_dir}/classifiers/split_1/joint_hidden2_classifier_BEST ]; then
      echo ${layer}
      python ${trainer1} models/classifiers/${layer}_classifier.pbtxt \
        trainers/classifiers/split_1.pbtxt eval.pbtxt || exit 1
    fi
done

# ERROR BACK-PROPAGATION FOR THE WHOLE NETWORK
if ${clobber} || [ ! -e ${model_output_dir}/classifiers/split_1/ff_classifier_BEST ]; then
echo Train FF
python ${trainer1} models/classifiers/model_dropout${i}.pbtxt \
  trainers/classifiers/split.pbtxt eval.pbtxt || exit 1
fi

done


