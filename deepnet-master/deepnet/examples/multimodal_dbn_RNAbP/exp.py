#import perf
import os

oStdout=os.popen('perf -roc -file /home/bioinformatics/deepnet-master/deepnet/examples/multimodal_dbn/results/joint_hidden_classifier_split_1.txt')
strStdout=oStdout.readline()
strOut=strStdout.strip()
print(strOut)
