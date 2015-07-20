"""Samples RNAstructure conditioned on label."""
from deepnet import inference
from deepnet import trainer as tr
import sys


def SampleText(model_file, op_file, base_output_dir, data_proto, gpu_mem, main_mem):
  datasets = ['train']
  layernames = ['RNA1seq_input_layer', 'RNA2seq_input_layer', 'RNA3seq_input_layer']
  layernames_to_unclamp = ['RNA1seq_input_layer', 'RNA2seq_input_layer', 'RNA3seq_input_layer','joint_hidden1']
  method = 'gibbs'  # 'gibbs'
  steps = 1000

  inference.DoInference(model_file, op_file, base_output_dir, layernames,
                        layernames_to_unclamp, memory='10G', method=method,
                        steps=steps, datasets=datasets, gpu_mem=gpu_mem,
                        main_mem=main_mem, data_proto=data_proto)

def main():
  model_file = sys.argv[1]
  op_file = sys.argv[2]
  output_dir = sys.argv[3]
  data_proto = sys.argv[4]
  if len(sys.argv) > 5:
    gpu_mem = sys.argv[5]
  else:
    gpu_mem = '2G'
  if len(sys.argv) > 6:
    main_mem = sys.argv[6]
  else:
    main_mem = '30G'
  board = tr.LockGPU()
  SampleText(model_file, op_file, output_dir, data_proto, gpu_mem, main_mem)
  tr.FreeGPU(board)


if __name__ == '__main__':
  main()
