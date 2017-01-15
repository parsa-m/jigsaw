import numpy as np 
import pdb
from caffe import layers as L, params as P, to_proto, NetSpec, get_solver, Net
import caffe 
from caffe.proto import caffe_pb2
import os 

# This function is specifically for ImageNet, You will need to modify it for loading your own dataset 
# Make sure this function returns a list or array that can be indexed into for retrieving the name and sub path of each file in train_data_dir 

def load_data(data_path):
  
  imgpaths=[]
  
  with open(data_path[:-6] + 'train.txt', 'rb') as f:
    for line in f:
      row=line.split()
      imgpaths.append(row[0])

  dataset_size = len(imgpaths)

  return imgpaths, dataset_size

# tweak solver parameters here 

def make_solver(outdir, gpu_num): 

  caffe.set_device(gpu_num)
  caffe.set_mode_gpu()

  s = caffe_pb2.SolverParameter()
  s.net =  outdir + "/network.prototxt"
  s.base_lr = 0.01
  s.lr_policy = "step" 
  s.gamma = 0.1 
  s.stepsize = 100000 
  s.display = 5000 
  s.snapshot = 10000
  s.momentum = 0.9
  s.weight_decay = 0.0005
  
  s.snapshot_prefix = outdir + "/model"
  s.solver_mode= caffe_pb2.SolverParameter.GPU

  with open(outdir+'solver.prototxt','w') as f:
      f.write(str(s))

  solver = caffe.get_solver(f.name)

  return solver 

def setup_dirs(exp_name, tmpdir, outdir, out_dir): 

  if not os.path.exists(tmpdir):
    os.makedirs(tmpdir)

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  if not os.path.exists(outdir):
    os.mkdir(outdir)

  if os.path.isfile(exp_name + '_quit'):
    os.remove(exp_name + '_quit')
