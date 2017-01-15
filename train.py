from multiprocessing import Process, Queue, Pool
import numpy as np
import os, subprocess, time, shutil, signal, glob, re
from caffe import layers as L, params as P, to_proto, NetSpec, get_solver, Net
from caffe.proto import caffe_pb2
from caffe.proto import caffe_pb2
import caffe
import pdb
from PIL import Image
from PIL import *
import scipy.io
import utils 
np.random.seed(1337)


########### Change Paths here ################
out_dir = '/x/parsa/output_folder_jigsaw/'
train_data_dir = '/z/ilsvrc2012/train/'
tmp_dir = '/x/parsa/temp_jigsaw_storage/'
exp_name = 'exp0'
########### Change Paths here ################

########## Tuning Parameters #################
batch_size = 256
gpu_num = 0 
dataset_mean = [122.67892, 116.66877, 104.00699] # ImageNet mean 
resize_square_dim = 256
total_crop_dim = 225
patch_dim = 64 # the patch dimension 
fc6_nout = 1024 # 1024 for jps and 512 for rec 
num_threads = 6  # number of processes 
size_q = 6 # size of queue for each process
########## Tuning Parameters #################



num_classes = 100
cell_dim = total_crop_dim / 3 #75 the dimension of slicing by three 
gap = cell_dim - patch_dim

if cell_dim*3 != total_crop_dim: raise NameError("total_crop_dim isn't divisible by three")

# these are the same permutations that were used in jigsaw paper
permutations_mat_dict = scipy.io.loadmat('100_permutations.mat')
permutation_array = permutations_mat_dict['pset']
permutation_array = permutation_array - 1 # subtracting by 1 to get 0-8 instead of 1-9

imgs, dataset_size = utils.load_data(train_data_dir)
all_idxs = range(dataset_size)
all_labels = np.random.randint(100, size=dataset_size)

def get_resized_image(idx, dataset):

  im =Image.open(train_data_dir + dataset[idx])
  im_np = np.array(im, dtype=np.uint8).copy()

  # resizing to resize_square_dim x resize_square_dim by reducing or increasing smallest dimension to resize_square_dim while respecting aspect ratio
  # and then cropping other dimension to resize_square_dim. 

  maxdim = resize_square_dim
  immin = min(im_np.shape[0:2])
  ratio = float(maxdim)/float(immin)

  if im_np.shape[0] < im_np.shape[1]: 
    im = im.resize((int(im_np.shape[1]*ratio),resize_square_dim)) 
  elif im_np.shape[1] < im_np.shape[0]: 
    im = im.resize((resize_square_dim,int(im_np.shape[0]*ratio))) 
  else: 
    im = im.resize((resize_square_dim,resize_square_dim))

  im = np.array(im, dtype=np.uint8)

  try:
    if(len(im.shape)==2):
      # print("found grayscale image")
      im = np.array([im,im,im]).transpose(1,2,0) 

    elif(im.shape[2]==4):
      print("found 4-channel png with channel 4 min "+str(np.min(im[:,:,3]))) 
      im = im[:,:,0:3] 

  except:
    print("image id: " + str(idx))
    raise

  xr = np.random.randint(0, im.shape[0] - total_crop_dim + 1)
  yr = np.random.randint(0, im.shape[1] - total_crop_dim + 1)
  im = im[xr:xr+total_crop_dim,yr:yr+total_crop_dim]

  return im.copy()


def conv_relu(bottom, nout, ks, weight_filler=dict(type='gaussian', std=0.01) , bias_filler=dict(type='constant', value=0), stride=1, pad=0, group=1, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]):

    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad, group=group, weight_filler=weight_filler, bias_filler=bias_filler, param=param)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout,  weight_filler=dict(type='gaussian', std=0.005) , bias_filler=dict(type='constant', value=1), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]):

    fc = L.InnerProduct(bottom, num_output=nout, weight_filler=weight_filler, bias_filler=bias_filler, param=param )
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks=1, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def alexnet_stack(data, n, fc6_nout):

    n.conv1, n.relu1 = conv_relu(data, 96, ks=11, stride=2, pad=0) 
    n.pool1 = max_pool(n.relu1, ks=3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 256, ks=5, pad=2, group=2, bias_filler=dict(type='constant', value=1)) 
    n.pool2 = max_pool(n.relu2, ks=3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2,  384, ks=3,  pad=1)
    n.conv4, n.relu4 = conv_relu(n.relu3, 384, ks=3, pad=1, group=2, bias_filler=dict(type='constant', value=1)) 
    n.conv5, n.relu5 = conv_relu(n.relu4, 256, ks=3, pad=1, group=2, bias_filler=dict(type='constant', value=1))
    n.pool5 = max_pool(n.relu5, ks=3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, fc6_nout) 
    n.drop6 = L.Dropout(n.fc6, dropout_ratio=0.5, in_place=True) 

def gen_net(batch_size=256):

    n=NetSpec()

    n.data = L.Input(shape=[dict(dim=[batch_size*9, 3, patch_dim, patch_dim])])
    n.label = L.Input(shape=[dict(dim=[batch_size])])
    alexnet_stack(n.data, n, fc6_nout)

    n.reshape_batch = L.Reshape(n.relu6, shape={'dim': [-1, 9*fc6_nout]}) 

    n.fc7, n.relu7 = fc_relu(n.reshape_batch, 4096, param=[dict(lr_mult=0.1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.drop7 = L.Dropout(n.fc7, dropout_ratio=0.5, in_place=True)

    n.fc8 = L.InnerProduct(n.relu7, num_output=100, weight_filler=dict(type='gaussian', std=0.01) , bias_filler=dict(type='constant', value=0),
      param=[dict(lr_mult=0.1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]) 
    n.loss = L.SoftmaxWithLoss(n.fc8, n.label) 

    prot=n.to_proto()
    return prot


def imgloader(dataq, batch_size, imgs,  tmpdir, seed, tid): 

  np.random.seed(seed)
  batch_num = 0

  while True: 

    image_indices = np.random.choice(all_idxs, size=batch_size, replace=False)

    label_set = []
    data_set = []

    for image_ind in image_indices:

      # label = np.random.randint(0, num_classes)
      label = all_labels[image_ind]
      permutation = permutation_array[label] 
      patches = []    

      im = get_resized_image(image_ind, imgs)     

      im = im.transpose(2, 0, 1) 
      im = im.astype(np.float32)

      # Doing mean subtraction and cropping of the 9 patches 
      
      im[0,:,:] = im[0,:,:] - dataset_mean[0]
      im[1,:,:] = im[1,:,:] - dataset_mean[1]
      im[2,:,:] = im[2,:,:] - dataset_mean[2]

      patches.append(im[:,:cell_dim,:cell_dim].copy()) #1
      patches.append(im[:,:cell_dim,cell_dim:cell_dim*2].copy()) #2
      patches.append(im[:,:cell_dim,cell_dim*2:cell_dim*3].copy()) #3

      patches.append(im[:,cell_dim:cell_dim*2,:cell_dim].copy()) #4
      patches.append(im[:,cell_dim:cell_dim*2,cell_dim:cell_dim*2].copy()) #5
      patches.append(im[:,cell_dim:cell_dim*2,cell_dim*2:cell_dim*3].copy()) #6

      patches.append(im[:,cell_dim*2:cell_dim*3,:cell_dim].copy()) #7
      patches.append(im[:,cell_dim*2:cell_dim*3,cell_dim:cell_dim*2].copy()) #8
      patches.append(im[:,cell_dim*2:cell_dim*3,cell_dim*2:cell_dim*3].copy()) #9

      # patches are cell_dim x cell_dim (3*cell_dim = total_crop_dim) , 
      # we then crop a patch_dim x patch_dim to introduce a gap and randomly pick location of left top corner

      for i in range(0,9): 
        sx = np.random.randint(0, gap+1)
        sy = np.random.randint(0, gap+1)
        patches[i] = patches[i][:,sx:sx+patch_dim,sy:sy+patch_dim].copy()


      for i in range(0,9): 
        data_set.append(patches[permutation[i]].copy())

      label_set.append(label)
   

    data_set = np.array(data_set)
    label_set = (np.array(label_set))

    npy_save_path = tmpdir + str(tid) + '_' + str(batch_num) + '.npy'

    try: 
      np.save(npy_save_path, data_set) # saving the batch to disk 
    except: 
      print "used too much memory"
      print npy_save_path
      ### TODO(Parsa): maybe insert time.sleep(2) here
      continue

    # putting batch path and label into process queue
    dataq.put((npy_save_path, label_set), timeout=600)
    batch_num += 1


outdir = out_dir + exp_name + '/'
tmpdir = tmp_dir + '/' + exp_name + '_out/'
utils.setup_dirs(exp_name, tmpdir, outdir, out_dir)

with open(outdir+'network.prototxt','w') as f:
  f.write(str(gen_net()))

solver = utils.make_solver(outdir, gpu_num)

# start the data prefetching threads.
dataq = []
procs = []

for i in range(num_threads):
  dataq.append(Queue(size_q))
  procs.append(Process(target=imgloader, args=(dataq[-1], batch_size, imgs, tmpdir, (hash(outdir)+i) % 1000000, i)))
  procs[-1].start() 

def signal_handler(signal, frame):
    print("Quit by creating file %s_quit in directory") % exp_name 

def deepcontext_quit():
  for proc in procs:
    proc.terminate()
  time.sleep(2)
  shutil.rmtree(tmpdir)
  os.kill(os.getpid(), signal.SIGKILL) 
signal.signal(signal.SIGINT, signal_handler) 

################################## Main Loop #####################################

curstep = 0 

while True:

  start=time.time()

  (npy_load_path, label_batch) = dataq[curstep % len(dataq)].get(timeout=600) # grab path to batch and label from queue
  data_batch = np.load(npy_load_path, mmap_mode='r') # read batch from disk 

  os.remove(npy_load_path)

  solver.net.blobs['data'].data[:] = data_batch[:]
  solver.net.blobs['label'].data[:] = label_batch[:]

  if curstep % 10 == 0: 
    print("queue size: " + str(dataq[curstep % len(dataq)].qsize()))
    print("data input time: " + str(time.time()-start))


  start=time.time()

  solver.step(1)
  curstep += 1

  if curstep % 10 == 0: 
    print("solver step time: " + str(time.time() - start))
    print("")
    print("step number: %s" % curstep)
    print("loss: " + str(solver.net.blobs['loss'].data))
    print float((solver.net.blobs['fc7'].data[:] !=  0).sum())/ float((solver.net.blobs['fc7'].data[:] !=  -1).sum())
    print("")


  dobreak = False

  if dobreak:
    break
  if os.path.isfile(exp_name + '_quit'):
    deepcontext_quit()
