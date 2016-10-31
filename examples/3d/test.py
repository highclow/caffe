import numpy as np
from scipy import ndimage

caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

caffe.set_mode_gpu()
net = caffe.Net('train_test.prototxt', caffe.TRAIN)
print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))

net.forward()

data = net.blobs['data'].data
weights = net.params['conv1'][0].data
bias = net.params['conv1'][1].data
res = net.blobs['conv1'].data

test = np.zeros([data.shape[0], weights.shape[0], data.shape[2],
     data.shape[3], data.shape[4]])
for i in xrange(data.shape[0]):
  for j in xrange(weights.shape[0]):
    for k in xrange(data.shape[1]):
      test[i,j,] += ndimage.correlate(data[i,k], weights[j,k], mode='constant', cval=0.0)

for j in xrange(weights.shape[0]):
  test[:,j] += bias[j]

test[test<0] = 0


print abs(res - test).sum()
