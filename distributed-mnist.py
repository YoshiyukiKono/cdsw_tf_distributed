# Altered from http://ischlag.github.io/2016/06/12/async-distributed-tensorflow/

'''
Distributed Tensorflow 1.2.0 example of using data parallelism and share model parameters.
Trains a simple sigmoid neural network on mnist for 20 epochs on three machines using one parameter server. 

More details here: ischlag.github.io
'''

from __future__ import print_function

import tensorflow as tf
import sys
import time
import cdsw_distributed_tensorflow as cdt

n_workers = 3
n_ps = 1

cluster_spec, session_addr = cdt.run_cluster(n_workers=n_workers, \
   n_ps=n_ps, \
   cpu=1, \
   memory=2, \
   worker_script="distributed_mnist_worker_script")

cdt.tensorboard('./distributed-mnist')
