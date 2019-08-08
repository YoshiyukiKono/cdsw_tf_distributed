# It looks like tensorflow requires you to decide which portions
# of the graph live on which servers.

import tensorflow as tf
import cdsw_distributed_tensorflow as cdt

cluster_spec, session_addr = cdt.run_cluster(n_workers=2, \
   n_ps=0, \
   cpu=1, \
   memory=2)

cluster = tf.train.ClusterSpec(cluster_spec)

x = tf.constant(2)

with tf.device("/job:worker/task:1"):
  y2 = x - 66

with tf.device("/job:worker/task:0"):
  y1 = x + 300
  y = y1 + y2

with tf.Session(session_addr) as sess:
  result = sess.run(y)
  writer = tf.summary.FileWriter('./distributed-basic', sess.graph)
  print(result)
writer.close()

cdt.tensorboard('./distributed-basic')
