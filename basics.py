import utils
import tensorflow as tf

# Tensorflow should be a language. You construct computation graphs that are a lot
# like the value state dependence graph, one of the haskell compiler's intermediate
# representations, then you ask it to evaluate them.

# Get a lazy value:
a = tf.add(2,3)

# Evaluate:
sess = tf.Session()
print sess.run(a)
sess.close()

# You can also sess.run a list of lazy nodes.

# All nodes you declare mash into one ambient graph. The default context holds a
# graph and compute resources. You can switch out the compute resources with tf.device.
# You can switch out the default graph with tf.Graph.as_default.

# High level NN libraries for tensorflow include tf-learn, keras, 
# tensorflow.contrib.learn

# Declare constants explicitly:

a = tf.constant([2, 2], name="a")
b = tf.constant([[0, 1], [2, 3]], name="b")
x = tf.add(a, b, name="add")
y = tf.multiply(a, b, name="mul")
with tf.Session() as sess:
  x_eval, y_eval = sess.run([x, y])
  print x_eval, y_eval 

# Don't use huge constants, they're inlined into the graph def.
  
# Tensorflow also has tf.zeroes, tf.ones, tf.linspace etc. which are like the numpy 
# versions.
# It also has numpy like rngs.
# It has elementwise and matrix ops: tf.mul, tf.matmul etc.
# It has a dtype system like numpy's.
# tf.Variable is a mutable var:

a = tf.Variable(2, name="scalar")

# You have to initialize Variables, and you can eval them:

init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  print a.eval()
  
# You can also call variable.assign; but it doesn't just assign in place, it creates
# a mutation in the dataflow graph. Probably used in reverse mode AD / backpropagation.

assgn_op = a.assign(5)
with tf.Session() as sess:
  sess.run(init)
  print a.eval()
  sess.run(assgn_op)
  print a.eval()

# The variable's value lives in the session, unlike with constants.

# You can use placeholders to create variables that won't be initialized until 'runtime':

a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)
c = a + b # Short for tf.add(a, b)
with tf.Session() as sess:
  print sess.run(c, {a: [1, 2, 3]}) # the tensor a is the key, not the string ‘a’

# Avoid using shape=None in placeholders, it's bad for static shape checking and shape
# inference.

# Avoid creating nodes in a loop even you feel it improves code aesthetics
# Unless you have to

# You can do multiple sess.run calls on the same session. That's how you train things.
# tf.train contains functions that take a loss node, then try to step all the Variables
# in the graph to minimize it.

# Some guy thinks tf.train.AdamOptimizer is the best overall choice.
