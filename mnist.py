import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials import mnist

#MNIST = mnist.input_data.read_data_sets("./tf-stanford-tutorials/data/mnist", one_hot=True)
# This isn't in the right format for this tutorial.
#import pickle
#MNIST = pickle.load(open("./tf-stanford-tutorials/data/mnist/mnist.pkl"))

# Step 2: Define parameters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 25
# Step 3: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0 - 9.
# each label is one hot vector.
X = tf.placeholder(tf.float32, [batch_size, 784])
Y = tf.placeholder(tf.float32, [batch_size, 10])
# Step 4: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weights")

b = tf.Variable(tf.zeros([1, 10]), name="bias")
# Step 5: predict Y from X and w, b
# the model that returns probability distribution of possible label of the image
# through the softmax layer
# a batch_size x 10 tensor that represents the possibility of the digits
logits = tf.matmul(X, w) + b
# Step 6: define loss function
# use softmax cross entropy with logits as the loss function
# compute mean cross entropy, softmax is applied internally
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy) # computes the mean over examples in the batch
# Step 7: define training op
# using gradient descent with learning rate of 0.01 to minimize cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  n_batches = int(MNIST.train.num_examples/batch_size)
  for i in range(n_epochs): # train the model n_epochs times
    for _ in range(n_batches):
      X_batch, Y_batch = MNIST.train.next_batch(batch_size)
      sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch})
      # average loss should be around 0.35 after 25 epochs

  # test the model
  n_batches = int(MNIST.test.num_examples / batch_size)
  total_correct_preds = 0
  for i in range(n_batches):
    X_batch, Y_batch = MNIST.test.next_batch(batch_size)
    _, loss_batch, logits_batch = sess.run([optimizer, loss, logits],
    feed_dict={X: X_batch, Y:Y_batch})
    preds = tf.nn.softmax(logits_batch)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # similar
    total_correct_preds += sess.run(accuracy)
  print "Accuracy {0}".format(total_correct_preds/MNIST.test.num_examples)
