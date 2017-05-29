import tensorflow as tf 
import numpy as np
import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data

# A variable has to be always initialized whereas placeholder doesn't need to be initialized at
# the time of its declaration.
def weight_variable(shape):
	random = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(random)

def bias_variable(shape, value=0.1):
	# constant = tf.truncated_normal(shape, stddev = 0.1)
	constant = tf.constant(value, shape=shape)
	return tf.Variable(constant)

def conv2D(X, weights):
	return tf.nn.conv2d(X, weights, strides=[1,1,1,1], padding='SAME') # padding = same; means the 
	# input and output image will be of the same width and height.

def max_pooling(X):
	return tf.nn.max_pool(X, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def main(_):
	# Load dataset
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	# define the training set feature and target variables
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])

	# first convolution layer
	w_conv1 = weight_variable([5, 5, 1, 32])   # [width, height, channels, n_filter]
	b_conv1 = bias_variable([32])

	x_image = tf.reshape(x, [-1, 28, 28, 1])

	out_conv1 = tf.nn.relu(conv2D(x_image, w_conv1) + b_conv1)
	out_pool1 = max_pooling(out_conv1)

	# second convolution layer
	w_conv2 = weight_variable([5,5,32,64])
	b_conv2 = bias_variable([64])

	out_conv2 = tf.nn.relu(conv2D(out_pool1, w_conv2) + b_conv2)
	out_pool2 = max_pooling(out_conv2)

	# fully connected layer
	w_dense1 = weight_variable([7*7*64, 1024])    # 1024 neurons in the dense layer
	b_dense1 = bias_variable([1024])

	out_pool2_flat = tf.reshape(out_pool2, [-1, 7*7*64])
	out_dense1 = tf.nn.relu(tf.add(tf.matmul(out_pool2_flat, w_dense1), b_dense1))

	# drop out layer- is a form of regularization technique used for dealing with overfitting
	# problem.
	keep_prob = tf.placeholder(tf.float32)
	out_droplayer = tf.nn.dropout(out_dense1, keep_prob)

	# final dense(fully connected) layer for predicting the output
	w_out = weight_variable([1024, 10])
	b_out = bias_variable([10])

	y_cnn = tf.matmul(out_droplayer, w_out) + b_out

	# cost function
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_cnn))
	train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	# compare the logits(predicted values) with the labels provided and come up with a vector
	# of boolean values.
	correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_cnn, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	for i in range(20000):
		batch = mnist.train.next_batch(50)
		if i%100 == 0:
			# keep_prob - the probability for a particular activation unit in hidden layer
			# to keep or drop
			train_accuracy = accuracy.eval(feed_dict = {x:batch[0], y_:batch[1], keep_prob:0.5})
			print ('step %i, Training Accuracy %g'%(i, train_accuracy))
		train.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

	print ("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels,
	keep_prob:0.5}))

if __name__ == '__main__':
  	parser = argparse.ArgumentParser()
  	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)