from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

import tensorflow as tf 
import numpy as np 
import data_helpers
import time
import argparse
import sys

batch_size = 100
max_steps = 5000

def convolve(X, weights):
	return tf.nn.conv2d(X, weights, strides = [1,1,1,1], padding = 'SAME')

def max_pool(X):
	return tf.nn.max_pool(X, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.01)
	return tf.Variable(initial)

def bias_variable(shape, value=0.1):
	initial = tf.constant(value, shape = shape)
	return tf.Variable(initial)

def main(_):
	dataset = data_helpers.load_data()
	x_train = dataset['images_train']
	y_train = dataset['labels_train']
	x_test = dataset['images_test']
	y_test = dataset['labels_test']

	# convolution layer 1
	x_input1 = tf.placeholder(tf.float32, shape=[None, 3072])
	y_output = tf.placeholder(tf.int64, shape=[None])

	# 32x32x3(rgb) images with the fourth dim. denoting the batch size
	x_input = tf.reshape(x_input1, [-1,32,32,3])

	# kernel size of 5x5x3 with 32 different filters
	w_conv1 = weight_variable([5,5,3,32])
	b_conv1 = bias_variable([32])

	out_conv1 = tf.nn.relu(convolve(x_input, w_conv1) + b_conv1)  # [32, 32, 32, batch_size]

	# pooling layer 1 - size of output 
	out_pool1 = max_pool(out_conv1)    # [16, 16, 32, batch_size]

	# convolution layer 2
	w_conv2 = weight_variable([5,5,32,64])
	b_conv2 = bias_variable([64])

	out_conv2 = tf.nn.relu(convolve(out_pool1, w_conv2) + b_conv2)  # [16, 16, 64, batch_size]

	# pooling layer 2
	out_pool2 = max_pool(out_conv2)   # [8, 8, 64, batch_size]

	# dense layer
	out_pool2_flat = tf.reshape(out_pool2, [-1, 8*8*64])   # [batch_size, 4096]

	w_dense1 = weight_variable([8*8*64, 1024])    
	b_dense1 = bias_variable([1024])
	out_dense1 = tf.nn.relu(tf.matmul(out_pool2_flat, w_dense1) + b_dense1)  # [batch_size, 1024]

	keep_prob = tf.placeholder(tf.float32)
	out_drop = tf.nn.dropout(out_dense1, keep_prob)

	# final dense(output) layer
	w_dense2 = weight_variable([1024, 10])
	b_dense2 = bias_variable([10])

	out_dense2 = tf.matmul(out_drop, w_dense2) + b_dense2   # [batch_size, 10]

	# calculate loss function. Using sparse helps to make the labels as a number(0-9) and 
	# logits as vector
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_dense2,
																		labels=y_output))
	train = tf.train.AdamOptimizer(1e-4).minimize(loss)

	correct_prediction = tf.equal(tf.argmax(out_dense2,1), y_output)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for i in range(max_steps):
			indices = np.random.choice(dataset['images_train'].shape[0], batch_size)
			images_batch = dataset['images_train'][indices]
			labels_batch = dataset['labels_train'][indices] 

			if i%100 == 0:
				train_accuracy = sess.run(accuracy, feed_dict={
					x_input1:images_batch,
					y_output:labels_batch,
					keep_prob:0.5
					})
				print ('Step:%i , Accuracy:%g'%(i, train_accuracy))
			sess.run(train, feed_dict={x_input1:images_batch, y_output:labels_batch, keep_prob:0.5})

		test_accuracy = sess.run(accuracy, feed_dict={
			x_input1: x_test,
			y_output: y_test,
			keep_prob:0.5
			})
		print ('Test Accuracy : %g'%test_accuracy)
	end_time = time.time()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
  	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  	FLAGS, unparsed = parser.parse_known_args()
  	start_time = time.time()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
	print ('Time Taken :%g'%(end_time - start_time))