import tensorflow as tf
import numpy as np
import argparse
import time
import sys
import csv

n_hidden = 10
max_steps = 20000
batch_size = 100

def weight_varibles(shape):
	initial = tf.truncated_normal(shape= shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variables(shape, value=0.1):
	initial = tf.constant(value, shape=shape)
	return tf.Variable(initial)

def main(_):
	train_x = []
	train_y = []
	test_x = []
	test_y = []

	with open('iris_training.csv') as csvfile:
		readcsv = csv.reader(csvfile, delimiter=',')
		for row in readcsv:
			x = np.array(row[0:4], np.float32)
			train_x.append(x)
			y = np.array(row[4], np.int64)
			train_y.append(y)
		train_x = np.array(train_x)
		train_y = np.array(train_y)

	with open('iris_test.csv') as csvfile:
		readcsv = csv.reader(csvfile, delimiter=',')
		for row in readcsv:
			x = np.array(row[0:4], dtype=np.float32)
			test_x.append(x)
			y = np.array(row[4], dtype=np.int64)
			test_y.append(y)
		test_x = np.array(test_x)
		test_y = np.array(test_y)

	x_input = tf.placeholder(tf.float32, shape=[None, 4])
	y_label = tf.placeholder(tf.int64, shape=[None])

	w_input = weight_varibles([4, 10])
	b_input = bias_variables([10])

	out_hidden = tf.nn.relu(tf.matmul(x_input, w_input) + b_input)

	w_hidden = weight_varibles([10, 3])
	b_hidden = bias_variables([3])

	out_logits = tf.matmul(out_hidden, w_hidden) + b_hidden

	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_logits,
																		labels=y_label))
	train = tf.train.AdamOptimizer(1e-4).minimize(loss)
	correct_prediction = tf.equal(tf.argmax(out_logits,1), y_label)

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(max_steps):
			indices = np.random.choice(train_x.shape[0], batch_size)
			train_x1 = train_x[indices]
			train_y1 = train_y[indices]
			if i%100 == 0:
				train_accuracy = sess.run(accuracy, feed_dict={
					x_input:train_x1,
					y_label:train_y1
					})

				print ('Step:{}, Train Accuracy:{}'.format(i, train_accuracy))

			sess.run(train, feed_dict={
				x_input:train_x1,
				y_label:train_y1
				})
		test_accuracy = sess.run(accuracy, feed_dict={
			x_input:test_x,
			y_label:test_y
			})
		print ('Test Accuracy:{}'.format(test_accuracy))
		end_time = time.time()
		print ('Time Taken:{}'.format(end_time-start_time))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
  	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  	FLAGS, unparsed = parser.parse_known_args()
  	start_time = time.time()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)