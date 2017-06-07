import tensorflow as tf
import numpy as np 
import argparse, sys
import time
from datahelpers import datahelpers

n_iteration = 1000
batch_size = 100
n_hidden1 = 1024
n_output = 300
learning_rate = 1e-4
hinge_margin = tf.constant(0.1, shape=[100, 99])
zero = tf.zeros([100, 99])

def weight_variable(shape):
	initial = tf.truncated_normal(shape=shape, stddev=0.9)
	return tf.Variable(initial)

def bias_variable(shape, value=0.8):
	initial = tf.constant(value, shape=shape)
	return tf.Variable(initial)

def main(_):
	print('Loading Data\n')
	dataset = datahelpers()
	print('Data loaded into dataset\n')

	n_class = dataset['NUMBER_OF_CLASSES']
	test_class = dataset['defaultTestClassLabels']   # test classes are unseen classes
	datasetlabels = dataset['datasetLabels']
	features = dataset['vggFeatures']      	# [30475, 4096]
	attributes = dataset['attributes']     	# [50, 300]
	n_totalExamples = len(datasetlabels)

	train_x = list([])
	test_x = list([])
	train_y = list([])
	test_y = list([])

	for example in range(n_totalExamples):
		if datasetlabels[example] in test_class:
			test_x.append(features[example])
			test_y.append(datasetlabels[example])
		else:
			train_x.append(features[example])
			train_y.append(datasetlabels[example])

	train_x = np.array(train_x)
	train_y = np.array(train_y)
	train_y = (train_y.reshape([1, len(train_x)])[0])
	test_x = np.array(test_x)
	test_y = np.array(test_y)
	test_y = (test_y.reshape([1, len(test_x)])[0])
	print('Data separated into train and test classes\n')

	n_trainingExamples = train_x.shape[0]   # ~24000
	n_input = train_x.shape[1]              # 4096

	input_placeholder = tf.placeholder(tf.float32, shape=[None, n_input]) # [batch_size, 4096]
	label_placeholder = tf.placeholder(tf.int32, shape=[None])            # [batch_size]
	attr_placeholder = tf.placeholder(tf.float32, shape=[None, n_output]) # [batch_size, 300]
	minNNindex = tf.placeholder(tf.float32, shape=[None])

	# Hidden layer 1
	w1 = weight_variable([n_input, n_hidden1])
	b1 = bias_variable([n_hidden1])

	out_hidden1 = tf.nn.relu(tf.matmul(input_placeholder, w1) + b1)

	# output layer
	w_out = weight_variable([n_hidden1, n_output])
	b_out = bias_variable([n_output])

	output = tf.matmul(out_hidden1, w_out) + b_out  # prediction or vector repr in embedding 
																# space [batch_size, 300]
	#output = tf.nn.softmax(logits=h, dim=1)

	'''
	# L2 loss implemented
	loss = tf.reduce_mean((output - attr_placeholder)*(output - attr_placeholder))
	'''

	# Hinge loss implementation
	matrix = tf.matmul(output,tf.transpose(attr_placeholder))   # [100, 100]

	generate_exp = tf.add(hinge_margin, [[(-1*matrix[i][i] + matrix[i][j]) for j in range(100) if j!=i]
											for i in range(100)])
	# loss = tf.reduce_sum(tf.maximum(zero, generate_exp), reduction_indices = 1)
	loss = tf.reduce_sum(tf.maximum(zero, generate_exp))
	print('Loss function added to the tensorflow graph\n')

	train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	accuracy = tf.reduce_mean(tf.cast(tf.equal(minNNindex,tf.cast(label_placeholder, tf.float32)), tf.float32))

	with tf.Session() as sess:
		print('Session Started')
		sess.run(tf.global_variables_initializer())
		#attr = tf.cast(tf.negative(attributes), tf.float32)   # for squared loss
		attr = tf.cast(tf.transpose(attributes), tf.float32)   # for hinge loss

		for i in range(n_iteration):
			# an array of length batch_size from [0,trainingExamples-1]
			indices = np.random.choice(n_trainingExamples, batch_size) 
			train_batch_x = train_x[indices]             # [100, 4096]

			train_batch_y = train_y[indices]              # [100]
			train_batch_attr = attributes[train_batch_y-1]  #[100, 300]

			sess.run(train, feed_dict={
				input_placeholder:train_batch_x,
				attr_placeholder:train_batch_attr
				})

			if i%101 == 0 and i!=0:
				#temp_list = []      # temp_list will store the predicted class of the 100 inputs

				'''
				# for squared loss
				for j in range(batch_size):
					print('Inside for loop:'+str(j))

					prediction = tf.constant(batch_prediction[j])   # [300, ]
					prediction = tf.reshape(prediction, [1, 300])

					# temp_list is going wrong here
					"""
					# for squared loss
					temp_list.append(tf.argmin(tf.sqrt(tf.reduce_sum(tf.square(tf.add(prediction, 
						attr)), reduction_indices=1))))
					"""

					"""
					# for hinge loss
					z = (tf.matmul(prediction, attr)[0])
					temp_list.append(tf.argmax(z))
					"""
				'''
				'''
				print('Output:{}'.format(sess.run(output, feed_dict={
					input_placeholder:train_batch_x
					})))
				'''
				# output is coming different from batch_prediction
				z = tf.matmul(sess.run(output,feed_dict={input_placeholder:train_batch_x}), attr)   # [100, 50]
				temp_list = tf.argmax(z, axis=1)

				temp_list = np.array(sess.run(temp_list, feed_dict={
					input_placeholder:train_batch_x
					}))
				print('Predicted labels:{}, Actual labels:{}'.format(temp_list, train_batch_y))
				train_accuracy = sess.run(accuracy, feed_dict={
					label_placeholder:train_batch_y,
					minNNindex:temp_list
					})
				print ('Step:{}, Training Accuracy:{}'.format(i, train_accuracy))
			print('Step:{}'.format(i))
			
	end_time = time.time()
	print('Training Time:{}'.format(end_time - start_time))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	path = '/home/saqib1707/Documents/MyProjects/Zero-Shot_learning/zsl/'

  	parser.add_argument('--data_dir', type=str, default=path, help='Directory for storing input data')
  	FLAGS, unparsed = parser.parse_known_args()
  	start_time = time.time()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)