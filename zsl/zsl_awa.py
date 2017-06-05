import tensorflow as tf
import numpy as np 
import argparse, sys
import time
from datahelpers import datahelpers

n_iteration = 20000
batch_size = 100
n_hidden1 = 1024
n_output = 300
learning_rate = 0.05

def weight_variable(shape):
	initial = tf.truncated_normal(shape=shape, stddev=0.9)
	return tf.Variable(initial)

def bias_variable(shape, value=0.8):
	initial = tf.constant(value, shape=shape)
	return tf.Variable(initial)

def main(_):
	dataset = datahelpers()
	n_class = dataset['NUMBER_OF_CLASSES']
	# test classes are unseen classes
	test_class = dataset['defaultTestClassLabels']
	datasetlabels = dataset['datasetLabels']
	#datasetlabels = datasetlabels.reshape([1, ])[0]
	#print datasetlabels
	features = dataset['vggFeatures']      # [30475, 4096]
	#print features.shape
	attributes = dataset['attributes']     # [50, 300]
	#print ('Attributes:{}'.format(attributes[10]))
	#print attributes.shape
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
			#print features[example]
			train_y.append(datasetlabels[example])

	train_x = np.array(train_x)
	#print train_x
	train_y = np.array(train_y)
	#print train_y
	train_y = (train_y.reshape([1, len(train_x)])[0])
	#print ('Train_y:{}'.format(train_y))
	#print train_y
	test_x = np.array(test_x)
	test_y = np.array(test_y)

	n_trainingExamples = len(train_x)

	n_input = train_x.shape[1]  # 4096

	input_placeholder = tf.placeholder(tf.float32, shape=[None, n_input]) # [batch_size, 4096]
	label_placeholder = tf.placeholder(tf.int32, shape=[None])            # [batch_size]
	attr_placeholder = tf.placeholder(tf.float32, shape=[None, n_output]) # [batch_size, 300]
	#prediction = tf.placeholder(tf.float32, shape=[1, 300])               # [1, 300]
	minNNindex = tf.placeholder(tf.float32, shape=[None])

	# Hidden layer 1
	w1 = weight_variable([n_input, n_hidden1])
	b1 = bias_variable([n_hidden1])

	out_hidden1 = tf.nn.relu(tf.matmul(input_placeholder, w1) + b1)

	# output layer
	w_out = weight_variable([n_hidden1, n_output])
	b_out = bias_variable([n_output])

	output = tf.nn.relu(tf.matmul(out_hidden1, w_out) + b_out)  # prediction or vector repr in embedding 
																# space [batch_size, 300]

	#output = tf.nn.softmax(logits=h, dim=1)

	loss = tf.reduce_mean((output - attr_placeholder)*(output - attr_placeholder))

	train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	accuracy = tf.reduce_mean(tf.cast(tf.equal(minNNindex,tf.cast(label_placeholder, tf.float32)), tf.float32))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for i in range(n_iteration):
			indices = np.random.choice(n_trainingExamples, batch_size)
			train_batch_x = train_x[indices]             # [100, 4096]
			#print train_batch_x.shape
			train_batch_y = train_y[indices]              # [100]
			#print train_batch_y
			train_batch_attr = attributes[train_batch_y-1]  #[100, 300]
			#print train_batch_attr.shape

			_, prediction1 = sess.run([train, output], feed_dict={
				input_placeholder:train_batch_x,
				attr_placeholder:train_batch_attr
				})

			if i%100 == 0:
				# do something
				temp_list = []
				'''
				print('prediction:{}'.format(sess.run(h, feed_dict={
					input_placeholder:train_batch_x
					})))
				'''
				'''
				prediction1 = (sess.run(output, feed_dict={
					input_placeholder:train_batch_x
					}))
				'''
				#prediction1 = output
				for j in range(batch_size):
					'''
					prediction = sess.run(h, feed_dict={
						input_placeholder:train_batch_x
						})
					prediction = prediction[j]
					'''
					prediction = prediction1[j]   # [100,]
					temp_list.append(tf.argmin(tf.sqrt(tf.reduce_sum(tf.square(tf.add(prediction, 
						tf.cast(tf.negative(attributes), tf.float32))), reduction_indices=1))))


					#print sess.run(tf.sqrt(tf.reduce_mean(tf.square(tf.add(prediction, 
					#	tf.cast(tf.negative(attributes), tf.float32))), reduction_indices=1)))

				temp_list = np.array(sess.run(temp_list))
				print temp_list
				train_accuracy = sess.run(accuracy, feed_dict={
					input_placeholder:train_batch_x,
					attr_placeholder:train_batch_attr,
					label_placeholder:train_batch_y,
					minNNindex:temp_list
					})
				print ('Step:{}, Training Accuracy:{}'.format(i, train_accuracy))
			
	end_time = time.time()
	print('Training Time:{}'.format(end_time - start_time))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	path = '/home/saqib1707/Documents/MyProjects/Zero-Shot_learning/zsl/'

  	parser.add_argument('--data_dir', type=str, default=path, help='Directory for storing input data')
  	FLAGS, unparsed = parser.parse_known_args()
  	start_time = time.time()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)