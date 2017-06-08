import tensorflow as tf
import numpy as np 
import argparse, sys
import time
from datahelpers import datahelpers
import matplotlib.pyplot as plt

n_iteration = 1000
batch_size = 100
n_hidden1 = 1024
n_output = 300
learning_rate = 1e-4
hinge_margin = tf.constant(0.1, shape=[batch_size, batch_size-1])
zero = tf.zeros([batch_size, batch_size-1])
#summary_dir = '/home/saqib1707/Documents/MyProjects/Zero-Shot-Learning/zsl' # can't use this summary dir
																			# when running on remote pc
summary_dir = '/tmp/zsl_awa'

def weight_variable(shape):
	initial = tf.truncated_normal(shape=shape, mean=0.0, stddev=1.0, dtype=tf.float32)
	return tf.Variable(initial)

def bias_variable(shape, value=0.1):
	initial = tf.constant(value, shape=shape)
	return tf.Variable(initial)

def main(_):
	print('Loading Data')
	dataset = datahelpers()
	print('Data loaded into dataset')

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
	train_y = (train_y.reshape([1, train_x.shape[0]])[0])
	test_x = np.array(test_x)
	test_y = np.array(test_y)
	test_y = (test_y.reshape([1, test_x.shape[0]])[0])
	print('Data separated into train and test classes')

	n_trainingExamples = train_x.shape[0]   # ~24000
	n_input = train_x.shape[1]              # 4096

	print('Placeholders declaration')
	input_placeholder = tf.placeholder(tf.float32, shape=[None, n_input]) # [batch_size, 4096]
	label_placeholder = tf.placeholder(tf.float32, shape=[None])            # [batch_size]
	attr_placeholder = tf.placeholder(tf.float32, shape=[None, n_output]) # [batch_size, 300]
	minNNindex = tf.placeholder(tf.float32, shape=[None])
	print('Placeholders added')

	# Hidden layer 1
	print('Model addition to graph')
	w1 = weight_variable([n_input, n_hidden1])
	b1 = bias_variable([n_hidden1])

	# out_hidden1 = tf.nn.relu(tf.matmul(input_placeholder, w1) + b1)
	out_hidden1 = tf.matmul(input_placeholder, w1) + b1         # for linear transformation

	# output layer
	w_out = weight_variable([n_hidden1, n_output])
	b_out = bias_variable([n_output])

	output = tf.matmul(out_hidden1, w_out) + b_out  # prediction or vector repr in embedding 
																# space [batch_size, 300]
	#output = tf.nn.softmax(logits=h, dim=1)
	print('Model addition done')

	'''
	# L2 loss implemented
	loss = tf.reduce_mean((output - attr_placeholder)*(output - attr_placeholder))
	'''
	print('Loss function addition to graph')
	# Hinge loss implementation
	matrix = tf.matmul(output,tf.transpose(attr_placeholder))   # [batch_size, batch_size]

	generate_exp = tf.add(hinge_margin, [[(-1*matrix[i][i] + matrix[i][j]) for j in range(batch_size) if j!=i]
											for i in range(batch_size)])         # [batch_size, batch_size-1]
	#loss = tf.reduce_sum(tf.maximum(zero, generate_exp), reduction_indices = 1)
	with tf.name_scope('Loss'):
		loss = tf.reduce_sum(tf.maximum(zero, generate_exp))
		tf.summary.scalar('Loss/Cost', loss)

	print('Loss function added to the tensorflow graph')

	train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	accuracy = tf.reduce_mean(tf.cast(tf.equal(minNNindex,label_placeholder), tf.float32))

	with tf.Session() as sess:
		# Normalizing attributes
		print('Normalizing attributes')
		attributes = sess.run(tf.nn.l2_normalize(attributes, dim=1))
		'''
		with tf.Session() as sess:
			attr_sum_root = sess.run(tf.sqrt(tf.reduce_sum(tf.square(attributes), reduction_indices=1)))  # [50,]
			for i in range(attributes.shape[0]):
				attributes[i] = tf.divide(attributes[i], attr_sum_root[i])
			print(sess.run(tf.sqrt(tf.reduce_sum(tf.square(attributes), reduction_indices=1))))
		'''
		print('Normalizing Done')
		print('Session Started')
		# Summary merged
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(summary_dir + '/train',
                                      sess.graph)
		test_writer = tf.summary.FileWriter(summary_dir + '/test')
		sess.run(tf.global_variables_initializer())

		# for taking into consideration only the 40 seen classes during training
		train_class_attr = list([])
		for k in range(n_class):
			if (k+1) not in test_class:
				train_class_attr.append(attributes[k])
		train_class_attr = np.array(train_class_attr)
		#attr = tf.cast(tf.negative(lst), tf.float32)   # for squared loss
		attr = tf.cast(tf.transpose(train_class_attr), tf.float32)   # for hinge loss [300, 40]

		x_axis = []
		y_axis = []
		for i in range(n_iteration):
			# an array of length batch_size from [0,trainingExamples-1]
			indices = np.random.choice(n_trainingExamples, batch_size) 
			train_batch_x = train_x[indices]             # [batch_size, 4096]
			train_batch_y = train_y[indices]              # [batch_size]
			train_batch_attr = attributes[train_batch_y-1]  #[batch_size, 300]

			_, summary = sess.run([train, merged], feed_dict={
				input_placeholder:train_batch_x,
				attr_placeholder:train_batch_attr
				})
			train_writer.add_summary(summary, i)

			if i%101 == 0 and i!=0:    # 101 has no relation to batch_size
				'''
				# for squared loss
				temp_list = list([])    # temp_list will store the predicted class of the [batch_size] inputs
				batch_prediction = sess.run(output, feed_dict={
					input_placeholder:train_batch_x
					})
				for j in range(batch_size):
					print('Inside for loop:'+str(j))

					prediction = tf.constant(batch_prediction[j])   # [300,]
					prediction = tf.reshape(prediction, [1, 300])

					# temp_list is going wrong here
					temp_list.append(tf.argmin(tf.sqrt(tf.reduce_sum(tf.square(tf.add(prediction, 
						attr)), reduction_indices=1))))
				'''

				# output is coming different from batch_prediction
				# for hinge loss
				output_norm = tf.nn.l2_normalize(sess.run(output,feed_dict={
					input_placeholder:train_batch_x}), dim=1)
				#input2 = tf.nn.l2_normalize(attr, dim=0)
				# Using cosine similarity between the predicted embedding vector and the embeding vector of 
				# the 40 seen classes
				z = sess.run(tf.matmul(output_norm, attr))   # [100, 40]
				temp_list = tf.argmax(z, axis=1)

				temp_list = np.array(sess.run(temp_list, feed_dict={
					input_placeholder:train_batch_x
					}))

				print('Predicted labels:\n{},\n Actual labels:\n{}'.format(temp_list, train_batch_y))

				train_accuracy, summary = sess.run([accuracy, merged], feed_dict={
					input_placeholder:train_batch_x,
					label_placeholder:train_batch_y,
					attr_placeholder:train_batch_attr,
					minNNindex:temp_list
					})
				test_writer.add_summary(summary, i)

				print ('Step:{}, Training Accuracy:{}'.format(i, train_accuracy))
			print('Step:{}'.format(i))
			y_axis.append(sess.run(loss, feed_dict={
				input_placeholder:train_batch_x,
				attr_placeholder: train_batch_attr
				}))
		x_axis = np.array(range(n_iteration))
		y_axis = np.array(y_axis)

		line, = plt.plot(x_axis, y_axis, lw=2.0)
		plt.legend()
		plt.show()
	end_time = time.time()
	print('Training Time:{}'.format((end_time - start_time)/60.0))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	path = '/home/saqib1707/Documents/MyProjects/Zero-Shot_learning/zsl'

  	parser.add_argument('--data_dir', type=str, default=path, help='Directory for storing input data')
  	FLAGS, unparsed = parser.parse_known_args()
  	start_time = time.time()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)