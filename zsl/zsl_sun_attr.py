# SUN dataset
import tensorflow as tf
import numpy as np
import argparse, sys
import time
from datahelpers import datahelpersSUN
import matplotlib.pyplot as plt
import random

n_iteration = 1200
batch_size = 120
n_output = 102
#n_hidden1 = 1000
learning_rate = 1e-3
#alpha = 0.01
summary_dir = '/tmp/zsl_sun'

def weight_variable(shape):
	initial = tf.truncated_normal(shape=shape, mean=0.0, stddev=2.0, dtype=tf.float32)
	return tf.Variable(initial)

def bias_variable(shape, value=0.5):
	initial = tf.constant(value, shape=shape)
	return tf.Variable(initial)

def get_next_batch(step, n_fullbatch=4, n_trainingExamples=500):
	step = step%(n_fullbatch+1)
	if step%n_fullbatch == 0 and step!=0:
		lst = range(step*batch_size, n_trainingExamples)
		return lst+range(0, batch_size-len(lst)), True
	return range(step*batch_size, (step+1)*batch_size), False

def training(train_x, train_y, train_class, test_x, test_y, test_class, attributes):
	start_time = time.time()
	train_class_len = len(train_class)
	#print ('Train class len:{}'.format(train_class_len))  # 707
	zero = np.zeros([batch_size, train_class_len], dtype=np.float32)
	hinge_margin = np.full((batch_size, train_class_len), 5.0, dtype=np.float32)  # margin = 5.0
	n_trainingExamples = train_x.shape[0]
	#print('Training Examples:{}'.format(n_trainingExamples))  # 14140
	n_input = train_x.shape[1]    # 1024

	# for selecting same number of samples per class while training
	class_dict = {}
	for label in train_class:
		class_dict[label] = 0
	for ex in range(n_trainingExamples):
		class_dict[train_y[ex]] += 1

	input_placeholder = tf.placeholder(tf.float32, shape=[None,n_input], name='deepFeatures') #[batch_size,4096]
	label_placeholder = tf.placeholder(tf.float32, shape=[None], name='actualLabels')   # [batch_size]
	attr_placeholder = tf.placeholder(tf.float32, shape=[n_output, None], name='attributes') # [batch_size, 300]
	minNNindex = tf.placeholder(tf.float32, shape=[None], name='predictedLabels')  # [batch_size]
	index = tf.placeholder(tf.int32, shape=[None], name='index')
	print('Placeholders added')

	# Fully connected layer
	'''
	w1 = weight_variable([n_input, n_hidden1])
	b1 = bias_variable([n_hidden1])

	out_hidden1 = tf.nn.relu(tf.add(tf.matmul(input_placeholder, w1), b1))

	w2 = weight_variable([n_hidden1, n_output])
	b2 = bias_variable([n_output])
	output = tf.add(tf.matmul(out_hidden1, w2),b2, name='output')   # [batch_size, n_output]
	'''
	w1 = weight_variable([n_input, n_output])
	b1 = bias_variable([n_output])

	output = tf.add(tf.matmul(input_placeholder, w1),b1, name='output')
	print('Model addition done')

	# Hinge loss implementation
	# output = [batch_size, n_output]  attr_placeholder = [n_output, train_class_len]
	
	# [batch_size, len(train_class)]
	# generate_exp = [batch_size, len(train_class)-1]
	generate_exp = tf.add(hinge_margin, [[(-1*matrix[i][index[i]] + matrix[i][j]) 
			for j in range(train_class_len) if j!=index[i]]  for i in range(batch_size)])
	#loss = tf.reduce_mean(tf.reduce_sum(tf.square(output - attr_placeholder),
	#										reduction_indices=1))  # L2 loss implemented
	loss = tf.reduce_sum(tf.maximum(zero, generate_exp))   # for hinge loss
	#regularizers = tf.nn.l2_loss(w1)+tf.nn.l2_loss(w2)
	#loss = tf.reduce_mean(loss+alpha*regularizers)

	print('Loss function added to the tensorflow graph')
	train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(minNNindex,label_placeholder), tf.float32), name='op_to_restore')

	with tf.Session() as sess:
		#attributes = sess.run(tf.nn.l2_normalize(attributes, dim=1))
		#print('Normalizing Done')
		sess.run(tf.global_variables_initializer())

		# for taking into consideration only the 40 seen classes during training
		# for squared loss  [40, 300]
		#train_class_attr = tf.cast(tf.negative(attributes[train_class-1]), tf.float32)
		#test_class_attr = tf.cast(tf.negative(attributes[test_class-1]), tf.float32)

		# for hinge loss
		# train_class_attr = [n_output,len(train_class)]
		train_class_attr=sess.run(tf.cast(tf.transpose(attributes[train_class-1]), tf.float32))
		test_class_attr =sess.run(tf.cast(tf.transpose(attributes[test_class-1]),tf.float32))

		x_axis = []
		y_axis = []
		n_fullbatch = int(n_trainingExamples/batch_size)
		lst = range(n_trainingExamples)
		random.shuffle(lst)
		train_x = train_x[lst]
		train_y = train_y[lst]
		print('At flag 1')
		for i in range(n_iteration):
			#indices = random.sample(xrange(n_trainingExamples), batch_size)   # unique random numbers
			'''
			sample_per_class = batch_size/train_class_len
			indices = list([])
			start_idx = 0
			end_idx = 0
			for label in train_class:
				end_idx = start_idx + class_dict[label]
				indices.extend(random.sample(range(start_idx, end_idx), sample_per_class))
				start_idx = end_idx
			random.shuffle(indices)
			'''
			print('At flag 2')
			indices, flag = get_next_batch(i, n_fullbatch, n_trainingExamples)

			train_batch_x = train_x[indices]             # [batch_size, 4096]
			train_batch_y = train_y[indices]              # [batch_size]
			#train_batch_attr = attributes[train_batch_y-1]  #[batch_size, 300]

			dontknow = np.reshape([np.where(train_class==item) for item in train_batch_y], (batch_size))
			print('At flag 3')
			sess.run(train, feed_dict={
				input_placeholder:train_batch_x,
				attr_placeholder:train_class_attr,
				index:dontknow
				})
			if i%100 == 0 and i!=0:
				'''
				# for squared loss
				temp_list = list([])   # temp_list will store the predicted class of the [batch_size] inputs
				#output_norm = tf.nn.l2_normalize(sess.run(output,feed_dict={
				#	input_placeholder:train_batch_x}), dim=1)
				out = sess.run(output, feed_dict={
					input_placeholder:train_batch_x
					})

				for j in range(batch_size):
					prediction = tf.reshape(out[j], [1,n_output])   # [300,]
					temp_list.append(tf.argmin(tf.reduce_sum(tf.square(tf.add(prediction, 
						train_class_attr)), reduction_indices=1)))
				'''
				# for hinge loss
				# Using cosine similarity between the predicted embedding vector and the embeding vector of 
				# the 40 seen classes
				dot_product = tf.matmul(output, train_class_attr)  # [batch_size, len(train_class)]
				temp_list = np.array(sess.run(tf.argmax(dot_product, axis=1), feed_dict={
					input_placeholder:train_batch_x
					}))
				pred_result = train_class[temp_list]
				#print('Predicted labels:\n{},\n Actual labels:\n{}'.format(pred_result, train_batch_y))
				train_accuracy= sess.run(accuracy, feed_dict={
					label_placeholder:train_batch_y,
					minNNindex:pred_result
					})
				print ('Step:{}, Training Accuracy:{} %'.format(i, train_accuracy*100))
			print('At flag 4')
			loss_val = sess.run(loss, feed_dict={
				input_placeholder:train_batch_x,
				attr_placeholder: train_class_attr,
				index:dontknow
				})
			'''
			if (i == n_iteration-1):
				saver.save(sess,model2save)
			'''
			print('Step:{}, Loss:{}'.format(i, loss_val))
			if flag:
				random.shuffle(lst)
				train_x = train_x[lst]
				train_y = train_y[lst]
			#y_axis.append(loss_val)

		# Finding the learned model accuracy on entire training dataset
		'''
		# for L2 loss model
		temp_list = list([])
		out = sess.run(output,feed_dict={
				input_placeholder:train_x})

		for j in range(n_trainingExamples):
			print('for loop:'+str(j))
			prediction = tf.reshape(out[j], [1,n_output])   # [300,]
			temp_list.append(tf.argmin(tf.reduce_sum(tf.square(tf.add(prediction, 
					train_class_attr)), reduction_indices=1)))
		'''
		print('Calculating the training accuracy on {} classes'.format(train_class_len))
		dot_product = tf.matmul(output, train_class_attr)  # [batch_size, len(train_class)]
		temp_list = np.array(sess.run(tf.argmax(dot_product, axis=1), feed_dict={
			input_placeholder:train_x
			}))
		pred_result = train_class[temp_list]
		final_accuracy = (sess.run(accuracy, feed_dict={
			label_placeholder:train_y,
			minNNindex:pred_result
			}))*100
		print('\nFinal Training Accuracy on all training images:{} %'.format(final_accuracy))
		time_taken = (time.time() - start_time)/60.0
		print('Training Ended with training time:{} mins\n'.format(time_taken))
		# Plotting the loss function
		'''
		x_axis = np.array(range(n_iteration))
		y_axis = np.array(y_axis)
		fig = plt.figure()
		ax = plt.subplot(111)
		ax.plot(x_axis, y_axis)
		plt.title('Loss vs iterations')
		#fig.savefig('/Documents/zero-shot-learning-Saqib/plots/img8.png')
		#line, = plt.plot(x_axis, y_axis, lw=2.0)
		plt.legend()
		plt.show()
		'''

		# hinge loss - cross validation
		# find the cross validation accuracy on the 10 held out classes
		dot_product = tf.matmul(output, test_class_attr)  # [batch_size, len(train_class)]
		temp_list = np.array(sess.run(tf.argmax(dot_product, axis=1), feed_dict={
					input_placeholder:test_x
			}))
		pred_result = test_class[temp_list]
		feed_dict={
			label_placeholder:test_y,
			minNNindex:pred_result
		}
		test_accuracy = (sess.run(accuracy, feed_dict=feed_dict))*100
		print('Final Test Accuracy on unseen classes:{} %'.format(test_accuracy))

		return final_accuracy, test_accuracy

def testing(test_x, test_y, test_class, attributes, model='zsl-awa-w2v-model-0.1.meta'):
	print('Model testing started with model-{}'.format(model))
	n_testExamples = test_x.shape[0]
	with tf.Session() as sess:
		# for squared loss  [10, 300]
		#test_class_attr = tf.cast(tf.negative(attributes[test_class-1]), tf.float32)
		# for hinge loss
		test_class_attr = sess.run(tf.cast(tf.transpose(attributes[test_class-1]), tf.float32))
		saver_test = tf.train.import_meta_graph(model)
		saver_test.restore(sess, tf.train.latest_checkpoint('./'))
		print('Model restored')

		graph = tf.get_default_graph()
		input_placeholder=graph.get_tensor_by_name("deepFeatures:0")
		label_placeholder=graph.get_tensor_by_name("actualLabels:0")
		attr_placeholder=graph.get_tensor_by_name("attributes:0")
		minNNindex=graph.get_tensor_by_name("predictedLabels:0")
		output = graph.get_tensor_by_name("output:0")
		accuracy=graph.get_tensor_by_name("op_to_restore:0")  # accuracy
		print('Data restoring done')

		'''
		# L2 loss implementation
		temp_list = list([])
		out = sess.run(output, feed_dict={
			input_placeholder:test_x
			})
		for j in range(n_testExamples):
			print('step:'+str(j))
			prediction = tf.reshape(out[j], [1, n_output])   # [300,]
			temp_list.append(tf.argmin(tf.sqrt(tf.reduce_sum(tf.square(tf.add(prediction, 
						test_class_attr)), reduction_indices=1))))

		'''
		# hinge loss
		dot_product = tf.matmul(output, test_class_attr)  # [batch_size, len(train_class)]
		temp_list = np.array(sess.run(tf.argmax(dot_product, axis=1), feed_dict={
					input_placeholder:test_x
			}))
		pred_result = test_class[temp_list]
		feed_dict={
			label_placeholder:test_y,
			minNNindex:pred_result
		}
		test_accuracy = (sess.run(accuracy, feed_dict=feed_dict))*100
		print('Final Test Accuracy on unseen classes:{} %'.format(test_accuracy))
	#print('Final testing time:{} mins'.format((end_time-start_time)/60.0))
	return test_accuracy

def main(train=True, cv=False, test=False):
	dataset = datahelpersSUN()
	print('Data loaded into dataset')

	classes = dataset['classes']
	n_class = classes.shape[0]
	test_class = dataset['defaultTestClassLabels']
	cv_class = dataset['defaultCVClassLabels']
	train_class = np.array([label for label in range(1, n_class+1) 
				if label not in np.concatenate((test_class, cv_class))])
	extra_class = np.sort(np.concatenate((train_class, cv_class)))
	#train_class_len = len(train_class)
	datasetlabels = dataset['Y']
	features = dataset['X']      	# [n_totalExamples, n_input]  0<features[i][j]<1
	#print[[features[i][j]for j in range(4096)if (features[i][j]<0 or features[i][j]>1)]for i in range(30475)]
	attributes = dataset['attr']     	# [n_class, n_output]   -2<attributes[i][j]<2.05 
	#print([[attributes[i][j] for j in range(300) if (attributes[i][j]<-2 or attributes[i][j]>2)]
	# for i in range(50)])
	n_totalExamples = datasetlabels.shape[0]
	train_x = list([])
	test_x = list([])
	train_y = list([])
	test_y = list([])
	cv_x = list([])
	cv_y = list([])

	for ex in range(n_totalExamples):
		if datasetlabels[ex] in train_class:
			train_x.append(features[ex])
			train_y.append(datasetlabels[ex])
		elif datasetlabels[ex] in test_class:
			test_x.append(features[ex])
			test_y.append(datasetlabels[ex])
		else:
			cv_x.append(features[ex])
			cv_y.append(datasetlabels[ex])
			
	train_x = np.array(train_x)
	train_y = np.array(train_y)
	test_x = np.array(test_x)
	test_y = np.array(test_y)
	cv_x = np.array(cv_x)
	cv_y = np.array(cv_y)
	extra_x = np.concatenate((train_x, cv_x))
	extra_y = np.concatenate((train_y, cv_y))
	print('Data separated into train and test classes')

	#n_trainingExamples = train_x.shape[0]   # 24295
	#n_testExamples = test_x.shape[0]        # 6180
	#n_input = train_x.shape[1]              # 4096

	if (train == True):
		margin_value = 0.1
		final_accuracy, time_taken = training(train_x, train_y, train_class, attributes, margin_value)
	# cross-validation
	if (cv == True):
		cv_accuracy_dict = {}
		train_accuracy_dict = {}
		#margin_list = list([5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 50.0])
		margin_value = 5.0
		batch_value = 120
		#batch_list = list([120, 240, 360])
		'''
		for item in batch_list:
			#model2save = 'zsl-awa-w2v-model-{}'.format(item)
			train_accuracy_dict[item], cv_accuracy_dict[item]  = training(train_x, train_y, train_class,
											    cv_x, cv_y, cv_class, attributes, item) #model2save)
			#saved_model = 'zsl-awa-w2v-model-{}.meta'.format(item)
			#cv_accuracy[item] = testing(cv_x, cv_y, cv_class, attributes, saved_model)
		'''

		#print('Cross validation accuracy on all margin values:{}'.format(cv_accuracy_dict))
		# margin value for which cv accuracy is maximum
		#batch_value = cv_accuracy_dict.keys()[np.argmax(cv_accuracy_dict.values())]
		# Now using this margin value train and test on the 40 classes and 10 classes resp.
		#test_model = 'zsl-awa-w2v-model-{}.meta'.format(margin_value)
		#test_accuracy = testing(test_x, test_y, test_class, attributes, test_model)
 
		final_train_accuracy, final_test_accuracy = training(extra_x, extra_y, extra_class, test_x, test_y,
												test_class, attributes)
		print('Final training and test accuracy:{} % , {} %'.format(final_train_accuracy, final_test_accuracy))

	if (test == True):
		test_model = 'zsl-awa-w2v-model-{}.meta'.format(0.1)
		test_accuracy = testing(test_x, test_y, test_class, attributes, test_model)
	# logging useful information in a file
	log_dict = {
		#'alpha':alpha,
		'sample':'serialwise-with-random-shuffling-after-1-epoch',
		'description':'linear model',
		'dataset':'sun',
		'learning rate': learning_rate,
		'n_iterations':n_iteration,
		'batch size':batch_size,
		'hinge margin':margin_value,
		'number of layers':2,
		#'n_hidden1':n_hidden1,
		#'optimum margin':margin_value,
		#'Non-linearity':'relu',
		#'train_accuracy in cv step':train_accuracy_dict,
		#'cv_accuracy in cv step':cv_accuracy_dict,
		'final train accuracy':final_train_accuracy,
		'final test accuracy':final_test_accuracy
	}
	with open('logfile.md', 'a') as file:
		file.write('\n{}'.format(log_dict))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	path = '/home/saqib1707/Documents/MyProjects/Zero-Shot_learning/zsl'

  	parser.add_argument('--data_dir', type=str, default=path, help='Directory for storing input data')
  	parser.add_argument('--train', type=bool, default=True)
  	parser.add_argument('--cv', type=bool, default=False)
  	parser.add_argument('--test', type=bool, default=False)
  	args = parser.parse_args()
  	train=False
  	cv=True
  	test=False
  	main(train, cv, test)

