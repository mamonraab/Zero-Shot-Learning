'''
AwA dataset| single FC layer| no activation function| w2v semantic space| embedding space-semantic space
hinge ranking loss| no regularization | attribute-300 normalization is necessary| 
'''
import tensorflow as tf
import numpy as np
import argparse, sys
import time
from datahelpers import datahelpersAWA, datahelpersAWAfused
import matplotlib.pyplot as plt
import random

n_iteration = 5650  #10 epochs with random shuffling after each epoch 
batch_size = 43
n_output = 300
learning_rate = 1e-2  # less the learning rate more close it will reach the global minimum
alpha = 0.001
margin_value = 5.0
summary_dir = '/tmp/zsl-awa-w2v'

def weight_variable(shape):
	initial = tf.truncated_normal(shape=shape, mean=0.0, stddev=1.0, dtype=tf.float32)
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

def get_next_batch_new(step, n_fullbatch):
	flag = False
	step = step%n_fullbatch
	if step == n_fullbatch-1:
		flag = True
	return range(step*batch_size, (step+1)*batch_size), flag

def training(train_x, train_y, train_class, test_x, test_y, test_class, attributes):
	start_time = time.time()
	train_class_len = len(train_class)
	test_class_len = len(test_class)
	#zero = np.zeros([batch_size, train_class_len], dtype=np.float32)
	#hinge_margin = np.full((batch_size, train_class_len), margin_value, dtype=np.float32)  # margin = 5.0
	n_trainingExamples = train_x.shape[0]
	n_testExamples = test_x.shape[0]
	n_input = train_x.shape[1]

	# for selecting same number of samples per class while training
	'''
	class_dict = {}
	for label in train_class:
		class_dict[label] = 0
	for ex in range(n_trainingExamples):
		class_dict[train_y[ex]] += 1
	class_dict_val = np.array(class_dict.values(), np.float32)
	class_dict_key = np.array(class_dict.keys())
	'''
	input_placeholder = tf.placeholder(tf.float32, shape=[None,n_input], name='deepFeatures') #[batch_size,4096]
	label_placeholder = tf.placeholder(tf.float32, shape=[None], name='actualLabels')   # [batch_size]
	attr_placeholder = tf.placeholder(tf.float32, shape=[n_output, None], name='attributes') #[n_output, train_class_len]
	minNNindex = tf.placeholder(tf.float32, shape=[None], name='predictedLabels')  # [batch_size]
	index = tf.placeholder(tf.int32, shape=[None], name='index')
	#weights = tf.placeholder(tf.float32, shape=[None])
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
	output = tf.add(tf.matmul(input_placeholder, w1),b1, name='output')  #[batch_size, n_output]
	print('Model addition done')

	# Hinge loss implementation
	matrix = tf.matmul(output, attr_placeholder)  #[batch_size,len(train_class)]
	generate_exp = tf.add(margin_value, [[(-1*matrix[i][index[i]] + matrix[i][j])
				for j in range(train_class_len)] for i in range(batch_size)])

	loss1 = tf.reduce_sum(tf.maximum(0.0, generate_exp), reduction_indices=1)-margin_value
	regularizers = tf.nn.l2_loss(w1)
	loss = tf.reduce_mean(loss1) + (alpha*regularizers)/batch_size
	print('Loss function added to the tensorflow graph')

	train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(minNNindex,label_placeholder), tf.float32),name='accuracy')
	with tf.Session() as sess:
		#loss_weights = sess.run(tf.nn.l2_normalize(tf.reciprocal(class_dict_val), dim=0))
		#new_dict = dict(zip(class_dict_key, loss_weights))
		#loss = tf.reduce_sum(tf.multiply(tf.reduce_sum(
		#								tf.maximum(zero, generate_exp), reduction_indices=1), weights))
		#regularizers = tf.nn.l2_loss(w1)+tf.nn.l2_loss(w2)
		#loss = tf.reduce_mean(loss+alpha*regularizers)
		attributes = sess.run(tf.nn.l2_normalize(attributes, dim=1))
		print('Normalizing Done')
		sess.run(tf.global_variables_initializer())
		#saver = tf.train.Saver()
		# for taking into consideration only the 40 seen classes during training
		# for squared loss
		#train_class_attr = tf.cast(tf.negative(attributes[train_class-1]), tf.float32)
		#test_class_attr = tf.cast(tf.negative(attributes[test_class-1]), tf.float32)

		# for hinge loss
		train_class_attr=sess.run(tf.cast(tf.transpose(attributes[train_class-1]), tf.float32)) #[n_output,len(train_class)]
		test_class_attr =sess.run(tf.cast(tf.transpose(attributes[test_class-1]),tf.float32))

		#x_axis = []
		#y_axis = []
		n_fullbatch = int(n_trainingExamples/batch_size)
		lst = range(n_trainingExamples)
		random.shuffle(lst)
		train_x = train_x[lst]
		train_y = train_y[lst]

		for step in range(n_iteration):
			'''
			if step > 500:
				global learning_rate
				learning_rate = 1e-4
			'''
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
			#if n_iteration==1500:
			#	indices, flag = get_next_batch(step, n_fullbatch, n_trainingExamples)
			#else:
			indices, flag = get_next_batch_new(step, n_fullbatch)

			train_batch_x = train_x[indices]             # [batch_size, 4096]
			train_batch_y = train_y[indices]              # [batch_size]

			dontknow = np.reshape([np.where(train_class==item) for item in train_batch_y], (batch_size))
			#w = np.array([new_dict[label] for label in train_batch_y])
			sess.run(train, feed_dict={
				input_placeholder:train_batch_x,
				attr_placeholder:train_class_attr,
				index:dontknow
				#weights:w
				})
			loss_val = sess.run(loss, feed_dict={
				input_placeholder:train_batch_x,
				attr_placeholder: train_class_attr,
				index:dontknow
				#weights:w
				})
			if (step+1)%100 == 0:
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
				# Loss metric minimization
				out= sess.run(output, feed_dict={
					input_placeholder:train_batch_x,
					})
				matrix = np.matmul(out, train_class_attr)  #[batch_size, train_class_len]
				loss_per_image = np.zeros([train_class_len, batch_size])  # [batch_size]
				for count in range(train_class_len):
					generate_exp = np.add(margin_value, [[(-1*matrix[i][count] + matrix[i][j])
								for j in range(train_class_len) if j!=count]  for i in range(batch_size)])
					loss_per_image[count] = np.sum(np.maximum(0.0, generate_exp), axis=1)
				temp_list = np.argmin(loss_per_image, axis=0)
				# dot product maximization
				'''
				dot_product = tf.matmul(output, train_class_attr)  # [batch_size, len(train_class)]
				temp_list = np.array(sess.run(tf.argmax(dot_product, axis=1), feed_dict={
					input_placeholder:train_batch_x
					}))
				'''
				pred_result = train_class[temp_list]
				#print('Predicted labels:\n{},\n Actual labels:\n{}'.format(pred_result, train_batch_y))
				train_accuracy= (sess.run(accuracy, feed_dict={
					label_placeholder:train_batch_y,
					minNNindex:pred_result
					}))*100
				print ('Step:{}, Training Accuracy:{} %'.format(step, train_accuracy))
			'''
			if (i == n_iteration-1):
				saver.save(sess,model2save)
			'''
			print('Step:{}, Loss:{}'.format(step, loss_val))

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
		# loss function minimization
		out = sess.run(output, feed_dict={
			input_placeholder:train_x
		})
		matrix = np.matmul(out, train_class_attr)  #[batch_size, train_class_len]
		loss_per_image = np.zeros([train_class_len, n_trainingExamples])
		for count in range(train_class_len):
			generate_exp = np.add(margin_value, [[(-1*matrix[i][count] + matrix[i][j])
						for j in range(train_class_len) if j!=count]  for i in range(n_trainingExamples)])
			loss_per_image[count] = np.sum(np.maximum(0.0, generate_exp), axis=1)
		temp_list = np.argmin(loss_per_image, axis=0)
		# dot product maximization
		'''
		dot_product = tf.matmul(output, train_class_attr)  # [batch_size, len(train_class)]
		temp_list = np.array(sess.run(tf.argmax(dot_product, axis=1), feed_dict={
			input_placeholder:train_x
			}))
		'''
		pred_result = train_class[temp_list]
		final_train_accuracy = (sess.run(accuracy, feed_dict={
			label_placeholder:train_y,
			minNNindex:pred_result
			}))*100
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
		# loss function minimization
		print('Calculating the test accuracy on {} classes'.format(test_class_len))
		out = sess.run(output, feed_dict={
			input_placeholder:test_x
		})
		matrix = np.matmul(out, test_class_attr)  #[n_testExamples, test_class_len]
		loss_per_image = np.zeros([test_class_len, n_testExamples])
		for count in range(test_class_len):
			generate_exp = np.add(margin_value, [[(-1*matrix[i][count] + matrix[i][j])
						for j in range(test_class_len) if j!=count]  for i in range(n_testExamples)])
			loss_per_image[count] = np.sum(np.maximum(0.0, generate_exp), axis=1)
		temp_list = np.argmin(loss_per_image, axis=0)
		# dot product maximization
		'''
		dot_product = tf.matmul(output, test_class_attr)  # [batch_size, len(train_class)]
		temp_list = np.array(sess.run(tf.argmax(dot_product, axis=1), feed_dict={
					input_placeholder:test_x
			}))
		'''
		pred_result = test_class[temp_list]
		final_test_accuracy = (sess.run(accuracy, feed_dict={
			label_placeholder:test_y,
			minNNindex:pred_result
			}))*100
		return final_train_accuracy, final_test_accuracy

"""
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
"""
def main(train=True, cv=False, test=False):
	dataset = datahelpersAWA()
	mat = datahelpersAWAfused()
	features = mat['X']            # googleNet features
	print('Data loaded into dataset')

	n_class = dataset['NUMBER_OF_CLASSES']
	test_class = dataset['defaultTestClassLabels']
	cv_class = dataset['defaultCVClassLabels']
	train_class = np.array([label for label in range(1, n_class+1) 
				if label not in np.concatenate((test_class, cv_class))])
	extra_class = np.sort(np.concatenate((train_class, cv_class)))
	datasetlabels = dataset['datasetLabels']

	#features = dataset['vggFeatures']      	# [n_totalExamples, n_input]  0<features[i][j]<1
	attributes = dataset['attributes']     	# [n_class, n_output]   -2<attributes[i][j]<2.05 
	#attributes = dataset['attr_85']
	n_totalExamples = datasetlabels.shape[0]
	#train_x = train_y = test_x = test_y = list([])  #### wrong since it refers to the same object
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
	'''
	if (train == True):
		#margin_value = 0.1
		final_accuracy, time_taken = training(train_x, train_y, train_class, attributes, margin_value)
	'''
	# cross-validation
	if (cv == True):
		'''
		cv_accuracy_dict = {}
		train_accuracy_dict = {}
		margin_list = list([0.1, 1.0, 5.0, 10.0, 15.0, 20.0])
		#margin_value = 5.0
		alpha_list = list([0.0, 0.001, 0.01, 0.1, 1.0, 5.0])
		
		n_iteration = 1500
		for alpha in alpha_list:
			for margin in margin_list:
				#model2save = 'zsl-awa-w2v-model-{}'.format(item)
				item = (alpha, margin)
				print ('Trying out with this item:{}'.format(item))
				train_accuracy_dict[item], cv_accuracy_dict[item]  = training(train_x, train_y, train_class,
												    cv_x, cv_y, cv_class, attributes, item, n_iteration)
				with open('hyperfile.md', 'a') as file:
					file.write('{}\t{}\n'.format(train_accuracy_dict[item], cv_accuracy_dict[item]))
				#saved_model = 'zsl-awa-w2v-model-{}.meta'.format(item)
				#cv_accuracy[item] = testing(cv_x, cv_y, cv_class, attributes, saved_model)
		print('Training Accuracy on all the hyper-parameters:{}'.format(train_accuracy_dict))
		print('Cross validation accuracy on all the hyper-parameters:{}'.format(cv_accuracy_dict))
		# margin value for which cv accuracy is maximum
		alpha_value, margin_value = cv_accuracy_dict.keys()[np.argmax(cv_accuracy_dict.values())]
		# Now using this margin value train and test on the 40 classes and 10 classes resp.
		#test_model = 'zsl-awa-w2v-model-{}.meta'.format(margin_value)
		#test_accuracy = testing(test_x, test_y, test_class, attributes, test_model)
		item = (alpha_value, margin_value)
		n_iteration=5650
		'''
		final_train_accuracy, final_test_accuracy = training(extra_x, extra_y, extra_class, test_x, test_y,
												test_class, attributes) 
		print('Final training and test accuracy:{} % , {} %'.format(final_train_accuracy, final_test_accuracy))

	# logging useful information in a file
	log_dict = {
		#'alpha_list':alpha_list,
		#'margin-list':margin_list,
		#'alpha-optimized':alpha_value,
		#'margin-optimized':margin_value,
		'metric':'loss-minimization',
		'alpha':alpha,
		'margin':margin_value,
		#'alpha':'no-regularization',
		'sample':'serialwise-with-random-shuffling-after-each-epoch',
		#'sample':'sample-per-class-same',
		'description':'linear-model-without-activation',
		'attributes':'attr300+norm',
		'dataset':'awa',
		'feature':'vgg',
		'learning rate': learning_rate,
		'n_iterations':n_iteration,
		'batch size':batch_size,
		#'hinge margin':margin_value,
		'number of layers':2,
		#'n_hidden1':n_hidden1,
		#'optimum margin':margin_value,
		#'train_accuracy in cv step':train_accuracy_dict,
		#'cv_accuracy in cv step':cv_accuracy_dict,
		'final train accuracy':final_train_accuracy,
		'final test accuracy':final_test_accuracy
	}
	with open('logfile.md', 'a') as file:
		file.write('\n{}\n'.format(log_dict))


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
