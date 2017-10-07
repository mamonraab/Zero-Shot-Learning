'''
Apy dataset| single FC layer| no activation function| w2v semantic space| embedding space-semantic space
hinge ranking loss| no regularization | attribute-300 normalization is necessary| 
'''
import tensorflow as tf
import numpy as np
import sys
import time
from datahelpers import datahelpersCUB
import matplotlib.pyplot as plt
import random

n_iteration = 1500  #10 epochs with random shuffling after each epoch 
batch_size = 64
n_output = 312
learning_rate = 1e-2  # less the learning rate more close it will reach the global minimum
alpha = 0.0
margin_value = 5.0
summary_dir = '/tmp/zsl-apy'

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
'''
def get_next_batch_new(step, n_fullbatch):
	flag = False
	step = step%n_fullbatch
	if step == n_fullbatch-1:
		flag = True
	return range(step*batch_size, (step+1)*batch_size), flag
'''
def training(train_x, train_y, train_class, test_x, test_y, test_class, attributes):
	start_time = time.time()
	train_class_len = len(train_class)
	test_class_len = len(test_class)
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
	op2 = [[(-1*matrix[i][index[i]] + matrix[i][j])
				for j in range(train_class_len)] for i in range(batch_size)]
	generate_exp = tf.add(margin_value, op2)

	loss1 = tf.reduce_sum(tf.maximum(0.0, generate_exp), reduction_indices=1)-margin_value
	regularizers = tf.nn.l2_loss(w1)
	loss = tf.reduce_mean(loss1) + (alpha*regularizers)
	print('Loss function added to the tensorflow graph')

	train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(minNNindex,label_placeholder), tf.float32),name='accuracy')
	with tf.Session() as sess:
		#loss_weights = sess.run(tf.nn.l2_normalize(tf.reciprocal(class_dict_val), dim=0))
		#new_dict = dict(zip(class_dict_key, loss_weights))
		#loss = tf.reduce_sum(tf.multiply(tf.reduce_sum(
		#								tf.maximum(zero, generate_exp), reduction_indices=1), weights))
		#loss = tf.reduce_mean(loss+alpha*regularizers)
		attributes = sess.run(tf.nn.l2_normalize(attributes, dim=1))
		print('Normalizing Done')
		sess.run(tf.global_variables_initializer())

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
			indices, flag = get_next_batch(step, n_fullbatch, n_trainingExamples)
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
			loss_val, check_matrix = sess.run([loss,op2], feed_dict={
				input_placeholder:train_batch_x,
				attr_placeholder: train_class_attr,
				index:dontknow
				#weights:w
				})
			if (step+1)%100 == 0:
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
				print ('Step:{}, Training Accuracy:{} % \n check_matrix:{}'.format(step, train_accuracy, check_matrix))
			print('Step:{}, Loss:{}'.format(step, loss_val))

			if flag:
				random.shuffle(lst)
				train_x = train_x[lst]
				train_y = train_y[lst]
			#y_axis.append(loss_val)

		# Finding the learned model accuracy on entire training dataset
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
		return final_train_accuracy, final_test_accuracy

def main(train=True, cv=False, test=False):
	dataset = datahelpersCUB()
	print('Data loaded into dataset')

	n_class = dataset['attr312'].shape[0]
	dataset['defaultTestClassLabels'] = np.array(random.sample(range(1, n_class+1), 50))  # 50 unseen class
	dataset['defaultCVClassLabels'] = np.array(random.sample([label for label in range(1, n_class+1)
										if label not in dataset['defaultTestClassLabels']], 50))
	test_class = dataset['defaultTestClassLabels']
	cv_class = dataset['defaultCVClassLabels']
	train_class = np.array([label for label in range(1, n_class+1) 
				if label not in np.concatenate((test_class, cv_class))])
	extra_class = np.sort(np.concatenate((train_class, cv_class)))
	datasetlabels = dataset['datasetLabels']

	features = dataset['X']      	        # [n_totalExamples, n_input]
	attributes = dataset['attr312']     	# [n_class, n_output] 
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
		'sample':'serialwise-with-random-shuffling-after-each-epoch',
		#'sample':'sample-per-class-same',
		'description':'linear-model-without-activation',
		'attributes':'attr300+norm',
		'dataset':'apy',
		'feature':'vgg',
		'learning rate': learning_rate,
		'n_iterations':n_iteration,
		'batch size':batch_size,
		'number of layers':2,
		#'train_accuracy in cv step':train_accuracy_dict,
		#'cv_accuracy in cv step':cv_accuracy_dict,
		'final train accuracy':final_train_accuracy,
		'final test accuracy':final_test_accuracy
	}
	with open('logfile.md', 'a') as file:
		file.write('\n{}\n'.format(log_dict))

if __name__ == '__main__':
  	train=False
  	cv=True
  	test=False
  	main(train, cv, test)