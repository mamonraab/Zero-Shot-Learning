'''
AwA dataset| two FC layer| relu(both layers)| w2v/85-dim semantic space| embedding space-visual
feature space | L2 loss| L2-regularization|
'''
import tensorflow as tf
import numpy as np
import argparse, sys
import time
from datahelpers import datahelpersAWA, datahelpersAWAfused
import matplotlib.pyplot as plt
import random

n_iteration = 28250   #10 epochs with random shuffling after each epoch 
batch_size = 43
n_output = 300
n_hidden1 = 300
learning_rate = 1e-2
alpha = 1e-3            # regularization parameter(hyper-parameter)
summary_dir = '/tmp/zsl_awa_model'

def weight_variable(shape):
	initial = tf.truncated_normal(shape=shape, mean=0.0, stddev=1.0, dtype=tf.float32)
	return tf.Variable(initial)

def bias_variable(shape, value=0.5):
	initial = tf.constant(value, shape=shape)
	return tf.Variable(initial)
'''
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

def training(train_x, train_y, train_class, test_x, test_y, test_class, attributes):
	start_time = time.time()
	train_class_len = len(train_class)   # 40
	test_class_len = len(test_class)
	n_trainingExamples = train_x.shape[0]
	n_testExamples = test_x.shape[0]
	n_input = train_x.shape[1]

	mean = np.mean(train_x, axis=0)
	std = np.max(train_x, axis=0)-np.min(train_x, axis=0)
	train_x = (train_x-mean)/std
	test_x = (test_x-mean)/std

	# for selecting same number of samples per class while training
	'''
	class_dict = {}
	for label in train_class:
		class_dict[label] = 0
	for ex in range(n_trainingExamples):
		class_dict[train_y[ex]] += 1
	'''
	input_placeholder = tf.placeholder(tf.float32, shape=[None,n_input], name='visualfeatures') #[batch_size, 300/85]
	label_placeholder = tf.placeholder(tf.float32, shape=[None], name='actualLabels')   # [batch_size]
	attr_placeholder = tf.placeholder(tf.float32, shape=[None, n_output], name='attributes') # [batch_size, 1024]
	minNNindex = tf.placeholder(tf.float32, shape=[None], name='predictedLabels')  # [batch_size]
	print('Placeholders added')

	# Fully connected layers
	w1 = weight_variable([n_input, n_hidden1])
	b1 = bias_variable([n_hidden1])
	out_hidden1 = tf.nn.relu(tf.add(tf.matmul(input_placeholder, w1), b1))
	w2 = weight_variable([n_hidden1, n_output])
	b2 = bias_variable([n_output])
	output = tf.add(tf.matmul(out_hidden1, w2),b2)
	'''
	w1 = weight_variable([n_input, n_output])
	b1 = bias_variable([n_output])
	output = tf.nn.relu(tf.matmul(input_placeholder, w1)+b1)
	'''
	print('Model addition done')

	loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(output - attr_placeholder),
											reduction_indices=1))  # L2 loss implemented
	regularizers = tf.nn.l2_loss(w1)            #+tf.nn.l2_loss(w2)
	loss = loss1+(alpha*regularizers)/batch_size

	print('Loss function added to the tensorflow graph')
	train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(minNNindex,label_placeholder), tf.float32), name='op_to_restore')

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#train_x, test_x = sess.run([tf.nn.l2_normalize(train_x, dim=1),
		#							tf.nn.l2_normalize(test_x, dim=1)])
		'''
		attributes = sess.run(tf.nn.l2_normalize(attributes, dim=1))
		train_x = sess.run(tf.nn.l2_normalize(train_x, dim=1))
		test_x = sess.run(tf.nn.l2_normalize(test_x, dim=1))
		'''
		#print('Normalizing Done')
		# for taking into consideration only the 40 seen classes during training
		# for squared loss  [40, 300]
		train_class_attr = tf.cast(attributes[train_class-1], tf.float32)
		test_class_attr = tf.cast(attributes[test_class-1], tf.float32)
		#x_axis = []
		#y_axis = []

		n_fullbatch = int(n_trainingExamples/batch_size)
		lst = range(n_trainingExamples)
		random.shuffle(lst)
		train_x = train_x[lst]
		train_y = train_y[lst]

		for step in range(n_iteration):
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
			#indices, flag = get_next_batch(i, n_fullbatch, n_trainingExamples)
			indices, flag = get_next_batch_new(step, n_fullbatch)
			train_batch_x = train_x[indices]             # [batch_size, 1024]
			train_batch_y = train_y[indices]              # [batch_size]
			train_batch_attr = attributes[train_batch_y-1]  #[batch_size, 300] (most of these will be same)

			sess.run(train, feed_dict={
				input_placeholder:train_batch_x,
				attr_placeholder:train_batch_attr
				})
			loss_val = sess.run(loss, feed_dict={
				input_placeholder:train_batch_x,
				attr_placeholder:train_batch_attr
				})
			if (step+1)%100 == 0:
				temp_list = list([])
				out = sess.run(output, feed_dict={
					input_placeholder:train_batch_x
					})
				'''
				count = 0
				for x in range(40):
					for y in range(n_output):
						if out[x][y] > 0:
							count+=1
				print ('This is the count:{}'.format(count))
				'''
				for j in range(batch_size):
					temp_list.append(tf.argmin(tf.reduce_sum(tf.square(tf.subtract(out[j], 
						train_class_attr)), reduction_indices=1)))
				temp_list = sess.run(temp_list)
				pred_result = train_class[temp_list]
				#print('Predicted labels:\n{},\n Actual labels:\n{}'.format(pred_result, train_batch_y))
				train_accuracy= sess.run(accuracy, feed_dict={
					label_placeholder:train_batch_y,
					minNNindex:pred_result
					})*100
				print ('Step:{}, Training Accuracy:{} %'.format(step, train_accuracy))
			print('Step:{}, Loss:{}'.format(step, loss_val))

			if flag:
				random.shuffle(lst)
				train_x = train_x[lst]
				train_y = train_y[lst]
			#y_axis.append(loss_val)

		# Finding the learned model accuracy on entire training dataset for L2 loss model
		print('Calculating the training accuracy on {} classes'.format(train_class_len))
		temp_list = list([])
		temp = list([])
		out = sess.run(output, feed_dict={
			input_placeholder:train_x
		})
		for j in range(n_trainingExamples):
			#print('Inside step:{}'.format(j))
			temp.append(tf.argmin(tf.reduce_sum(tf.square(tf.subtract(out[j],
				train_class_attr)), reduction_indices=1)))
			if (j+1)%6000 == 0:
				temp = sess.run(temp)
				temp_list.extend(temp)
				temp = []
		temp = sess.run(temp)
		temp_list.extend(temp)
		temp = []
		pred_result = train_class[temp_list]
		final_train_accuracy = (sess.run(accuracy, feed_dict={
			label_placeholder:train_y,
			minNNindex:pred_result
			}))*100
		time_taken = (time.time() - start_time)/60.0
		print('Training Ended with training time:{} mins\n'.format(time_taken))
		print('Calculating the test accuracy on {} classes'.format(test_class_len))
		temp_list = list([])
		out = sess.run(output, feed_dict={
			input_placeholder:test_x
		})
		for j in range(n_testExamples):
			temp_list.append(tf.argmin(tf.reduce_sum(tf.square(tf.subtract(out[j],
				test_class_attr)), reduction_indices=1)))
		temp_list = sess.run(temp_list)
		pred_result = test_class[temp_list]
		final_test_accuracy = (sess.run(accuracy, feed_dict={
			label_placeholder:test_y,
			minNNindex:pred_result
			}))*100
		#print('Final Test Accuracy on unseen classes:{} %'.format(test_accuracy))

		'''
		# Plotting the loss function
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
	dataset = datahelpersAWA()
	mat = datahelpersAWAfused()
	#dataset['attr2'] = mat['attr2']   # for 85-dim continuous attributes
	dataset['features'] = mat['X']     # googleNet feature-1024 dimensional

	print('Data loaded into dataset')

	n_class = dataset['NUMBER_OF_CLASSES']
	test_class = dataset['defaultTestClassLabels']
	cv_class = dataset['defaultCVClassLabels']
	train_class = np.array([label for label in range(1, n_class+1) 
				if label not in np.concatenate((test_class, cv_class))])
	extra_class = np.sort(np.concatenate((train_class, cv_class)))
	datasetlabels = dataset['datasetLabels']

	# feature scaling and mean normalization
	features = dataset['features']      	# [n_totalExamples, n_input]   0 <= features[i][j] <= 26.93
	attributes = dataset['attributes']     	# w2v  [n_class, n_output]  -1.559 <= attributes[i][j] <= 2.046

	#attributes = dataset['attr2']           # 85 dim continuous attr   -1.0 <= attributes[i][j] <= 100.0
	#attr_mean = np.mean(attributes, axis=0)
	#attr_stddev = np.std(attributes, axis=0)
	#attributes = (attributes-attr_mean)/attr_stddev          # -2.36 <= attributes[i][j] <= 6.83

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
		elif datasetlabels[ex] in cv_class:
			cv_x.append(features[ex])
			cv_y.append(datasetlabels[ex])
		else:
			test_x.append(features[ex])
			test_y.append(datasetlabels[ex])
			
	train_x = np.array(train_x)
	train_y = np.array(train_y)
	test_x = np.array(test_x)
	test_y = np.array(test_y)
	cv_x = np.array(cv_x)
	cv_y = np.array(cv_y)
	extra_x = np.concatenate((train_x, cv_x))
	extra_y = np.concatenate((train_y, cv_y))
	print('Data separated into train and test classes')

	if (train == True):
		margin_value = 0.1
		final_accuracy, time_taken = training(train_x, train_y, train_class, attributes, margin_value)
	# cross-validation
	if (cv == True):
		cv_accuracy_dict = {}
		train_accuracy_dict = {}
		#batch_value = 240
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
		#'alpha':'no-regularization',
		'alpha':alpha,
		#'sample':'random',
		'metric':'distance-minimization',
		'sample':'serialwise-with-random-shuffling-after-each-epoch',
		#'sample':'sample-per-class-same',
		#'description':'two-FC-layers-with-one-hidden-layer-relu(85-dim-continuous-attr)',
		'description':'two-FC-layers-one-relu-opposite-ldezm',
		'attributes':'attr300',
		'features':'googlenet+feature-scale',
		'dataset':'awa',
		'learning rate': learning_rate,
		'n_iterations':n_iteration,
		'batch size':batch_size,
		'number of layers':2,
		#'n_hidden1':n_hidden1,
		#'Non-linearity':'relu',
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
