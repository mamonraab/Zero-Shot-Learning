import tensorflow as tf
import numpy as np
from datahelpers import datahelpers
import time


# restore the saved data and feed it the test data
if __name__ == '__main__':
	start_time = time.time()
	print('Loading Data')
	dataset = datahelpers()
	print('Data loaded into dataset')

	n_output = 300
	n_class = dataset['NUMBER_OF_CLASSES']
	test_class = dataset['defaultTestClassLabels']   # test classes are unseen classes
	datasetlabels = dataset['datasetLabels']
	features = dataset['vggFeatures']      	# [30475, 4096]  0<features[i][j]<1
	attributes = dataset['attributes']     	# [50, 300]   -2<attributes[i][j]<2.05 
	n_totalExamples = datasetlabels.shape[0]

	'''
	test_x = list([])
	test_y = list([])

	for example in range(n_totalExamples):
		if datasetlabels[example] in test_class:
			test_x.append(features[example])
			test_y.append(datasetlabels[example])
	'''
	test_x, test_y = zip(*[(features[ex], datasetlabels[ex]) for ex in range(n_totalExamples)
			 if datasetlabels[ex] in test_class])
	test_x = np.array(test_x)
	test_y = np.array(test_y)
	test_y = test_y.reshape([1, test_x.shape[0]])[0]
	print('Data separated into train and test classes')

	n_testExamples = test_x.shape[0]        # ~6180
	with tf.Session() as sess:
		# for squared loss  [10, 300]
		test_class_attr = tf.cast(tf.negative(attributes[test_class-1]), tf.float32)
		saver = tf.train.import_meta_graph('zsl-awa-w2v-model.meta')
		saver.restore(sess, tf.train.latest_checkpoint('./'))

		graph = tf.get_default_graph()
		input_placeholder=graph.get_tensor_by_name("deepFeatures:0")
		label_placeholder=graph.get_tensor_by_name("actualLabels:0")
		attr_placeholder=graph.get_tensor_by_name("attributes:0")
		minNNindex=graph.get_tensor_by_name("predictedLabels:0")
		output = graph.get_tensor_by_name("output:0")

		temp_list = list([])
		out = sess.run(output, feed_dict={
			input_placeholder:test_x
			})
		for j in range(n_testExamples):
			print('step:'+str(j))
			prediction = tf.reshape(out[j], [1, n_output])   # [300,]
			temp_list.append(tf.argmin(tf.sqrt(tf.reduce_sum(tf.square(tf.add(prediction, 
						test_class_attr)), reduction_indices=1))))

		temp_list = np.array(sess.run(temp_list, feed_dict={
				input_placeholder:test_x
				}))
		pred_result = test_class[temp_list]
		feed_dict={
			input_placeholder:test_x,
			label_placeholder:test_y,
			attr_placeholder:attributes[test_y-1],
			minNNindex:pred_result
		}

		op_to_restore=graph.get_tensor_by_name("op_to_restore:0")  # accuracy
		print('Final Test Accuracy on unseen classes:{} %'.format(sess.run(op_to_restore,
			feed_dict=feed_dict)*100))
	end_time = time.time()
	print('Final testing time:{} mins'.format((end_time-start_time)/60.0))