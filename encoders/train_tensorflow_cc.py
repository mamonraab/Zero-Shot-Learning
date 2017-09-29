import numpy as np
import pdb
import scipy.io
import tensorflow as tf
import math
import time
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from get_data_for_cc import classifier_data

GPU_PERCENTAGE = 0.9
class train_tf_cc_input:
	cc1_input_train_perm = np.array([])
	cc1_output_train_perm = np.array([])
	cc1_input_valid_perm = np.array([])
	cc1_output_valid_perm = np.array([])
	obj_classifier_data = classifier_data()
	dimension_hidden_layer1 = []
	EPOCHS_CC = []
	classI = []
	classJ = []
	dataset_name = []
	dim_feature = []
    	def function(self):
        	print("This is train_tensorflow_cc_input class")

class train_tf_cc_output:
	decoded_data_train_cc1 = np.array([])
	encoded_data_train_cc1 = np.array([])
	decoded_data_valid_cc1 = np.array([])
	encoded_data_valid_cc1 = np.array([])
	decoded_data_test_cc1 = np.array([])
	encoded_data_test_cc1 = np.array([])
	obj_classifier = classifier_data()

    	def function(self):
        	print("This is train_tensorflow_cc_output class")

def function_train_keras_cc(obj_train_tf_cc_input):
	print "***CC using keras***"	
	n_samp, input_dim = (obj_train_tf_cc_input.cc1_input_train_perm).shape
	#this is the size of our encoded representations
	encoding_dim = obj_train_tf_cc_input.dimension_hidden_layer1 # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

	# this is our input placeholder
	input_img = Input(shape=(input_dim,))
	# "encoded" is the encoded representation of the input
	encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(0.01))(input_img)
	# "decoded" is the lossy reconstruction of the input
	decoded = Dense(input_dim, activation='relu')(encoded)

	# this model maps an input to its reconstruction
	autoencoder = Model(input_img, decoded)
	# this model maps an input to its encoded representation
	encoder = Model(input_img, encoded)

	# create a placeholder for an encoded (32-dimensional) input
	encoded_input = Input(shape=(encoding_dim,))
	# retrieve the last layer of the autoencoder model
	decoder_layer = autoencoder.layers[-1]
	# create the decoder model
	decoder = Model(encoded_input, decoder_layer(encoded_input))

	autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
	autoencoder.fit(obj_train_tf_cc_input.cc1_input_train_perm, obj_train_tf_cc_input.cc1_output_train_perm,
			epochs=100,
			batch_size=100,
			shuffle=True,
			validation_data=(obj_train_tf_cc_input.cc1_input_valid_perm, obj_train_tf_cc_input.cc1_output_valid_perm))

	#pdb.set_trace()	
	#Encoding train-valid-test data
        obj_train_tf_cc_output = train_tf_cc_output()

	obj_train_tf_cc_output.encoded_data_train_cc1 = encoder.predict(obj_train_tf_cc_input.obj_classifier.train_data)
	obj_train_tf_cc_output.decoded_data_train_cc1 = decoder.predict(obj_train_tf_cc_output.encoded_data_train_cc1)

	obj_train_tf_cc_output.encoded_data_valid_cc1 = encoder.predict(obj_train_tf_cc_input.obj_classifier.valid_data)
        obj_train_tf_cc_output.decoded_data_valid_cc1 = decoder.predict(obj_train_tf_cc_output.encoded_data_valid_cc1)

	obj_train_tf_cc_output.encoded_data_test_cc1 = encoder.predict(obj_train_tf_cc_input.obj_classifier.test_data)
	obj_train_tf_cc_output.decoded_data_test_cc1 = decoder.predict(obj_train_tf_cc_output.encoded_data_test_cc1)

	
	#pdb.set_trace()	
	return obj_train_tf_cc_output


def function_train_tensorflow_cc(obj_train_tf_cc_input):
	print "***CC using tensorflow***"	
	n_samp, input_dim = (obj_train_tf_cc_input.cc1_input_train_perm).shape
	n_hidden = obj_train_tf_cc_input.dimension_hidden_layer1
	x = tf.placeholder("float", [None, input_dim])
    # Weights and biases to hidden layer
	Wh = tf.Variable(tf.random_uniform((input_dim, n_hidden), -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))
	bh = tf.Variable(tf.zeros([n_hidden]))
	h = tf.nn.tanh(tf.matmul(x,Wh) + bh)
    # Weights and biases to hidden layer
    #Wo = tf.transpose(Wh) # tied weights
	Wo = tf.Variable(tf.random_uniform((n_hidden, input_dim), -1.0 / math.sqrt(n_hidden), 1.0 / math.sqrt(n_hidden)))
	bo = tf.Variable(tf.zeros([input_dim]))
	y = tf.nn.tanh(tf.matmul(h,Wo) + bo)
    # Objective functions
	y_ = tf.placeholder("float", [None,input_dim])
	cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	meansq = tf.reduce_mean(tf.square(y_-y))
	train_step = tf.train.GradientDescentOptimizer(0.05).minimize(meansq)
	
	#config = tf.ConfigProto()
	#config.gpu_options.per_process_gpu_memory_fraction = GPU_PERCENTAGE

	init = tf.initialize_all_variables()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	sess.run(init)
	n_rounds = obj_train_tf_cc_input.EPOCHS_CC
	batch_size = min(4, n_samp)
	n_samp_valid = obj_train_tf_cc_input.cc1_input_valid_perm.shape[0]
	batch_size_valid = min(100, n_samp_valid)
	#pdb.set_trace()
	cc1_start = time.time()
	number_of_batches = int(n_samp / batch_size)
	print "Number of samples for training %d, batches %d"%(n_samp, batch_size)
	for i in range(n_rounds):
		shuffled_data_indices = np.random.permutation(n_samp)
		for batch in range(number_of_batches):
			sample = shuffled_data_indices[batch*batch_size: batch*batch_size + batch_size]
			batch_xs = obj_train_tf_cc_input.cc1_input_train_perm[sample][:]
			batch_ys = obj_train_tf_cc_input.cc1_output_train_perm[sample][:]
			sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
			#pdb.set_trace()
			if batch % 1000 == 0:
				shuffled_valid_data_indices = np.random.permutation(n_samp_valid)
				sample_valid = shuffled_valid_data_indices[:batch_size_valid]
				batch_x_valid = obj_train_tf_cc_input.cc1_input_valid_perm[sample_valid][:]
				batch_y_valid = obj_train_tf_cc_input.cc1_output_valid_perm[sample_valid][:]
				print "Epoch %4d Batch %4d:  train MSE %f valid MSE %f" %(i, batch, sess.run(meansq, feed_dict={x: obj_train_tf_cc_input.cc1_input_train_perm, y_:obj_train_tf_cc_input.cc1_output_train_perm}), \
				sess.run(meansq, feed_dict={x: batch_x_valid, y_:batch_y_valid}))
			#print i, sess.run(cross_entropy, feed_dict={x: batch_xs, y_:batch_ys}), sess.run(meansq, feed_dict={x: batch_xs, y_:batch_ys})
	cc1_end = time.time()
	cc1_time = cc1_end - cc1_start

  #Get cc features for training  and validation samples
	obj_train_tf_cc_output = train_tf_cc_output()
	obj_train_tf_cc_output.decoded_data_train_cc1 = sess.run(y, feed_dict={x: obj_train_tf_cc_input.obj_classifier.train_data})
	obj_train_tf_cc_output.encoded_data_train_cc1 = sess.run(h, feed_dict={x: obj_train_tf_cc_input.obj_classifier.train_data})

	obj_train_tf_cc_output.decoded_data_valid_cc1 = sess.run(y, feed_dict={x: obj_train_tf_cc_input.obj_classifier.valid_data})
	obj_train_tf_cc_output.encoded_data_valid_cc1 = sess.run(h, feed_dict={x: obj_train_tf_cc_input.obj_classifier.valid_data})
	
  #Get cc features for testing samples
	print"Encoding %d test samples for this class "%obj_train_tf_cc_input.obj_classifier.test_data.shape[0]
	obj_train_tf_cc_output.decoded_data_test_cc1 = sess.run(y, feed_dict={x: obj_train_tf_cc_input.obj_classifier.test_data})
	obj_train_tf_cc_output.encoded_data_test_cc1 = sess.run(h, feed_dict={x: obj_train_tf_cc_input.obj_classifier.test_data})
	print "Exiting tf cc"
	#pdb.set_trace()	
  
 #Get cc features for debug

	debug_valid_decoded = sess.run(y, feed_dict={x: obj_train_tf_cc_input.cc1_input_valid_perm})		
	debug_valid_encoded = sess.run(h, feed_dict={x: obj_train_tf_cc_input.cc1_input_valid_perm})
	dimension_hidden_layer1 = obj_train_tf_cc_input.dimension_hidden_layer1	
	classI = obj_train_tf_cc_input.classI
	classJ = obj_train_tf_cc_input.classJ
	dataset_name = obj_train_tf_cc_input.dataset_name
	file_name = 'data/' + '_debug_' + str(obj_train_tf_cc_input.dim_feature) + '_' + str(dimension_hidden_layer1) + \
		 '_' + dataset_name + '_' + str(classI) + '_' + str(classJ) + '_valid_decoded'
	print file_name	
	scipy.io.savemat(file_name, dict(debug_valid_decoded = debug_valid_decoded))
	file_name = 'data/' + '_debug_' + str(obj_train_tf_cc_input.dim_feature) + '_' + str(dimension_hidden_layer1) + \
		 '_' + dataset_name + '_' + str(classI) + '_' + str(classJ) + '_valid_encoded'
	print file_name	
	scipy.io.savemat(file_name, dict(debug_valid_encoded = debug_valid_encoded))
	
	file_name = 'data/' + '_debug_' + str(obj_train_tf_cc_input.dim_feature) + '_' + str(dimension_hidden_layer1) + \
		 '_' + dataset_name + '_' + str(classI) + '_' + str(classJ) + '_valid_input'
	print file_name	
	scipy.io.savemat(file_name, dict(debug_valid_input = obj_train_tf_cc_input.cc1_input_valid_perm))
	
	file_name = 'data/' + '_debug_' + str(obj_train_tf_cc_input.dim_feature) + '_' + str(dimension_hidden_layer1) + \
		 '_' + dataset_name + '_' + str(classI) + '_' + str(classJ) + '_valid_output'
	print file_name	
	scipy.io.savemat(file_name, dict(debug_valid_output = obj_train_tf_cc_input.cc1_output_valid_perm))
	
	return obj_train_tf_cc_output


class classifier_output:

	accuracy = []
	predicted_labels = []
	
	def function(self):
		print("This is a classifier_output object")

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def function_train_classifier_for_cc(obj_classifier_input):

	obj_classifier_input.train_labels = obj_classifier_input.train_labels - 1
	labels_train = keras.utils.to_categorical(obj_classifier_input.train_labels, num_classes = obj_classifier_input.number_of_train_classes)
	obj_classifier_input.test_labels = obj_classifier_input.test_labels - 1
	labels_test = keras.utils.to_categorical(obj_classifier_input.test_labels, num_classes = obj_classifier_input.number_of_train_classes)
	#pdb.set_trace()
	print("Train data and labels")
	print(obj_classifier_input.cross_features_train.shape, labels_train.shape)
	print("Test data and labels")
	print(obj_classifier_input.cross_features_test.shape, labels_test.shape)
	
	RANDOM_SEED = 42
	tf.set_random_seed(RANDOM_SEED)
	train_X = obj_classifier_input.cross_features_train#train_data
	test_X = obj_classifier_input.cross_features_test#test_data
	train_y = labels_train
	test_y = labels_test
	
	# Layer's sizes
	x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
	h_size = obj_classifier_input.dim_hidden_layer1                # Number of hidden nodes
	y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)
	print "Hidden layer dimension %d"%h_size
	# Symbols
	X = tf.placeholder("float", shape=[None, x_size])
	y = tf.placeholder("float", shape=[None, y_size])

	# Weight initializations
	w_1 = init_weights((x_size, h_size))
	w_2 = init_weights((h_size, y_size))

	# Forward propagation
	yhat    = forwardprop(X, w_1, w_2)
	predict = tf.argmax(yhat, axis=1)

	# Backward propagation
	cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
	updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

	# Run SGD
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = GPU_PERCENTAGE
	sess = tf.Session(config=config)
	init = tf.global_variables_initializer()
	sess.run(init)
	n_samp = train_X.shape[0]
	batch_size = min(4, n_samp)
	number_of_batches = int(n_samp/batch_size)
	print "Number of batches %d, batch size %d, epochs %d"%(number_of_batches, batch_size, obj_classifier_input.epochs)	
	train_accuracy = []
	test_accuracy = []
	
	for epoch in range(obj_classifier_input.epochs):
		shuffled_data_indices = np.random.permutation(n_samp)
		for batch in range(number_of_batches):
			sample = shuffled_data_indices[batch*batch_size: batch*batch_size + batch_size]
			batch_x = train_X[sample][:]
			batch_y = train_y[sample][:]
			sess.run(updates, feed_dict={X: batch_x, y: batch_y})
			train_accuracy = np.mean(np.argmax(train_y, axis=1) == sess.run(predict, feed_dict={X: train_X, y: train_y}))
			test_accuracy  = np.mean(np.argmax(test_y, axis=1) == sess.run(predict, feed_dict={X: test_X, y: test_y}))
			#pdb.set_trace()
			print("Epoch = %4d, Batch %4d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, batch, 100. * train_accuracy, 100. * test_accuracy))
	sess.close()

if 0:
	# define baseline model
	def baseline_model(dim_input, dim_hidden_layer, dim_output):
		# create model
		model = Sequential()
		model.add(Dense(dim_hidden_layer, input_dim=dim_input, init='normal', activation='relu'))
		model.add(Dense(dim_output, init='normal', activation='sigmoid'))
		# Compile model
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model


	def function_train_classifier_for_cc(obj_classifier_input):
		seed = 7
		np.random.seed(seed)
		
	obj_classifier_input.train_labels = obj_classifier_input.train_labels - 1
	labels_train = keras.utils.to_categorical(obj_classifier_input.train_labels, num_classes = obj_classifier_input.number_of_train_classes)
	labels_test = keras.utils.to_categorical(obj_classifier_input.test_labels, num_classes = obj_classifier_input.number_of_train_classes)

	NUMBER_OF_EPOCHS = 100
	BATCH_SIZE = 128
	DIM_HIDDEN_LAYER_1 = int(0.7 * obj_classifier_input.train_data.shape[1])

	estimator = KerasClassifier(build_fn=baseline_model(obj_classifier_input.train_data.shape[1], \
		 DIM_HIDDEN_LAYER_1, obj_classifier_input.number_of_train_classes), nb_epoch=NUMBER_OF_EPOCHS, batch_size=BATCH_SIZE, verbose=0)	
	kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
	results = cross_val_score(estimator,  obj_classifier_input.train_data, labels_train, cv=kfold)
	print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

	#Find accuracy
	obj_classifier_output = classifier_output()
	obj_classifier_output.predicted_labels = predicted_labels

	return obj_classifier_output
