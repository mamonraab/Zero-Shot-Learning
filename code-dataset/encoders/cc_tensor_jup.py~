
# coding: utf-8

# In[1]:


#
#
#..........cc_tensor_jup.py..........
#
#
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from get_data_for_cc import function_get_training_data_cc, input_cc, output_cc, function_normalise_data
from get_data_for_cc import input_data, function_get_input_data, classifier_data
from train_tensorflow_cc import train_tf_cc_input, train_tf_cc_output, function_train_tensorflow_cc, function_train_keras_cc
from train_tensorflow_cc import classifier_output, function_train_classifier_for_cc
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import scipy.io
import matplotlib
from keras.datasets import mnist
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras.callbacks import TensorBoard
import numpy as np
import pdb
print "*****************************************************************************************************************************************"
#written from jup to noraml


# In[2]:


EPOCHS = 10
EPOCHS_CC = 1
BATCH_SIZE = 128
BATCH_SIZE_CC = 128
TR_TS_VA_SPLIT = np.array([0.6, 0.2, 0.2])
MIN_NUMBER_OF_SAMPLES_ACROSS_CLASSES = 50
NOISE_FACTOR = 0.1
INCREASE_FACTOR = 100
dataset_list = ['sample', 'apy', 'mnist']
DATASET_INDEX = 2
system_list = ['desktop', 'laptop']
SYSTEM_INDEX = 1
#DATA_SAVE_PATH = '/home/SharedData/omkar/data/'
DATA_SAVE_PATH = '../zsl/datasets/'
USE_ENCODER_FEATURES = 0
#Prepare encoder model...................
if DATASET_INDEX == 0:
	dimension_hidden_layer1_coder = 6
	REDUCED_DIMENSION_VISUAL_FEATURE = 0
	min_num_samples_per_class = 10
elif DATASET_INDEX == 1:
	dimension_hidden_layer1_coder = 500
	REDUCED_DIMENSION_VISUAL_FEATURE = 000
	min_num_samples_per_class = 51
else:
	dimension_hidden_layer1_coder = 50
	REDUCED_DIMENSION_VISUAL_FEATURE = 
	min_num_samples_per_class = 
	
#Load input data..................................
obj_input_data = input_data()
obj_input_data.dataset_name = dataset_list[DATASET_INDEX]
obj_input_data.system_type = system_list[SYSTEM_INDEX]

obj_input_data = function_get_input_data(obj_input_data)
#pdb.set_trace()
visual_features_dataset_ori = obj_input_data.visual_features_dataset
train_class_labels = obj_input_data.train_class_labels
test_class_labels = obj_input_data.test_class_labels
attributes = obj_input_data.attributes
dataset_labels = obj_input_data.dataset_labels

#PCA
if REDUCED_DIMENSION_VISUAL_FEATURE:
	visual_features_dataset_norm = StandardScaler().fit_transform(visual_features_dataset_ori)
	visual_features_dataset = PCA(n_components = REDUCED_DIMENSION_VISUAL_FEATURE).fit_transform(visual_features_dataset_norm)
	visual_features_dataset = function_normalise_data(visual_features_dataset)
else:
	visual_features_dataset = function_normalise_data(visual_features_dataset_ori)

number_of_train_classes = np.size(train_class_labels)
number_of_test_classes = np.size(test_class_labels)
dimension_visual_data = visual_features_dataset.shape[1]
number_of_samples_dataset = visual_features_dataset.shape[0]
dimension_attributes = attributes.shape[1]
number_of_classes = attributes.shape[0]
print "Dataset visual features shape is: %d X %d" % visual_features_dataset.shape
print "Dimension of visual data: %d" %dimension_visual_data
print "Number of dataset samples: %d" %number_of_samples_dataset
print "Dimension of attributes: %d" %dimension_attributes
print "Number of classes: %d" %number_of_classes
print "Train classes are"
print(train_class_labels)
print "Test classes are"
print "Noise factor %f"%NOISE_FACTOR 
print "Data augmentation factor %d"%INCREASE_FACTOR 
print(test_class_labels)
#scipy.io.savemat('data/cnn_features.mat', \
#                dict(cnn_features = visual_features_dataset))
print("****Not Saved cnn features")
print "Dimension Coder Hidden1 %d" %(dimension_hidden_layer1_coder)
#pdb.set_trace()
#......................cc.......................
number_of_cc = number_of_train_classes * number_of_train_classes - number_of_train_classes
print "Number of c coders %d "%number_of_cc
cross_coders_train_data_input = []
cross_coders_train_data_output = []

# In[ ]:

#Get mean feature vector for each class
mean_feature_mat = np.empty((0, dimension_visual_data), float)
number_of_samples_per_class_train = []
number_of_samples_per_class_test = []
number_of_samples_per_class_valid = []

#pdb.set_trace()
obj_classifier = classifier_data()
cnt = 0
for classI in train_class_labels:
	indices = np.flatnonzero(dataset_labels == classI)
	classI_features = visual_features_dataset[indices.astype(int), :]
	mean_feature = classI_features.mean(0)
	mean_feature_mat = np.append(mean_feature_mat, mean_feature.reshape(1, dimension_visual_data), axis = 0)	
	number_of_samples_per_class_train.append(int(TR_TS_VA_SPLIT[0] * np.size(indices))) 
	number_of_samples_per_class_test.append(int(TR_TS_VA_SPLIT[2] * np.size(indices))) 
	number_of_samples_per_class_valid.append(int(TR_TS_VA_SPLIT[1] * np.size(indices)))
	start_vl = number_of_samples_per_class_train[-1]
	end_vl = start_vl + number_of_samples_per_class_valid[-1]
	start_ts = end_vl
	end_ts = start_ts + number_of_samples_per_class_test[-1]
	if cnt == 0:
		cnt = 1	
		obj_classifier.train_data = classI_features[:number_of_samples_per_class_train[-1], :] 
		obj_classifier.valid_data = classI_features[start_vl:end_vl, :] 
		obj_classifier.test_data = classI_features[start_ts:end_ts, :] 
		obj_classifier.train_labels = np.full((1, number_of_samples_per_class_train[-1]), classI, dtype=int) 
		obj_classifier.valid_labels = np.full((1, number_of_samples_per_class_valid[-1]), classI, dtype=int) 
		obj_classifier.test_labels = np.full((1, number_of_samples_per_class_test[-1]), classI, dtype=int) 
	else:	
		obj_classifier.train_data = np.vstack((obj_classifier.train_data, classI_features[:number_of_samples_per_class_train[-1], :])) 
		obj_classifier.valid_data = np.vstack((obj_classifier.valid_data, classI_features[start_vl:end_vl, :])) 
		obj_classifier.test_data = np.vstack((obj_classifier.test_data, classI_features[start_ts:end_ts, :])) 
		obj_classifier.train_labels = np.hstack((obj_classifier.train_labels, np.full((1, number_of_samples_per_class_train[-1]), classI, dtype=int))) 
		obj_classifier.valid_labels = np.hstack((obj_classifier.valid_labels, np.full((1, number_of_samples_per_class_valid[-1]), classI, dtype=int))) 
		obj_classifier.test_labels = np.hstack((obj_classifier.test_labels, np.full((1, number_of_samples_per_class_test[-1]), classI, dtype=int))) 

file_name = DATA_SAVE_PATH + dataset_list[DATASET_INDEX] + '_train_labels_' + str(dimension_hidden_layer1_coder) + '_'+ str(REDUCED_DIMENSION_VISUAL_FEATURE)
scipy.io.savemat(file_name, dict(train_labels = obj_classifier.train_labels))

file_name = DATA_SAVE_PATH + dataset_list[DATASET_INDEX] + '_test_labels_' + str(dimension_hidden_layer1_coder) + '_'+ str(REDUCED_DIMENSION_VISUAL_FEATURE)
scipy.io.savemat(file_name, dict(test_labels = obj_classifier.test_labels))

file_name = DATA_SAVE_PATH + dataset_list[DATASET_INDEX] + '_valid_labels_' + str(dimension_hidden_layer1_coder) + '_'+ str(REDUCED_DIMENSION_VISUAL_FEATURE)
scipy.io.savemat(file_name, dict(valid_labels = obj_classifier.valid_labels))
# In[ ]:
#pdb.set_trace()
cc_start = time.time() 
cnt = 0
for classI in train_class_labels:
	cross_features_classI_train = []
	cross_features_classI_labels_train = []
	cross_features_classI_test = []
	cross_features_classI_labels_test = []
	number_of_classI_samples_train = 0
	number_of_classI_samples_test = 0
	for classJ in train_class_labels:
		print "**************************************"
			#cc1..............................................
			
 		cc1_start = time.time()
			#Get data for training cc.........................
		obj_input_cc = input_cc()
		obj_input_cc.classI = classI
		obj_input_cc.classJ = classJ
		obj_input_cc.visual_features = visual_features_dataset
		obj_input_cc.train_valid_split = TR_TS_VA_SPLIT
		obj_input_cc.dataset_labels = dataset_labels
		obj_input_cc.min_num_samples_per_class = min_num_samples_per_class
		
		obj_cc1_train_valid_data = function_get_training_data_cc(obj_input_cc)
		cc1_input_train_perm = obj_cc1_train_valid_data.input_train_perm
		cc1_input_train_perm = np.tile(cc1_input_train_perm, (INCREASE_FACTOR, 1))
		cc1_input_train_perm = cc1_input_train_perm + NOISE_FACTOR * np.random.normal(0, 1, cc1_input_train_perm.shape)
		cc1_input_train_perm = function_normalise_data(cc1_input_train_perm)
		
		if classI == classJ:
			cc1_output_train_perm = cc1_input_train_perm
		else:	
			cc1_output_train_perm  = obj_cc1_train_valid_data.output_train_perm
			cc1_output_train_perm = np.tile(cc1_output_train_perm, (INCREASE_FACTOR, 1))
			cc1_output_train_perm = cc1_output_train_perm + NOISE_FACTOR * np.random.normal(0, 1, cc1_output_train_perm.shape)
			cc1_output_train_perm = function_normalise_data(cc1_output_train_perm)
	
		#Train tensorflow cc.....................................
		print "Training cc over %d samples"%(cc1_input_train_perm.shape[0])
		#pdb.set_trace()
		obj_train_tf_cc_input = train_tf_cc_input()
		obj_train_tf_cc_input.classI = classI
		obj_train_tf_cc_input.classJ = classJ
		obj_train_tf_cc_input.dataset_name = dataset_list[DATASET_INDEX]
		obj_train_tf_cc_input.dim_feature = visual_features_dataset.shape[1]
		obj_train_tf_cc_input.cc1_input_train_perm = cc1_input_train_perm
		obj_train_tf_cc_input.cc1_output_train_perm = cc1_output_train_perm
		obj_train_tf_cc_input.cc1_input_valid_perm = function_normalise_data(obj_cc1_train_valid_data.input_valid_perm)
		obj_train_tf_cc_input.cc1_output_valid_perm = function_normalise_data(obj_cc1_train_valid_data.output_valid_perm)
		obj_train_tf_cc_input.obj_classifier = obj_classifier
		obj_train_tf_cc_input.dimension_hidden_layer1 = dimension_hidden_layer1_coder
		obj_train_tf_cc_input.EPOCHS_CC = EPOCHS_CC
		obj_train_tf_cc_output = function_train_tensorflow_cc(obj_train_tf_cc_input)
		#obj_train_tf_cc_output = function_train_keras_cc(obj_train_tf_cc_input)
			
		#pdb.set_trace()			
		#COncatenate features
		if cnt == 0:
			if USE_ENCODER_FEATURES:
				print "Using encoded features"
				cross_features_train = function_normalise_data(obj_train_tf_cc_output.encoded_data_train_cc1)
				cross_features_valid = function_normalise_data(obj_train_tf_cc_output.encoded_data_valid_cc1)
				cross_features_test = function_normalise_data(obj_train_tf_cc_output.encoded_data_test_cc1)
			else:
				print "Using decoded features"
				cross_features_train = obj_train_tf_cc_output.decoded_data_train_cc1
				cross_features_valid = obj_train_tf_cc_output.decoded_data_valid_cc1
				cross_features_test = obj_train_tf_cc_output.decoded_data_test_cc1
			cnt  = 1
		else:
			if USE_ENCODER_FEATURES:
				print "Using enco:ded features"
				cross_features_train = np.hstack((cross_features_train, function_normalise_data(obj_train_tf_cc_output.encoded_data_train_cc1)))
				cross_features_valid = np.hstack((cross_features_valid, function_normalise_data(obj_train_tf_cc_output.encoded_data_valid_cc1)))
				cross_features_test = np.hstack((cross_features_test, function_normalise_data(obj_train_tf_cc_output.encoded_data_test_cc1)))
			else:
				print "Using decoded features"
				cross_features_train = np.hstack((cross_features_train, obj_train_tf_cc_output.decoded_data_train_cc1))
				cross_features_valid = np.hstack((cross_features_valid, obj_train_tf_cc_output.decoded_data_valid_cc1))
				cross_features_test = np.hstack((cross_features_test, obj_train_tf_cc_output.decoded_data_test_cc1))
		cc_end = time.time() 
		print "Processing time for cc %f"%((cc_end - cc_start))

file_name = DATA_SAVE_PATH +  dataset_list[DATASET_INDEX] + '_'+ str(dimension_hidden_layer1_coder) + '_'+ str(REDUCED_DIMENSION_VISUAL_FEATURE) + 							'_cc1_cross_feat_ALL_CLASS_tr_' + str(classI) + '_' + '.mat'		
print"Saving cross features *train data* for classes 1 to %d ...%s"%(classI, file_name)
scipy.io.savemat(file_name, dict(cross_feautures_tr = cross_features_train))
	
file_name = DATA_SAVE_PATH + dataset_list[DATASET_INDEX] + '_'+ str(dimension_hidden_layer1_coder) + '_'+ str(REDUCED_DIMENSION_VISUAL_FEATURE) + 							'_cc1_cross_feat_ALL_CLASS_val_' + str(classI) + '_' + '.mat'		
print"Saving cross features *valid data* for classes 1 to %d ...%s"%(classI, file_name)
scipy.io.savemat(file_name, dict(cross_feautures_val = cross_features_valid))

file_name = DATA_SAVE_PATH + dataset_list[DATASET_INDEX] + '_'+ str(dimension_hidden_layer1_coder) + '_'+ str(REDUCED_DIMENSION_VISUAL_FEATURE) + 							'_cc1_cross_feat_ALL_CLASS_ts_' + str(classI) + '_' + '.mat'		
print"Saving cross features *test data* for classes 1 to %d ...%s"%(classI, file_name)
scipy.io.savemat(file_name, dict(cross_feautures_ts = cross_features_test))
	

# In[ ]:
cross_features_tr_val_ts = np.vstack((cross_features_train, cross_features_valid))
cross_features_tr_val_ts = np.vstack((cross_features_tr_val_ts, cross_features_test))
cross_features_tr_val_ts_norm = StandardScaler().fit_transform(cross_features_tr_val_ts)
cross_features_tr_val_ts_pcaed = PCA(n_components = REDUCED_DIMENSION_VISUAL_FEATURE).fit_transform(cross_features_tr_val_ts_norm)

#pdb.set_trace()
obj_classifier.epochs = 1
obj_classifier.number_of_train_classes = number_of_train_classes
obj_classifier.dim_hidden_layer1 = int(0.6 * cross_features_tr_val_ts_pcaed.shape[1])
obj_classifier.cross_features_train = cross_features_tr_val_ts_pcaed[cross_features_train.shape[0], :]
obj_classifier.cross_features_valid = cross_features_tr_val_ts_pcaed[cross_features_valid.shape[0], :]
obj_classifier.cross_features_test = cross_features_tr_val_ts_pcaed[cross_features_test.shape[0], :]
obj_classifier_output = function_train_classifier_for_cc(obj_classifier)

print "Dimension Coder Hidden1 %d, Classifier %d" %(dimension_hidden_layer1_coder, dimension_hidden_layer1_classifier)
#pdb.set_trace()			
#written from jup to noraml


# In[ ]:


if 0:
		  scipy.io.savemat('data/encoded', dict(encoded =obj_train_tf_cc_output.encoded_data_train_cc1))
		  scipy.io.savemat('data/input', dict(input = obj_train_tf_cc_input.obj_classifier.train_data))
		  scipy.io.savemat('data/decoded', dict(decoded = obj_train_tf_cc_output.decoded_data_train_cc1))
 

