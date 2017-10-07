import scipy.io 
import numpy as np
import random

def datahelpersAWA():
	'''
	AwA features provided by the matlab file extracted using the VGG subnet
	'''
	dataset = {}
	mat1 = scipy.io.loadmat('../datasets/awa-matlab-python.mat')
	mat2 = scipy.io.loadmat('../datasets/AWA_inform_release.mat')
	mat3 = scipy.io.loadmat('../datasets/AwA_googlenet.mat')  # for X

	fieldname = ['datasetLabels', 'vggFeatures', 'attributes', 'NUMBER_OF_CLASSES', 'defaultTestClassLabels',
					'numberOfSamplesPerTrainClass', 'classNames', 'dataset_path']

	for field in fieldname:
		dataset[field] = getData(mat1, field)
	lst = [label for label in range(1,dataset['NUMBER_OF_CLASSES']+1) 
					if label not in dataset['defaultTestClassLabels']]
	#dataset['defaultCVClassLabels'] = np.array([5, 9, 11, 14, 19, 29, 45, 50])
	dataset['defaultCVClassLabels'] = np.sort(np.array(random.sample(lst, 10)))
	dataset['attr85'] = mat2['attr2']
	dataset['googlenetFeatures'] = mat3['X']
	return dataset

def datahelpersAPY():
	'''
	ApY features provided by the matlab file extracted using the VGG subnet
	'''
	dataset = {}
	mat = scipy.io.loadmat('../datasets/apy-matlab-python.mat')

	fieldname = ['datasetLabels', 'vggFeatures', 'attributes', 'NUMBER_OF_CLASSES', 'defaultTestClassLabels',
					'numberOfSamplesPerTrainClass', 'classNames', 'dataset_path']

	for field in fieldname:
		dataset[field] = getData(mat, field)
	lst = [label for label in range(1,dataset['NUMBER_OF_CLASSES']+1) 
					if label not in dataset['defaultTestClassLabels']]
	dataset['defaultCVClassLabels'] = np.sort(np.array(random.sample(lst, 8)))
	return dataset

def datahelpersSUN():
	'''
	SUN scene attribute feature loaded from the matlab file using GoogleNet feature
	'''
	dataset = {}
	mat = scipy.io.loadmat('../datasets/SUN_googlenet.mat')
	fieldname = ['attr', 'classes', 'Y', 'X']

	for field in fieldname:
		dataset[field] = getSUNdata(mat, field)
	n_class = dataset['classes'].shape[0]
	dataset['defaultTestClassLabels'] = random.sample(range(1, n_class+1), 10)  # 10 unseen class
	dataset['defaultCVClassLabels'] = random.sample([label for label in range(1, n_class+1)
										if label not in dataset['defaultTestClassLabels']], 20)
	return dataset

def getSUNdata(mat, field):
	'''
	Used internally by datahelpersSUN() function
	'''
	obj = mat[field]
	if field == 'attr':
		return obj
	elif field == 'X':
		return obj
	elif field == 'Y':
		shape = obj.shape[0]
		y = np.reshape(obj, shape)
		return y
	else:
		shape = obj.shape[0]
		classes = np.reshape(obj, shape)
		return classes

def datahelpersCUB():
	'''
	CUB birds attribute feature loaded from the matlab file using GoogleNet feature
	'''
	dataset = {}
	mat1 = scipy.io.loadmat('../datasets/CUB_googlenet.mat')
	mat2 = scipy.io.loadmat('../datasets/CUB_inform_release.mat')

	dataset['X'] = mat1['X']
	dataset['datasetLabels'] = mat2['Y'][0]
	dataset['attr312'] = mat2['attr2']
	return dataset

def getData(mat, field):
	'''
	Used internally by the datahelpersAWA as well as datahelpersAPY function
	'''
	obj = mat['inputData'][field]
	if field == 'datasetLabels':
		labels = obj[0][0]
		labels = labels.reshape([1, labels.shape[0]])[0]
		return labels
	elif field == 'vggFeatures':
		return np.transpose(obj[0][0])
	elif field == 'attributes':
		return np.transpose(obj[0][0])
	elif field == 'NUMBER_OF_CLASSES':
		return obj[0][0][0][0]
	elif field == 'defaultTestClassLabels':
		return obj[0][0][0]
	elif field == 'numberOfSamplesPerTrainClass':
		return obj[0][0][0][0]
	elif field == 'classNames':
		return obj[0][0]
	else:
		return None

def getAttributes():
	'''
	To get the attribute vector from the file specified below downloaded from
	the AWA official website
	'''
	attributes = list([])
	path = 'predicate-matrix-continuous.txt'
	with open(path) as file:
		for line in file:
			lst = line.split()
			attributes.append(lst)
	attributes = np.array(attributes, dtype=np.float32)
	return attributes

def datahelpersAWAfused():
	dataset = {}
	mat1 = scipy.io.loadmat('../datasets/AwA_googlenet.mat')  # for X
	mat2 = scipy.io.loadmat('../datasets/AWA_inform_release.mat')   # for attr2(continuous attributes)
	dataset['X'] = mat1['X']
	dataset['attr2'] = mat2['attr2']
	return dataset

if __name__ == '__main__':
	datahelpers()