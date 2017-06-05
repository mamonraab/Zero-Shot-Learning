import scipy.io 
import numpy as np

def datahelpers():
	dataset = {}
	mat = scipy.io.loadmat('awa-matlab-python.mat')

	fieldname = ['datasetLabels', 'vggFeatures', 'attributes', 'NUMBER_OF_CLASSES', 'defaultTestClassLabels',
					'numberOfSamplesPerTrainClass', 'classNames', 'dataset_path']

	for field in fieldname:
		dataset[field] = getData(mat, field)
	return dataset

def getData(mat, field):
	obj = mat['inputData'][field]
	if field == 'datasetLabels':
		return obj[0][0]
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

if __name__ == '__main__':
	datahelpers()