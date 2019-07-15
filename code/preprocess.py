import os
from skimage import data, exposure, feature
from numpy import array as np_array
import numpy as np

N_BINS=16

def prepareImage(fileName):
	return exposure.equalize_hist(data.imread(fileName,as_gray=True))

def getPreparedImages(dir):
	fileNames = [x for x in os.listdir(dir) if x.endswith('.jpg')]
	fileNames.sort()
	parent = []
	child = []

	for i in range(0,len(fileNames),2):
		parent.append(prepareImage(os.path.join(dir,fileNames[i])))
		child.append(prepareImage(os.path.join(dir,fileNames[i+1])))

	return np_array([np_array(parent), np_array(child)])

def flattenedImages(images):
	return np_array([i.flatten() for i in images])

def extractLBP(images):
	LBP_BLOCK_MAX_X = 4
	LBP_BLOCK_MAX_Y = 4
	l,b = images[0].shape
	BLOCK_L = l//LBP_BLOCK_MAX_X
	BLOCK_B = b//LBP_BLOCK_MAX_Y
	featureVecs = []

	for image in images:
		vector = []
		_x = 0
		for x in range(LBP_BLOCK_MAX_X):
			x_ = _x + BLOCK_L
			_y = 0
			for y in range(LBP_BLOCK_MAX_Y):
				y_ = _y + BLOCK_B
				newHist, _ = exposure.histogram(image[_x:x_,_y:y_],nbins=N_BINS)
				vector.extend(newHist)
				_y = y_
			_x = x_

		featureVecs.append(vector)

	featureVecs = np_array(featureVecs)
	return featureVecs

def extractLE(images):
	return flattenedImages(images)

def extractSIFT(images):
	return flattenedImages(images)

def extractTPLBP(images):
	return flattenedImages(images)

def extractHOG(images):
	featureVecs = []

	for image in images:
		featureVecs.append(feature.hog(image,
									   block_norm='L1',
									   feature_vector=True))

	featureVecs = np_array(featureVecs)
	return featureVecs

def extractDAISY(images):
	featureVecs = []

	for image in images:
		featureVecs.append(feature.daisy(image,
										 step=8,
										 radius=8,
										 rings=3).flatten())

	featureVecs = np_array(featureVecs)
	return featureVecs

def getFlats(dir, whichFeatures):
	allImages = getPreparedImages(dir)

	getFeatures = None
	if whichFeatures == 0:
		getFeatures = flattenedImages
	elif whichFeatures == 1:
		getFeatures = extractLBP
	elif whichFeatures == 2:
		getFeatures = extractLE
	elif whichFeatures == 3:
		getFeatures = extractSIFT
	elif whichFeatures == 4:
		getFeatures = extractTPLBP
	elif whichFeatures == 5:
		getFeatures = extractHOG
	elif whichFeatures == 6:
		getFeatures = extractDAISY

	allVectors = np_array([getFeatures(allImages[0]),
						   getFeatures(allImages[1])])	

	print(allVectors.shape)
	return allVectors


def getPreparedImagesCNN(dir):
	fileNames = [x for x in os.listdir(dir) if x.endswith('.jpg')]
	fileNames.sort()
	parent = []
	child = []

	for i in range(0,len(fileNames),2):
		parent.append(data.imread(os.path.join(dir,fileNames[i])))
		child.append(data.imread(os.path.join(dir,fileNames[i+1])))

	return np_array([np_array(parent), np_array(child)])

def getCNNSamples(dir, mult=5):
	allVectors = getPreparedImagesCNN(dir)

	N = allVectors.shape[1]

	posPairs = []
	for i in range(N):
		posPairs.append(np.concatenate((allVectors[0][i],allVectors[1][i]),axis=2))

	OGRange = np_array(range(N),dtype=int)
	negPairs = []
	for i in range(mult):
		negIndex = np_array(range(N),dtype=int)

		np.random.shuffle(negIndex)
		while 0 in (negIndex-OGRange):
			np.random.shuffle(negIndex)

		for i in range(N):
			negPairs.append(np.concatenate((allVectors[0][i],allVectors[1][negIndex[i]]),axis=2))

	return np_array(posPairs),np_array(negPairs)

def getRelationships(dir):
	fileNames = [x for x in os.listdir(dir) if os.path.isdir(os.path.join(dir,x))]
	fileNames.sort()
	parents = []
	childrens = []

	for relationship in fileNames:
		newParent, newChild = getRelationshipImages(os.path.join(dir,relationship))
		parents.append(newParent)
		childrens.append(newChild)

	return np_array([np_array(parents), np_array(childrens)])

if __name__ == '__main__':
	getFlats("./data/KinFaceW-I/images/father-dau")