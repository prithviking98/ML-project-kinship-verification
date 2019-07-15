from sklearn import svm
from sklearn.neighbors import NearestNeighbors
import numpy as np

class ApnaSVM():
	def __init__(self, samples, W):
		self.WT = np.transpose(W)
		# print(W[0][0].dtype)
		self.N = samples.shape[1]*4//5
		self.samples = samples
		self.vectors = self.getVectors(samples, self.WT)
		self.posPairsTrain = self.getPosPairs(self.vectors[:,:self.N])
		self.negPairsTrain = self.getNegPairs(self.vectors[:,:self.N])
		self.posPairsTest = self.getPosPairs(self.vectors[:,self.N:])
		self.negPairsTest = self.getNegPairs(self.vectors[:,self.N:])
		self.svc = svm.SVC(C=0.8,gamma=0.000017)

	def getVectors(self, samples, WT):
		N = samples.shape[1]
		vectors1 = []
		vectors2 = []
		for i in range(N):
			vectors1.append(np.real(np.matmul(WT,samples[0][i])))
			vectors2.append(np.real(np.matmul(WT,samples[1][i])))

		vectors = np.array([vectors1,vectors2])
		# np.random.shuffle(vectors)

		return vectors

	def getPosPairs(self, vectors):
		N = vectors.shape[1]
		posPairs = []
		for i in range(N):
			posPairs.append(np.array([vectors[0][i]-vectors[1][i]]).flatten())

		posPairs = np.array(posPairs)
		return posPairs

	def getNegPairs(self, vectors):
		N = vectors.shape[1]
		OGRange = np.array(range(N),dtype=int)
		negIndex = np.array(range(N),dtype=int)

		np.random.shuffle(negIndex)
		while 0 in (negIndex-OGRange):
			np.random.shuffle(negIndex)

		negPairs = []
		for i in range(N):
			negPairs.append(np.array([vectors[0][i]-vectors[1][negIndex[i]]]).flatten())

		negPairs = np.array(negPairs)
		return negPairs

	def train(self):
		trainX = np.concatenate((self.posPairsTrain,
								 self.negPairsTrain))
		trainY = np.concatenate((np.ones(self.posPairsTrain.shape[0]),
								 np.zeros(self.negPairsTrain.shape[0])))
		self.svc.fit(trainX,trainY)

	def test(self):
		testX = np.concatenate((self.posPairsTest,
								self.negPairsTest))
		testY = np.concatenate((np.ones(self.posPairsTest.shape[0]),
								np.zeros(self.negPairsTest.shape[0])))
		score = self.svc.score(testX,testY)
		print(score)

class ApnaKNN():
	def __init__(self, samples, W):
		self.WT = np.transpose(W)
		# print(W[0][0].dtype)
		self.N = samples.shape[1]*4//5
		self.samples = samples
		self.vectors = self.getVectors(samples, self.WT)
		self.posPairsTrain = self.getPosPairs(self.vectors[:,:self.N])
		self.negPairsTrain = self.getNegPairs(self.vectors[:,:self.N],mult=4)
		self.posPairsTest = self.getPosPairs(self.vectors[:,self.N:])
		self.negPairsTest = self.getNegPairs(self.vectors[:,self.N:],mult=1)
		self.neigh = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)

	def getVectors(self, samples, WT):
		N = samples.shape[1]
		vectors1 = []
		vectors2 = []
		for i in range(N):
			vectors1.append(np.real(np.matmul(WT,samples[0][i])))
			vectors2.append(np.real(np.matmul(WT,samples[1][i])))

		vectors = np.array([vectors1,vectors2])
		# np.random.shuffle(vectors)

		return vectors

	def getPosPairs(self, vectors):
		N = vectors.shape[1]
		posPairs = []
		for i in range(N):
			posPairs.append(np.array([vectors[0][i]-vectors[1][i]]).flatten())

		posPairs = np.array(posPairs)
		return posPairs

	def getNegPairs(self, vectors, mult):
		N = vectors.shape[1]
		OGRange = np.array(range(N),dtype=int)

		negPairs = []
		for i in range(mult):
			negIndex = np.array(range(N),dtype=int)

			np.random.shuffle(negIndex)
			while 0 in (negIndex-OGRange):
				np.random.shuffle(negIndex)

			for i in range(N):
				negPairs.append(np.array([vectors[0][i]-vectors[1][negIndex[i]]]).flatten())

		negPairs = np.array(negPairs)
		return negPairs

	def train(self):
		trainX = np.concatenate((self.posPairsTrain,
								 self.negPairsTrain))
		self.trainY = np.concatenate((np.ones(self.posPairsTrain.shape[0]),
								 np.zeros(self.negPairsTrain.shape[0])))
		self.neigh.fit(trainX)

	def test(self):
		testX = np.concatenate((self.posPairsTest,
								self.negPairsTest))
		testY = np.concatenate((np.ones(self.posPairsTest.shape[0]),
								np.zeros(self.negPairsTest.shape[0])))
		_, kNNs = self.neigh.kneighbors(testX)
		results = self.trainY[kNNs[:,0]]
		wrongs = sum(np.abs(results-testY))
		score = (len(testX)-wrongs)*1.0/len(testX)
		print(score)

