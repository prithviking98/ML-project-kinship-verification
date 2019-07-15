import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle

class FastMatMul():
	def __init__(self, allVectors):
		self.allVectors = allVectors
		N = allVectors.shape[1]
		m = allVectors.shape[2]
		# self.matrix = np.full((N,N,m,m), -1)

	def getMatMulFast(self, i, j, allVectors):
		if matrix[i][j][0][0] == -1:
			diff = allVectors[0][i] - allVectors[1][j]
			diffT = np.transpose(diff)
			matrix[i][j] = np.matmul(diff,diffT)

		return matrix[i][j]

	def getMatMulEasy(self, i, j, allVectors):
		diff = allVectors[0][i] - allVectors[1][j]
		diffT = np.transpose(diff)
		return np.matmul(diff,diffT)

	def getMatMul(self, i, j, allVectors):
		return self.getMatMulEasy(i, j, allVectors)

def distanceMetric(x1, x2, WT):
	diff = x1 - x2
	WTdiff = np.matmul(WT,diff)
	return (np.dot(WTdiff,WTdiff))**0.5

def getkNNs(vectors, k, W):
	WT = np.transpose(W)
	metric = lambda x1,x2: distanceMetric(x1,x2,WT)
	neigh = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1)
	neigh.fit(vectors)
	_, kNNs = neigh.kneighbors(vectors)

	return kNNs[:,1:]

def getAllkNNs(allVectors, k, W):
	allkNNs = np.array([getkNNs(allVectors[0],k,W),
						getkNNs(allVectors[1],k,W)])

	return allkNNs

def getNewH(allVectors, allkNNs, fMM):
	N = allVectors.shape[1]
	m = allVectors.shape[2]
	k = allkNNs.shape[1]
	
	H1 = np.zeros((m,m))
	for i in range(N):
		for t1 in allkNNs[0][i]:
			H1 += fMM.getMatMul(i, t1, allVectors)
	H1 /= (N*k)

	H2 = np.zeros((m,m))
	for i in range(N):
		for t2 in allkNNs[1][i]:
			H2 += fMM.getMatMul(t2, i, allVectors)
	H2 /= (N*k)

	H3 = np.zeros((m,m))
	for i in range(N):
		H3 += fMM.getMatMul(i, i, allVectors)
	H3 /= N

	return H1, H2, H3

def getNewW(H1, H2, H3, l):
	w, v = np.linalg.eig(H1+H2-H3)
	# print(w)
	# print(v)
	# w = np.real(w)
	# v = np.real(v)
	# exit()
	# print(v.shape,H1.shape,H2.shape,H3.shape)
	# v[:][(-w).argsort()]

	return v[:,(-w).argsort()][:,:l]

def trainNRML(allVectors, k, T, eps, l=None):
	N = allVectors.shape[1]
	m = allVectors.shape[2]
	fMM = FastMatMul(allVectors)
	if l == None:
		l = int(0.8*m)

	# print("init")
	allkNNs = getAllkNNs(allVectors, k, np.eye(m))
	Wr_1 = None
	Wr = None

	file = ["alpha.W.pickle", "beta.W.pickle"]

	for r in range(T):
		print(r)
		Wr_1 = Wr
		H1, H2, H3 = getNewH(allVectors, allkNNs, fMM)
		Wr = getNewW(H1, H2, H3, l)

		pickle_out = open(file[r%2], 'wb')
		pickle.dump(Wr, pickle_out)
		pickle_out.close()

		if r > 1:
			if np.allclose(Wr,Wr_1,atol=eps):
				break

	return Wr

