import preprocess as pp
import nrml
from classifier import ApnaSVM, ApnaKNN
import sys
import numpy as np

relationship = sys.argv[2]
featureSet = int(sys.argv[1])
k = 5
T = 5
eps = 0.00001

allImages = pp.getFlats(relationship, featureSet)

print("NRML")
W = nrml.trainNRML(allImages, k, T, eps)
print("\t...done")

knn = ApnaKNN(allImages, W)
svc = ApnaSVM(allImages, W)
knn.train()
svc.train()
knn.test()
svc.test()