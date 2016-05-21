import numpy as np
import pandas as pd
import scipy
import pickle


import matplotlib.pyplot as plt

from sets import Set
from math import sqrt

L = 3000

# number of iterations
N = 100000

# how many to recommend
K = 4

#
D = 3000

domain = ""
total = 20000


# Feature matrix
#################################################################################################################################################


# allCategories = {}
# Matrix = []

# catIndx = 0;


# with open('/Users/haoni/Documents/LinearCascadingBandits/specific_bus_features_res.csv/part-00000') as f:
#     for i in xrange(D):
#         Categories = f.readline().strip('\n').split(',')[1:-1]
#         Matrix.append(Categories)
#         for category in Categories:
#             if category not in allCategories:
#                 allCategories[category] = catIndx;
#                 catIndx += 1
   

# X = np.zeros((D,len(allCategories)))
# for i in xrange(len(Matrix)):
#     for category in Matrix[i]:
#         X[i][allCategories[category]] = 1

# # X = np.zeros((D,D))
# # for i in xrange(D):
# # 	X[i][i] = 1


# X = X[:L,:]



# # number of features
# m = len(allCategories)
# print "M:",m

# print allCategories.keys()
# m = D

# Generate W
#################################################################################################################################################


allUser = {}
Matrix = []

indOfUser = 0;

with open('/home/nh/Documents/LinearCascadingBandits/'+domain+'bus_users_res.csv/part-00000') as f:
    for i in xrange(D):
        Users = f.readline().strip('\n').split(',')[1:]
        Matrix.append(Users)
        for user in Users:
            if user not in allUser:
                allUser[user] = indOfUser;
                indOfUser += 1
   

# m*n m: restaurant, n: user
finalMatrix = np.zeros((D,total))


for i in xrange(len(Matrix)):
    for user in Matrix[i]:
    	if allUser[user] < total:
        	finalMatrix[i][allUser[user]] = 1

# #half user for feature and half user for feedback generating
# train = np.random.randint(finalMatrix.shape[1], size=finalMatrix.shape[1]/2)
# train = np.random.randint(total, size=total/2)

# output = open(domain+'train.pkl', 'wb')
# pickle.dump(train, output)
# output.close()


# test = [i for i in xrange(total) if i not in train]
# test = np.asarray(test)

# output = open(domain+'test.pkl', 'wb')
# pickle.dump(test, output)
# output.close()

# print "done";

pkl_file = open(domain+'train.pkl', 'rb')
train = pickle.load(pkl_file)
pkl_file.close()
pkl_file = open(domain+'test.pkl', 'rb')
test = pickle.load(pkl_file)
pkl_file.close()

print "train.shape:",train.shape
print train
print "test.shape:",test.shape
print test


M = finalMatrix[:,train].T
print M.shape


print np.sum(finalMatrix[0])
print np.sum(finalMatrix[1])

finalMatrix = finalMatrix[:L,test]

print np.sum(finalMatrix[0])
print np.sum(finalMatrix[1])


sampleBase = np.copy(finalMatrix) 

print "!!"
print len(allUser)
print "!!"

print np.sum(finalMatrix[0])
print np.sum(finalMatrix[1])
print np.sum(finalMatrix[2])


optimalSet = Set([])


for n in xrange(K):
    curMax = 0
    curInd = -1
    for i in xrange(len(finalMatrix)):
        if i not in optimalSet:
            curSum = np.sum(finalMatrix[i])
            if (curSum > curMax) :
                curMax = curSum
                curInd = i
                print i,curSum
    optimalSet.add(curInd)
    for i in xrange(len(finalMatrix)):
        if i not in optimalSet:
            finalMatrix[i] = finalMatrix[i]-finalMatrix[curInd];
            finalMatrix[i][finalMatrix[i]<0] = 0

optimal = np.array(list(optimalSet))
print optimal



#################################################################################################################################################
#################################################################################################################################################


# l_factors = [10,20,40]
l_factors = [20]
e_color = {10:'r', 20:'b', 40:'g'}
plt.figure()


for f in l_factors:

	# # Vt = pca(M)
	# U, s, Vt = np.linalg.svd(M, full_matrices=False)
	# # S = np.diag(s)
	# # V = np.dot(S, V)
	# Vt = Vt[:f,:]
	# V = Vt.T
	# X = V
	# print "V",X.shape
	# output = open('X_20.pkl', 'wb')
	# pickle.dump(X, output)
	# output.close()

	pkl_file = open('X_20.pkl', 'rb')
	X = pickle.load(pkl_file)
	pkl_file.close()

	m = len(X[0])

	print X
	print X.shape
	X = X[:L,:]

	# intercept
	X_sub = X
	X = np.ones((X_sub.shape[0],X_sub.shape[1]+1))
	X[:,:-1] = X_sub
	m = m + 1


	res = []

	for iteration in xrange(10):

		W = np.zeros((N+1,L))
		for i in xrange(1,N+1):
			# col = np.random.randint(len(allUser))
			col = np.random.randint(sampleBase.shape[1])
			# col = np.random.randint(25)
			W[i] = sampleBase[:,col]

		# print W


		lamda = 1
		sigma = 1

		Sigma = []
		Beta = []
		for i in xrange(K):
			Sigma.append(lamda*np.identity(m))
			Beta.append(np.zeros((m,1))) 

		R = np.zeros((N+1,1))

		for t in xrange(1,N+1):
			print iteration,t

			topKInd = []
			for i in xrange(K):
			
				S = np.linalg.inv(Sigma[i])
				thetaHat = sigma**(-2)*np.dot(np.linalg.inv(Sigma[i]),Beta[i])
				# print Beta
				# print Sigma.shape
				thetaHat = thetaHat.T[0]
				# print thetaHat

				# print S
				# print thetaHat.shape
				theta = np.random.multivariate_normal(thetaHat, S)
				# print theta.shape

				wHat = np.dot(X,theta)
			# print wHat.shape
			# print np.ones((X.shape[0],)).shape

				sortedInd = np.argsort(wHat,axis=None)[::-1]
				for ind in sortedInd:
					if ind not in topKInd:
						topKInd.append(ind)
						break

		

			wt = W[t]
			wtK = wt[topKInd]

			# print wtK

			reward = 1 if np.any(wtK>0) else 0
			optimalreward = 1 if np.any(wt[optimal]) else 0

			R[t] =  optimalreward - reward + R[t-1] 


			C = K + 1
			for i in range(len(wtK)):
				if wtK[i] > 0:
					# print i+1
					C = i+1
					break
			# print C
			for k in xrange(1,min(C+1,K+1)):
				e = topKInd[k-1]
				Xe = np.array([X[e]])
				Sigma[k-1] += sigma**(-2)*np.dot(Xe.T,Xe)

				if C == k:
					Beta[k-1] += Xe.T

		R = R.flatten()

		print R.shape
		res.append(R[1:])

	res = np.array(res)
	np.savetxt('L'+str(L)+'_RankTS_4_20.csv', res, fmt='%.1e', delimiter=',')

	std = [np.std(res[:,i]) for i in xrange(res.shape[1])]
	SEM = [i/sqrt(10) for i in std]
	mean = [np.mean(res[:,i]) for i in xrange(res.shape[1])]

	x = range(len(mean))
	y = mean

	x = x[::1000]
	y = y[::1000]
	SEM = SEM[::1000]

	print y
	plt.errorbar(x, y, yerr=SEM, ecolor=e_color[f], color=e_color[f], label=str(f))

plt.legend(loc='upper right')
plt.title("")
plt.show()

# ts = pd.Series(R[1:], index=xrange(1,N+1))
# ts.plot()
# plt.show()
