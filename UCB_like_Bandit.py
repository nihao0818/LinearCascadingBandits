import numpy as np
import pandas as pd
import math
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
from sets import Set
from math import sqrt
import pickle




class UCBLikeBandit:
	"""Implement a series of UCB-like Bandit Algorithm"""
	def UCB1(self,w_hat,t,T):
		"""Compute UCB based on UCB1 algorithm

		parameter:
		w_hat: current mean weight of each bandit over the time
		t: current time
		T: current number of observations of each bandit
		"""

		return  w_hat + float(math.sqrt(1.5*math.log(t)))/np.sqrt(T) 

	def reg_probOfNoneClick(self,Top_K,C,t):
		"""Compute regret 

		parameter:
		w_hat: current mean weight of each bandit
		K: recommend K bandits each time
		Top_K: currently recommended K bandits
		"""

		global optimalSet

		optimalIndex = np.array(list(optimalSet))


		# compute the optimal reward
		# optimal = 1 if np.sum(C[t][:self.K]) > 0 else 0
		optimal = 1 if np.sum(C[t][optimalIndex]) > 0 else 0
		suboptimal = 1 if np.sum(C[t][Top_K]) > 0 else 0

		return optimal - suboptimal



	def observationGenerator(self):
		"""Generate a distribution of real reward  
		"""
		C_first_K=bernoulli.rvs(self.P, size=(self.N+2,self.K))
		C_rest=bernoulli.rvs(self.P-self.delta, size=(self.N+2,self.L-self.K))
		C = np.hstack((C_first_K,C_rest))
		# C = bernoulli.rvs(0.2, size=(N+2,L))
		C = np.asfarray(C, dtype='float')
		return C


	UCBs = {"UCB1":UCB1}
	REGs = {"reg_probOfNoneClick":reg_probOfNoneClick}

	def __init__(self,L=16,N=100000,K=4,P=0.2,delta=0.15,ucb_method="UCB1",reg_method="reg_probOfNoneClick"):
		"""Init the whole algorithm

		parameter:
		L: number of bandits
		N: total experiment time/iteration
		K: recommend K bandits each time
		P: first K bandits in the ground set follows a Bernouli distribution with P
		delta: the rest K bandits in the ground set follows a Bernouli distribution with delta
		ucb_method: which UCB algorithm to use
		reg_method: what kind of regret to use
		"""
		self.L = L 
		self.N = N 
		self.K = K 
		self.P = P 
		self.delta = delta 
		self.ucb_method = UCBLikeBandit.UCBs[ucb_method]
		self.reg_method = UCBLikeBandit.REGs[reg_method]

		#To store the regret every time
		self.R = np.zeros(self.N+2)
		self.R = np.asfarray(self.R, dtype='float')




	def fit(self,C):
		"""The main routine of the algorithm
		"""

		# C = self.observationGenerator()

		#T is the current number of observations of each bandit
		T = np.ones(self.L)
		T = np.asfarray(T, dtype='float')

		#w_hat is current mean weight of each bandit over the time
		# w_first_K=bernoulli.rvs(self.P, size=self.K)
		# w_rest=bernoulli.rvs(self.P-self.delta, size=self.L-self.K)
		# w_hat = np.hstack((w_first_K,w_rest))
		# w_hat = np.asfarray(w_hat, dtype='float')
		w_hat = np.zeros((self.L,))

		#for each time t
		for t in xrange(2,self.N+2):

			#compute UCB		
			U_t = self.ucb_method(self,w_hat,t-1,T)

			#sort and get top k values
			Top_K = np.argpartition(U_t, -self.K)[-self.K:]
			Top_K = Top_K[np.argsort(U_t[Top_K])]
			Top_K = Top_K[::-1]

			#compute regret
			self.R[t] =  self.reg_method(self,Top_K,C,t) + self.R[t-1] 

			#act and get the observation
			clicked = np.zeros(self.K)
			if np.amax(C[t][Top_K]) == 0:
				C_t = self.K+1
			else:
				C_t = np.argmax(C[t][Top_K])
				clicked[C_t] = 1

			T_old = np.empty_like (T)
			np.copyto(T_old,T)

			#compute which bandits are examed
			examed = min(C_t+1,self.K)

			#update
			T[Top_K[:examed]] = T[Top_K[:examed]]+1
			w_hat[Top_K[:examed]] = np.true_divide(T_old[Top_K[:examed]]*w_hat[Top_K[:examed]]+clicked[:examed],T[Top_K[:examed]])

	def draw(self):
		"""Display the regret/time graph
		"""
		ts = pd.Series(self.R[2:], index=xrange(2,self.N+2))
		ts.plot()
		plt.show()

	def getRegret(self):
		return self.R[2:]




if __name__ == "__main__":



	L = 256
	D = 256

	total = 20000

	allUser = {}
	Matrix = []

	indOfUser = 0;

# '/Users/haoni/Documents/LinearCascadingBandits/bus_users_res.csv/part-00000'
	with open('/home/nh/Documents/LinearCascadingBandits/bus_users_res.csv/part-00000') as f:
	    for i in xrange(D):
	        Users = f.readline().strip('\n').split(',')[1:]
	        Matrix.append(Users)
	        for user in Users:
	            if user not in allUser:
	                allUser[user] = indOfUser;
	                indOfUser += 1
	   

	finalMatrix = np.zeros((D,total))


	for i in xrange(len(Matrix)):
	    for user in Matrix[i]:
	    	if allUser[user] < total:
	        	finalMatrix[i][allUser[user]] = 1

	pkl_file = open('test.pkl', 'rb')
	test = pickle.load(pkl_file)
	pkl_file.close()

	finalMatrix = finalMatrix[:L,test]

	# finalMatrix = np.copy(finalMatrix[:,:25]) 
	# print finalMatrix.shape
	# finalMatrix = finalMatrix[:L]
	sampleBase = np.copy(finalMatrix) 

	# print "!!"
	# print len(allUser)
	# print "!!"

	print np.sum(finalMatrix[0])
	print np.sum(finalMatrix[1])
	print np.sum(finalMatrix[2])


	optimalSet = Set([])
	K = 4

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

	# numpy.savetxt("foo.csv", a, delimiter=",")


	L = 256
	N = 100000
	K = 4

	# f = open('bus_prob_.csv/part-00000')
	# prob, bus_id = np.genfromtxt(f,delimiter=",",unpack=True)
	
	# C = np.zeros((L,N+2))
	# for i in xrange(L):
	# 	C[i] = bernoulli.rvs(prob[i], size=(N+2,))

	# C = C.T

	res = []
	for t in xrange(10):
		C = np.zeros((N+2,L))
		for i in xrange(N+2):
			print t,i
			col = np.random.randint(sampleBase.shape[1])
			# col = np.random.randint(25)
			C[i] = sampleBase[:,col]

		ucbLikeBandit = UCBLikeBandit(L=L,N=N,K=K)
		ucbLikeBandit.fit(C)
		res.append(ucbLikeBandit.getRegret())

	res = np.array(res)

	np.savetxt('a_L256_UCB.csv', res, fmt='%.1e', delimiter=',')

	std = [np.std(res[:,i]) for i in xrange(res.shape[1])]
	SEM = [i/sqrt(10) for i in std]
	mean = [np.mean(res[:,i]) for i in xrange(res.shape[1])]

	x = range(len(mean))
	y = mean

	print y
	plt.figure()
	plt.errorbar(x, y, yerr=SEM)
	plt.title("")
	plt.show()


	# ucbLikeBandit = UCBLikeBandit()
	# ucbLikeBandit.fit(ucbLikeBandit.observationGenerator())

	# ucbLikeBandit.draw()



