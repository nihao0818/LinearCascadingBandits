import numpy as np
import pandas as pd
import math
from scipy.stats import bernoulli
import matplotlib.pyplot as plt


# L = 16 #total number of bandit
# N = 40000 #times
# K = 4 #everytime recommend K bandit
# delta = 0.15
# P = 0.2



class UCBLikeBandit:
	"""Implement a series of UCB-like Bandit Algorithm"""
	def UCB1(self,w_hat,t,T):
		return  w_hat + float(math.sqrt(1.5*math.log(t)))/np.sqrt(T) 

	def reg_probOfNoneClick(self,w_hat,K,Top_K):
		optimal	= np.argpartition(w_hat, -K)[-K:]
		optimal = optimal[np.argsort(w_hat[optimal])]
		optimal = optimal[::-1]

		return (1-np.prod(1-w_hat[optimal])) - (1-np.prod(1-w_hat[Top_K]))

	def observationGenerator(self):
		C_first_K=bernoulli.rvs(self.P, size=(self.N+2,self.K))
		C_rest=bernoulli.rvs(self.P-self.delta, size=(self.N+2,self.L-self.K))
		C = np.hstack((C_first_K,C_rest))
		# C = bernoulli.rvs(0.2, size=(N+2,L))
		C = np.asfarray(C, dtype='float')
		return C


	UCBs = {"UCB1":UCB1}
	REGs = {"reg_probOfNoneClick":reg_probOfNoneClick}

	def __init__(self,L=16,N=40000,K=4,P=0.2,delta=0.15,ucb_method="UCB1",reg_method="reg_probOfNoneClick"):
		self.L = L 
		self.N = N 
		self.K = K 
		self.P = P 
		self.delta = delta 
		self.ucb_method = UCBLikeBandit.UCBs[ucb_method]
		self.reg_method = UCBLikeBandit.REGs[reg_method]
		self.R = np.zeros(self.N+2)
		self.R = np.asfarray(self.R, dtype='float')




	def fit(self):
		C = self.observationGenerator()

		T = np.ones(self.L)
		T = np.asfarray(T, dtype='float')

		# w_hat = bernoulli.rvs(0.5, size=L)
		w_first_K=bernoulli.rvs(self.P, size=self.K)
		w_rest=bernoulli.rvs(self.P-self.delta, size=self.L-self.K)
		w_hat = np.hstack((w_first_K,w_rest))
		# w_hat = np.zeros(L)
		w_hat = np.asfarray(w_hat, dtype='float')
		# print w_hat

		for t in xrange(2,self.N+2):

			#compute UCB		
			U_t = self.ucb_method(self,w_hat,t-1,T)

			#sort and get top k values
			Top_K = np.argpartition(U_t, -self.K)[-self.K:]
			Top_K = Top_K[np.argsort(U_t[Top_K])]
			Top_K = Top_K[::-1]


			#regret
			self.R[t] =  self.reg_method(self,w_hat,self.K,Top_K) + self.R[t-1] 


			#act and get the observation
			clicked = np.zeros(self.K)
			if np.amax(C[t][Top_K]) == 0:
				C_t = self.K+1 
			else:
				C_t = np.argmax(C[t][Top_K])
				clicked[C_t] = 1



			T_old = np.empty_like (T)
			np.copyto(T_old,T)

			examed = min(C_t+1,self.K)

			T[Top_K[:examed]] = T[Top_K[:examed]]+1
			w_hat[Top_K[:examed]] = np.true_divide(T_old[Top_K[:examed]]*w_hat[Top_K[:examed]]+clicked[:examed],T[Top_K[:examed]])

	def draw(self):
		ts = pd.Series(self.R[2:], index=xrange(2,self.N+2))
		ts.plot()
		plt.show()





# def UCBLikeBanditAlgorithm(): 
# 	#modeled real clicks
# 	C_first_K=bernoulli.rvs(P, size=(N+2,K))
# 	C_rest=bernoulli.rvs(P-delta, size=(N+2,L-K))
# 	C = np.hstack((C_first_K,C_rest))
# 	# C = bernoulli.rvs(0.2, size=(N+2,L))
# 	C = np.asfarray(C, dtype='float')


# 	#regret


# 	T = np.ones(L)
# 	T = np.asfarray(T, dtype='float')

# 	# w_hat = bernoulli.rvs(0.5, size=L)
# 	w_first_K=bernoulli.rvs(P, size=K)
# 	w_rest=bernoulli.rvs(P-delta, size=L-K)
# 	w_hat = np.hstack((w_first_K,w_rest))
# 	# w_hat = np.zeros(L)
# 	w_hat = np.asfarray(w_hat, dtype='float')
# 	# print w_hat

# 	for t in xrange(2,N+2):

# 		#compute UCB		
# 		U_t = computeUCB(w_hat,t-1,T)

# 		#sort and get top k values
# 		Top_K = np.argpartition(U_t, -K)[-K:]
# 		Top_K = Top_K[np.argsort(U_t[Top_K])]
# 		Top_K = Top_K[::-1]

# 		optimal	= np.argpartition(w_hat, -K)[-K:]
# 		optimal = optimal[np.argsort(w_hat[optimal])]
# 		optimal = optimal[::-1]		

# 		#regret
# 		R[t] =  (1-np.prod(1-w_hat[optimal])) - (1-np.prod(1-w_hat[Top_K])) + R[t-1] 


# 		#act and get the observation
# 		clicked = np.zeros(K)
# 		if np.amax(C[t][Top_K]) == 0:
# 			C_t = K+1 
# 		else:
# 			C_t = np.argmax(C[t][Top_K])
# 			clicked[C_t] = 1



# 		T_old = np.empty_like (T)
# 		np.copyto(T_old,T)

# 		examed = min(C_t+1,K)

# 		T[Top_K[:examed]] = T[Top_K[:examed]]+1
# 		w_hat[Top_K[:examed]] = np.true_divide(T_old[Top_K[:examed]]*w_hat[Top_K[:examed]]+clicked[:examed],T[Top_K[:examed]])



# 	ts = pd.Series(R[2:], index=xrange(2,N+2))
# 	ts.plot()
# 	plt.show()




ucbLikeBandit = UCBLikeBandit()
ucbLikeBandit.fit()
ucbLikeBandit.draw()



