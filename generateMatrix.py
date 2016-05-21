import pandas as pd
import numpy as np
import pickle

from sets import Set


L = 30

allUser = {}
Matrix = []

indOfUser = 0;

with open('/Users/haoni/Documents/LinearCascadingBandits/bus_users_30.csv') as f:
    for i in xrange(L):
        Users = f.readline().strip('\n').split(',')[1:]
        print len(Users)
        Matrix.append(Users)
        for user in Users:
            if user not in allUser:
                allUser[user] = indOfUser;
                indOfUser += 1
   

finalMatrix = np.zeros((L,len(allUser)))


for i in xrange(len(Matrix)):
    print i
    for user in Matrix[i]:
        finalMatrix[i][allUser[user]] = 1


f = open('finalMatrix2.pickle', 'wb')
pickle.dump(finalMatrix, f, pickle.HIGHEST_PROTOCOL)
f.close()

