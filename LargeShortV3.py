# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 17:29:31 2023

@author: Victor GPC
"""


#########################################################################

import scipy.stats as ss
import numpy as np
import bisect
import random
from matplotlib import pyplot as plt

#########################################################################





'''################################################################'''
class ProcessN:
    
    def __init__(self, length,repetitions):
        self.L = length 
        self.repetitions = repetitions
        self.intervals_endPoints = [0,self.L]
        self.min = np.array([])
        self.max = np.array([])
        self.distances = []


    def insert(self,val):
        if(val in self.intervals_endPoints): exit
        else:
            bisect.insort(self.intervals_endPoints, val)
        return list


    '''
    Compute length of each sub-interval
    # List: an ordered list with values of the location of the cutting points.
    '''    
    def dist_values(self):
        lista = self.intervals_endPoints
        # with n+2 ordered points (counting the two end-points) we can compute n+1 lengths (distances)
        size = len(lista); diff = np.zeros(size-1)
        for i in range(size):
            if(i>0):
                diff[i-1] = lista[i]-lista[i-1]
        self.distances = diff
              
    
    def compute_lengths(self):
        
        for i in range(self.repetitions):
            # sample interval uniformly;
            #ran = random.choice(range(len(intervals_endPoints)-1))+1  #take the right end-point of the interval

            population = list(range(len(self.intervals_endPoints)))  # this is for index/label for interval selection
            if(i==0): ws = np.append(0,self.L)
            else: ws = np.append(0,self.distances)  # weights for interval selection
            ran = random.choices(population,weights=ws)[0]

            # new cutting point unif. randomly selected.
            # the cut distribution can be changed later if needed.
            x = np.random.uniform(self.intervals_endPoints[ran-1], self.intervals_endPoints[ran], 1)[0]
            #x = trunc_exp_rv(intervals_endPoints[ran-1], intervals_endPoints[ran], 500, 1)[0]
            #x = truncnorm.rvs(intervals_endPoints[ran-1], intervals_endPoints[ran], size=1)[0]

            self.insert(x)  # adding new value to list of end-interval points
            #beta.append(index)
            #index += 1

            #distances = [];  
            self.dist_values()  #dist_values could be optimized later.
            self.min = np.append(self.min, np.min(self.distances))
            self.max = np.append(self.max, np.max(self.distances))
            #defect[i] = max[i] - min[i]


'''################################################################'''


#### testing
test2 = ProcessN(1009,10139)
test2.compute_lengths()
distancesN = test2.distances
test2.intervals_endPoints



# Defect plot
start = 50
x = np.linspace(1, test2.repetitions, test2.repetitions - start)
y = test2.max[start:]
plt.plot(x,y)
y = test2.min[start:]
plt.plot(x,y)
plt.title("Min and Max values with respect to time")
plt.xlabel("index")
plt.ylabel("value")
plt.show()




# distances plot
x = np.linspace(0, test2.repetitions, test2.repetitions+1)
y = test2.distances
plt.scatter(x,y)
plt.title("Interval distances plot")
plt.xlabel("index")
plt.ylabel("value")
plt.show()


# histogram of distances


lambda1 = test2.L*np.mean(test2.distances)
lambda2 = test2.repetitions*np.mean(test2.distances)

P   = ss.expon.fit(test2.L*test2.distances)
rX  = np.linspace(0,3000, 3000)
rP  = ss.expon.pdf(rX, *P)
rP2 = ss.expon.pdf(rX,loc=0,scale=lambda1)
rP3 = ss.expon.pdf(rX,loc=0,scale=lambda2)

# Histogram 1
fig, ax = plt.subplots(1, 1)
ax.hist(test2.L*test2.distances,bins='auto',density=True, histtype='stepfilled', alpha=0.2)
ax.plot(rX, rP)
ax.plot()

# Histogram 2
fig, ax = plt.subplots(1, 1)
ax.hist(test2.L*test2.distances,bins='auto',density=True, histtype='stepfilled', alpha=0.2)
ax.plot(rX, rP2)
ax.plot()

# Histogram 3
fig, ax = plt.subplots(1, 1)
ax.hist(test2.repetitions*test2.distances,bins='auto',density=True, histtype='stepfilled', alpha=0.2)
ax.plot(rX, rP3)
ax.plot()


#########################################################################

P[1]
1/P[1]
np.mean(test2.L *test2.distances)
1/np.mean(test2.L *test2.distances)

test2.L
plt.hist(test2.L*test2.distances)
m1 = np.mean(test2.L*test2.distances)
v1 = np.var(test2.L*test2.distances)
v1 - m1**2



########################################
sample = np.random.exponential(10,8000)
plt.hist(sample)
m2 = np.mean(sample)
v2 = np.var(sample)
v2 - m2**2
# MLE of lambada for a exponential dist n/sum_i^n(x_i)
test2.repetitions / np.sum(test2.distances) 
np.sum(test2.distances) / test2.repetitions


########################################

test1 = ProcessN(1009,50461)
test1.compute_lengths()

lambda0 = test1.L*np.mean(test1.distances)
rX  = np.linspace(0,200, 200)
rP0 = ss.expon.pdf(rX,loc=0,scale=lambda0)

# Histogram 
fig, ax = plt.subplots(1, 1)
ax.hist(test1.L*test1.distances,bins='auto',density=True, histtype='stepfilled', alpha=0.2)
ax.plot(rX, rP0)
ax.plot()



# Fitting Exponential
P   = ss.expon.fit(test1.L*test1.distances)
rP  = ss.expon.pdf(rX, *P)
#histomgram fitted Exponential
fig, ax = plt.subplots(1, 1)
ax.hist(test1.L*test1.distances,bins='auto',density=True, histtype='stepfilled', alpha=0.2)
ax.plot(rX, rP)
ax.plot()



#########################################################################   




# length of interval to be 'dissected'
vlength =       [1009, 1103, 1259, 2659, 3307]
vrepetitions =  [8053,8963,9049,9601,9973,10139,11471,12251,13043,14797,14947,15443,16223,17239,50461,99529,100003]


numeros = []
medias = []
for irep in vrepetitions:
    L=1009
    ryan_temp = ProcessN(L,irep)
    ryan_temp.compute_lengths()
    ndist = ryan_temp.distances
    nmean = np.mean(ndist)
    temp = np.divide(L, nmean)
    numeros.append(temp)
    medias.append(nmean)
    




########################################################################
########################################################################



from joblib import Parallel, delayed

def SeekingLambda(L,irep):
    ryan_temp = ProcessN(L,irep)
    ryan_temp.compute_lengths()
    ndist = ryan_temp.distances
    nmean = np.mean(ndist)
    temp = np.divide(L, nmean)
    #numeros.append(temp)
    #medias.append(nmean)
    print("L: {}  and n: {}".format(L,irep))
    return temp, ndist



Li= vlength= [1009, 1103, 1259, 2659, 3307,4019,5021,6359]
vrepetitions =  [8053,8963,9049,9601,9973,10139,11471,12251,13043,14797,14947,15443,16223,17239,50461,99529,100003]

results0 = Parallel(n_jobs=-2, prefer="threads")(delayed(SeekingLambda)(Li[0],irep) for irep in vrepetitions)
results1 = Parallel(n_jobs=-2, prefer="threads")(delayed(SeekingLambda)(Li[1],irep) for irep in vrepetitions)
results2 = Parallel(n_jobs=-2, prefer="threads")(delayed(SeekingLambda)(Li[2],irep) for irep in vrepetitions)
results3 = Parallel(n_jobs=-2, prefer="threads")(delayed(SeekingLambda)(Li[3],irep) for irep in vrepetitions)
results4 = Parallel(n_jobs=-2, prefer="threads")(delayed(SeekingLambda)(Li[4],irep) for irep in vrepetitions)
results5 = Parallel(n_jobs=-2, prefer="threads")(delayed(SeekingLambda)(Li[5],irep) for irep in vrepetitions)
results6 = Parallel(n_jobs=-2, prefer="threads")(delayed(SeekingLambda)(Li[6],irep) for irep in vrepetitions)
results7 = Parallel(n_jobs=-2, prefer="threads")(delayed(SeekingLambda)(Li[7],irep) for irep in vrepetitions)






############################################################### 

import pickle

def save(filename, *args):
    # Get global dictionary
    glob = globals()
    d = {}
    for v in args:
        # Copy over desired values
        d[v] = glob[v]
    with open(filename, 'wb') as f:
        # Put them in the file 
        pickle.dump(d, f)

def load(filename):
    # Get global dictionary
    glob = globals()
    with open(filename, 'rb') as f:
        for k, v in pickle.load(f).items():
            # Set each global variable to the value from the file
            glob[k] = v
    


args = ['results0','results1','results2','results3','results4','results5','results6','results7','vlength','vrepetitions']
filename = 'C:/Users/Victor GPC/Desktop/var_temp2'
save(filename,*args)




load(filename)






#####################################################################

var_tem1 =  results0[7][1]
L1 =        results0[7][0]

lambda1     =   L1*np.mean(var_tem1)
P           =   ss.expon.fit(L1*var_tem1)
rX          =   np.linspace(0,6000, 6000)
rP2         =   ss.expon.pdf(rX,loc=0,scale=lambda1)

# Histogram 2
fig, ax = plt.subplots(1, 1)
ax.hist(L1*var_tem1,bins='auto',density=True, histtype='stepfilled', alpha=0.2)
ax.plot(rX, rP2)
ax.plot()




















