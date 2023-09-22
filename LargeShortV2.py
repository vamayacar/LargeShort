globals().clear()

''' Victor Amaya Carvajal
    va80@math.duke.edu
'''

import scipy.stats as ss
import numpy as np
import bisect
import random
from matplotlib import pyplot as plt
import scipy.stats as ss
from scipy.stats import truncnorm


'''
Add new element to the list, we then sort the list to have
a ordered list in increasing order. 
'''
def insert(list, n):
    if(n in list): return list
    else:
        bisect.insort(list, n)
    return list

'''
Compute length of each sub-interval
# List: an ordered list with values of the location of the cutting points.
'''
def dist_values(lista :list):
    # with n+2 ordered points (counting the two end-points) we can compute n+1 lengths (distances)
    size = len(lista); diff = np.zeros(size-1)
    for i in range(size):
        if(i>0):
            diff[i-1] = lista[i]-lista[i-1]
    return diff


'''
Function to sample from a truncated exponential distribution at the 
inteval [low,high], with parameter lambda = scale. 
'''
def trunc_exp_rv(low, high, scale, size):
    rnd_cdf = np.random.uniform(ss.expon.cdf(x=low, scale=scale),
                                ss.expon.cdf(x=high, scale=scale),
                                size=size)
    return ss.expon.ppf(q=rnd_cdf, scale=scale)

# trunc_exp_rv(0,10,1,1000).mean()
# 1/ (ss.expon.cdf(10, loc=0, scale=1) - ss.expon.cdf(0, loc=0, scale=1))


'''********************************************'''
""" functions i am not using, here just in case"""
import scipy.optimize as so
def solve_for_l(low, high, ept_mean):
    A = np.array([low, high])
    return 1/so.fmin(lambda L: ((np.diff(np.exp(-A*L)*(A*L+1)/L)/np.diff(np.exp(-A*L)))-ept_mean)**2,
                     x0=0.5,
                     full_output=False, disp=False)
def F(low, high, ept_mean, size):
    return trunc_exp_rv(low, high,
                        solve_for_l(low, high, ept_mean),
                        size)

#rv_data = F(1, 10, 2.5, 1e5)
#plt.hist(rv_data, bins=50)
#plt.xlim(0, 12)
#rv_data.mean()
""" end of functions not using """
'''********************************************'''





###############################################

# length of interval to be 'dissected'
L=length = 1009 #(this is a prime number)

# we start with the end-points of the interval we will be dissecting.
intervals_endPoints = [0,L] # localization

# number of cuts we want to make (or how many times we want to run the process)
repetitions = 100003  #(this is a prime number)
# Thus, we'll have a total of "repetitions+2" points (need to add the end points)
# for a total of "repetitions+1" intervals.


# variables initialization
min = np.array([])
max = np.array([])
distances = [L]
defect = np.zeros(repetitions)

for i in range(repetitions):
    # sample interval uniformly;
    #ran = random.choice(range(len(intervals_endPoints)-1))+1  #take the right end-point of the interval

    population = list(range(len(intervals_endPoints)))  # this is for index/label for interval selection
    ws = np.append(0,distances)  # weights for interval selection
    ran = random.choices(population,weights=ws)[0]

    # new cutting point unif. randomly selected.
    # the cut distribution can be changed later if needed.
    x = np.random.uniform(intervals_endPoints[ran-1], intervals_endPoints[ran], 1)[0]
    #x = trunc_exp_rv(intervals_endPoints[ran-1], intervals_endPoints[ran], 500, 1)[0]
    #x = truncnorm.rvs(intervals_endPoints[ran-1], intervals_endPoints[ran], size=1)[0]

    insert(intervals_endPoints, x)  # adding new value to list of end-interval points
    #beta.append(index)
    #index += 1

    distances = [];  distances = dist_values(intervals_endPoints)  #dist_values could be optimized later.
    min = np.append(min, np.min(distances))
    max = np.append(max, np.max(distances))
    defect[i] = max[i] - min[i]


#print('aqui imprimiremos varios valores')
#print(f'Min distance', min)
#print(f'Max distance', max)

# Defect plot
x = np.linspace(1, repetitions, repetitions-1)
y = defect[1:]
plt.plot(x,y)
plt.title("Defect plot")
plt.xlabel("index")
plt.ylabel("value")
plt.show()



# distances plot
x = np.linspace(0, repetitions, repetitions+1)
y = distances
plt.scatter(x,y)
plt.title("Interval distances plot")
plt.xlabel("index")
plt.ylabel("value")
plt.show()


# histogram of distances


lambda1 = np.mean(L*distances)
lambda2 = np.mean(repetitions*distances)

P = ss.expon.fit(L*distances)
rX = np.linspace(0,100, 100)
rP = ss.expon.pdf(rX, *P)
rP2 = ss.expon.pdf(rX,loc=0,scale=lambda1)
rP3 = ss.expon.pdf(rX,loc=0,scale=lambda2)

# Histogram 1
fig, ax = plt.subplots(1, 1)
ax.hist(L*distances,bins='auto',density=True, histtype='stepfilled', alpha=0.2)
ax.plot(rX, rP)
ax.plot()

# Histogram 2
fig, ax = plt.subplots(1, 1)
ax.hist(L*distances,bins='auto',density=True, histtype='stepfilled', alpha=0.2)
ax.plot(rX, rP2)
ax.plot()


# Histogram 3
fig, ax = plt.subplots(1, 1)
ax.hist(repetitions*distances,bins='auto',density=True, histtype='stepfilled', alpha=0.2)
ax.plot(rX, rP3)
ax.plot()




# minimum plot
x = np.linspace(1, repetitions, repetitions)
y = min
plt.plot(x,y)
#plt.show()

# maximum plot
x = np.linspace(1, repetitions, repetitions)
y = max
plt.plot(x,y)
plt.title("Min and Max values with respect to time")
plt.xlabel("index")
plt.ylabel("value")
plt.show()


# Ratio plot
x = np.linspace(1, repetitions, repetitions)
y =np.divide(max,min)
plt.plot(x,y)
plt.title("Min/min ratio plot")
plt.xlabel("index")
plt.ylabel("value")
plt.show()





'''
here i'll try to find the rate at which min goes to zero.
'''

start = 100
x = np.linspace(1, repetitions, repetitions - start)
y = min[start:]
#y2 = [ np.sqrt(i)*y[i] for i in range(len(y))  ]
y2 = np.exp(y)
plt.plot(x,y2)
plt.title("exp(min) ratio plot; ini=500")
plt.show()


ini = 500
fin = repetitions
# fin = repetitions
x = np.linspace(1, repetitions, repetitions)
y2 = np.exp(min[ini:fin])
plt.plot(x[ini:fin], np.divide(max[ini:fin],y2)  )
plt.title("Max/exp(min) ratio plot; ini=50")
plt.show()


y3 = np.log(max[ini:])**4
plt.plot(x[ini:], np.divide(y3,min[ini:])  )
plt.title("log(Max)/min ratio plot; ini=50")
plt.show()







