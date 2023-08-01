
globals().clear()

import numpy as np
import bisect
import random
from matplotlib import pyplot as plt


def insert(list, n):
    bisect.insort(list, n)
    return list

# arguments
# List: an ordered list with values of the location of the cutting points.
def dist_values(lista :list):
    # with n ordered points we can compute n-1 distances
    size = len(lista); diff = np.zeros(size-1)
    for i in range(size):
        if(i>0):
            diff[i-1] = lista[i]-lista[i-1]
    return diff



###############################################

# length of interval to be 'dissected'
length = 1

alpha = [0,length] # localization
#beta  = [0]       # index

# number of cuts we want to make (or how many times we want to run the process)
repetitions = 100
#index = 1

min = np.array([])
max = np.array([])
distances = []
defect = np.zeros(repetitions)

for i in range(repetitions):
    # sample interval uniformly;
    ran = random.choice(range(len(alpha) - 1)) + 1  #take the right end-point of the interval

    # new cutting point unif. randomly selected.
    # the cut distribution can be changed later if needed.
    x = np.random.uniform(alpha[ran-1], alpha[ran], 1)[0]

    insert(alpha, x)
    #beta.append(index)
    #index += 1

    #new_distance = alpha[ran] - alpha[ran-1]
    #distances[i] = new_distance
    distances = [];  distances = dist_values(alpha)  #dist_values could be optimized later.
    min = np.append(min, np.min(distances))
    max = np.append(max, np.max(distances))
    defect[i] = max[i] - min[i]


#print('aqui imprimiremos varios valores')
#print(f'Min distance', min)
#print(f'Max distance', max)

# Defect plot
x = np.linspace(1, repetitions, repetitions)
y = defect
plt.plot(x,y)
plt.title("Defect plot")
plt.xlabel("index")
plt.ylabel("value")
plt.show()



# distances plot
x = np.linspace(0, repetitions, repetitions+1)
y = distances
plt.scatter(x,y)
plt.title("Distances vector plot")
plt.xlabel("index")
plt.ylabel("value")
plt.show()


# distances plot
x = np.linspace(1, repetitions, repetitions)
y = min
plt.plot(x,y)
#plt.show()

# distances plot
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
