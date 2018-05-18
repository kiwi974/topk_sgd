import random
import sparseToolsDict as std
import math
#import matplotlib.pyplot as plt



########################################################################
# Generation of data to test stochastic gradient descent. One can
# visualize the data : remove the comments on the import of matplotlib
# (line 2) and one the plot (lines 102 to 105). We generate points in
# the square [0,10]x[0,10], the hyperplan is y = 10 - x.

# Input :
#       -nbData : number of generated data.

# Output :
#       -trainingSet : a set of tuples (label,example) which is a
#        training. The format of this set is
#        List(label : Int, example : List(float)) -> label = +1 or -1.
########################################################################




def generateData(nbData):

    # d is the double of the distance of each point in the square to the
    # separator hyperplan.
    d = 2

    # u is a hyperplan's orthogonal vector.
    u = {1: 1, 2: 1}
    # d is the double of the distance of each point in the square to the
    # separator hyperplan.
    d = .1

    # u is a hyperplan's orthogonal vector.
    u = {1: 1, 2: 1}

    # A and B denote each a different class, respectively associated
    # to the labels 1 and -1.
    A = []
    B = []

    # Number of examples we kept for our training set.
    nbExamples = 0

    # Number of data we rejected because they are not at the good
    # distance of the hyperplan.
    nbRejeted = 0

    absA, absB = [], []
    ordA, ordB = [], []

    cardA = 0
    cardB = 0

    while (nbExamples < nbData):
        a = random.randint(0,100)/10
        b = random.randint(0,100)/10
        sign = random.random()
        if (sign <= 0.25):
            a = -a
            b = -b
        elif ((0.25 <= sign) & (sign <= 0.5)):
            b = -b
        elif ((0.5 <= sign) & (sign <= 0.75)):
            a = -a
        genvect = {1:a,2:b}
        dist = abs((std.sparse_dot(u,genvect))/(math.sqrt(std.sparse_dot(u,u))))
        valide = (d==0) or ((d!=0) & (dist >= d))
        if (b > -a) & valide:
            A.append({-1:1,1:a,2:b})
            absA.append(a)
            ordA.append(b)
            nbExamples += 1
            cardA += 1
        elif (b < -a) & valide:
            B.append(({-1:-1,1:a,2:b}))
            absB.append(a)
            ordB.append(b)
            nbExamples += 1
            cardB += 1
        else:
            nbRejeted += 1

    #plt.scatter(absA,ordA,s=10,c='r',marker='*')
    #plt.scatter(absB,ordB,s=10,c='b',marker='o')
    #plt.plot([-10,10],[10,-10],'orange')
    #plt.show()



    trainingSet = A+B

    return trainingSet,absA,ordA,absB,ordB



#generateData(500)



########################################################################
# Sample a set in a subset of numSamples elements.
# Input :
#       -set : the set to sample.
#       -numSamples : the number of elements of set that we want in
#        the subset.
# Output :
#       -dataSample : the subset (sample of set).
#       -sampleSize : the size of the sample set (which is equal to
#        numSamples.
########################################################################


def sample(set,numSamples):
    n = len(set)
    dataSample = []
    cpyData = list(set)

    # Sample the data set on which complete the learning phase
    sampleSize = 0
    while ((sampleSize < numSamples) and (sampleSize < n)):
        s = random.randint(0, n - sampleSize - 1)
        dataSample.append(cpyData[s])
        del cpyData[s]
        sampleSize += 1

    return dataSample, sampleSize




########################################################################
# Computation of the error given by the SVM on a sample.
# Input :
#       -w : the vector of parameters.
#       -l : the depreciation factor of the SVM.
#       -sample : the training set.
#       -sampleSize : number of examples in sample, i.e. the size of
#        this set.
# Output :
#       -cost : the cost computed on sample.
########################################################################

def error(w,l,sample,sampleSize):
    norm = l*std.sparse_dot(w,w)
    sum = 0
    for i in range(sampleSize):
        label = sample[i].get(-1,0)
        example = std.take_out_label(sample[i])
        sum += max(0,1-label*std.sparse_dot(w,example))
    cost = norm + sum
    return cost



########################################################################
# Computation of the derivation of the error on a sample.
# Input :
#       -w : the vector of parameters.
#       -l : the depreciation factor of the SVM.
#       -sample : the training set.
#       -sampleSize : number of examples in sample, i.e. the size of
#        this set.
# Output :
#       -dcost : the value of the derivative of the cost computed on
#        sample.
########################################################################

def der_error(w,l,sample,sampleSize):
    d = std.sparse_mult(l, w)
    sum = {}
    for i in range(sampleSize):
        label = sample[i].get(-1,0)
        example = std.take_out_label(sample[i])
        if (label*(std.sparse_dot(w,example)) < 1):
            sum = std.sparse_vsum(sum,std.sparse_mult(-label,example))
    #print("der_err summ part = " + str(sum))
    dcost = std.sparse_vsum(d,sum)
    return dcost





########################################################################
# Stochastic gradient descent.
# Input : -data : the data set to sample and on which learn.
#         -w : the current vector of parameters.
#         -numSamples : the number of samples we want for the learning.
#         -step : departure step of the gradient descent.
#         -l : depreciation factor of the SVM.
# Output : -w : the vector of parameters according to a random sample
#               of data.
########################################################################



def descent(data,w,numSamples,l):

    # Sample of the data set.
    dataSample,sampleSize = sample(data,numSamples)

    # Derivative of the cost evaluated on dataSample.
    d = der_error(w,l,dataSample,sampleSize)

    # Modification of the parameter vector w.
    #w = std.sparse_vsous(w,std.sparse_mult(step,d))

    return d





