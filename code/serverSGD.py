"""The Python implementation of the gRPC stochastic gradient descent server.
Note : it's necessary to download and install the package waiting of Python to execute that code. it permits to create
barriers for the threads at different point of the algorithm. while loops was not enough : we can explain the different
problems we had with that synchronization part during the meeting, and justify the use of the librairy. """

from concurrent import futures

import math
import time
import waiting

import grpc
import random
import route_guide_pb2
import route_guide_pb2_grpc

import threading

import matplotlib.pyplot as plt

import sgd
import sparseToolsDict as std


_ONE_DAY_IN_SECONDS = 24*60*60

""" Define the number of clients you want to use."""
nbClients = 2



# Number of examples we want in our training set.
nbExamples = 4000

# Total number of descriptors per example
nbDescript = 2

# Place of the constante 1 in each example : it
# permits to include the hyperplan constant to the
# vector of parameters

hypPlace = nbDescript + 2

# Set of generated data for training.
trainingSet, trainaA,trainoA, trainaB, trainoB = sgd.generateData(nbExamples)
trainingSet = std.dataPreprocessing(trainingSet,hypPlace)

# Number of examples we want in our testing set.
nbTestingData = 1600

# Set of generated data for testing.
testingSet, testaA, testoA, testaB, testoB = sgd.generateData(nbTestingData)
testingSet = std.dataPreprocessing(testingSet,hypPlace)

# Pre-processing of the data (normalisation and centration).
#data = tools.dataPreprocessing(data)

# Initial vector to process the stochastic gradient descent :
# random generated.
w0 = {1:1.21,2:1.75,hypPlace:0.011}                  #one element, to start the computation
normGradW0 = math.sqrt(std.sparse_dot(w0,w0))
nbParameters = len(trainingSet[0])-1  #-1 because we don't count the label


# Maximum number of epochs we allow.
nbMaxCall = 100

# The depreciation of the SVM norm cost
l = 0.5





class RouteGuideServicer(route_guide_pb2_grpc.RouteGuideServicer):



    """ We define attributes of the class to perform the computations."""
    def __init__(self):
        # An iterator that will count the number of clients that contact the
        # server at each epoch.
        self.iterator = 0
        # A barrier condition to be sure that every waited client contacted
        # the server before to start the GetFeature method (kind of join).
        self.enter_condition = (self.iterator == nbClients)
        # An other barrier condition, that acts like a join on the threads to.
        self.exit_condition = (self.iterator == 0)
        # A list to store all the vectors sent by each client at each epoch.
        self.vectors = []
        # The current epoch (0 -> send the data to the clients).
        self.epoch = 0
        # The previous vecor of parameters : the last that had been sent.
        self.oldParam = w0
        # The name of one of the thread executing GetFeature : this one, and
        # only this one will something about the state of the computation in
        # the server.
        self.printerThreadName = ''
        # The final vector of parameters we find
        self.paramVector = {}
        # Error on the training set, computed at each cycle of the server
        self.trainingErrors = []
        # Error on the testing set, computed at each cycle of the server
        self.testingErrors = []
        # Step of the descent
        self.step = 1



    def GetFeature(self, request, context):

        ######################################################################
        # Section 1 : wait for all the clients -> get their vectors and
        # appoint one of them as the printer.

        global normGradW0
        self.iterator += 1
        if (request.poids == "pret" or request.poids == "getw0"):
            self.vectors.append(request.poids)
        else:
            self.vectors.append(std.str2dict(request.poids))
        self.enter_condition = (self.iterator == nbClients)
        waiting.wait(lambda : self.enter_condition)

        self.printerThreadName = threading.current_thread().name

        ######################################################################

        ######################################################################
        # Section 2 : compute the new vector -> send the data, a merge of
        # all the vectors we got from the clients or the message 'stop' the
        # signal to the client that we converged.

        normPrecW = 0
        normGradW = 0
        normDiff = 0

        if (request.poids == 'pret'):
            vector = std.datadict2Sstr(trainingSet)
        elif (request.poids == 'getw0'):
            vector = std.dict2str(w0)
        else :
            gradParam = std.mergeSGD(self.vectors)
            if (self.epoch == 2):
                normGradW0 = math.sqrt(std.sparse_dot(gradParam,gradParam))
            normGradW = math.sqrt(std.sparse_dot(gradParam, gradParam))
            gradParam = std.sparse_mult(self.step,gradParam)
            vector = std.sparse_vsous(self.oldParam,gradParam)
            diff = std.sparse_vsous(self.oldParam,vector)
            normDiff = math.sqrt(std.sparse_dot(diff,diff))
            normPrecW = math.sqrt(std.sparse_dot(self.oldParam,self.oldParam))
            if ((normDiff <= 10 ** (-8) * normPrecW) or (self.epoch > nbMaxCall) or (normGradW <= 10**(-8)*normGradW0)):
                self.paramVector = vector
                vector = 'stop'
            else:
                vector = std.dict2str(vector)

        ######################################################################

        ######################################################################
        # Section 3 : wait that all the threads pass the computation area, and
        # store the new computed vector.
                 
        self.iterator -= 1

        self.exit_condition = (self.iterator == 0)
        waiting.wait(lambda : self.exit_condition)

        realComputation = (request.poids != 'pret') and (request.poids != 'getw0') and (vector != 'stop')

        if (realComputation):
            self.oldParam = std.str2dict(vector)

        ######################################################################

        ###################### ONE THREAD ONLY SECTION  ######################

        if (threading.current_thread().name == self.printerThreadName):
            # Printing of the trace
            std.printTrace(self.epoch, vector, self.paramVector, self.testingErrors, self.trainingErrors, trainaA,                trainaB, trainoA, trainoB,hypPlace, normDiff, normGradW, normPrecW, normGradW0, realComputation,                         self.oldParam, trainingSet,testingSet, nbTestingData, nbExamples, nbMaxCall)
            # Modification of the epoch and of the step
            self.epoch += 1
            self.step = self.step*0.99
            print("step = " + str(self.step))
        ############################### END OF PRINT ###########################

        ######################################################################

        ######################################################################
        # Section 4 : empty the storage list of the vectors, and wait for all
        # the threads.

        self.vectors = []
        waiting.wait(lambda : (self.vectors == []))

        ######################################################################

        #time.sleep(1)
        return route_guide_pb2.Vector(poids=vector)





def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    route_guide_pb2_grpc.add_RouteGuideServicer_to_server(
        RouteGuideServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
