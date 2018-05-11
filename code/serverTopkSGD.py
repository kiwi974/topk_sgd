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

_ONE_DAY_IN_SECONDS = 24 * 60 * 60

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
trainingSet, trainaA, trainoA, trainaB, trainoB = sgd.generateData(nbExamples)
trainingSet = std.dataPreprocessing(trainingSet, hypPlace)

# Number of examples we want in our training set.
nbTestingData = 1600

# Set of generated data for testing.
testingSet, testaA, testoA, testaB, testoB = sgd.generateData(nbTestingData)
testingSet = std.dataPreprocessing(testingSet, hypPlace)

# Pre-processing of the data (normalisation and centration).
# data = tools.dataPreprocessing(data)

# Initial vector to process the stochastic gradient descent :
# random generated.
w0 = {1: 1.56, 2: 1.75, hypPlace: 0.011}  # one element, to start the computation
normw0 = math.sqrt(std.sparse_dot(w0, w0))
nbParameters = len(trainingSet[0]) - 1  # -1 because we don't count the label

# Maximum number of epochs we allow.
nbMaxCall = 200

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
        self.epoch = -1
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
        self.step = 8

    def GetFeature(self, request, context):

        ######################################################################
        # Section 1 : wait for all the clients -> get their vectors and
        # appoint one of them as the printer.

        self.iterator += 1
        if (request.poids == "pret" or request.poids == "getw0"):
            self.vectors.append(request.poids)
        else:
            kv = request.poids.split("<||>")
            self.vectors.append([float(kv[0]),float(kv[1])])
        self.enter_condition = (self.iterator == nbClients)
        waiting.wait(lambda: self.enter_condition)

        self.printerThreadName = threading.current_thread().name

        ######################################################################

        ######################################################################
        # Section 2 : compute the new vector -> send the data, a merge of
        # all the vectors we got from the clients or the message 'stop' the
        # signal to the client that we converged.

        if (request.poids == 'pret'):
            vector = std.datadict2Sstr(trainingSet)
        elif (request.poids == 'getw0'):
            vector = std.dict2str(w0) + "<<||>>" + str(self.step)
        else:
            # Modification of the vector of parameters
            gradParam = std.mergeTopk(self.vectors)
            vector = std.sparse_vsous(self.oldParam,gradParam)
            # Checking of the stoping criterion
            diff = std.sparse_vsous(self.oldParam, vector)
            normDiff = math.sqrt(std.sparse_dot(diff, diff))
            normGradW = math.sqrt(std.sparse_dot(vector, vector))
            normPrecW = math.sqrt(std.sparse_dot(self.oldParam, self.oldParam))
            if ((normDiff <= 10 ** (-3) * normPrecW) or (self.epoch > nbMaxCall) or (normGradW <= 10 ** (-3) * normw0)):
                self.paramVector = vector
                vector = 'stop'
            else:
                vector = std.dict2str(vector) + "<<||>>" + str(self.step)

        ######################################################################

        ######################################################################
        # Section 3 : wait that all the threads pass the computation area, and
        # store the new computed vector.

        self.iterator -= 1

        self.exit_condition = (self.iterator == 0)
        waiting.wait(lambda: self.exit_condition)

        realComputation = (request.poids != 'pret') and (request.poids != 'getw0') and (vector != 'stop')


        if (realComputation):
            self.oldParam = std.str2dict(vector.split("<<||>>")[0])

        ######################################################################

        ###################### PRINT OF THE CURRENT STATE ######################
        if (threading.current_thread().name == self.printerThreadName):
            print('')
            print('############################################################')
            if (self.epoch == 0):
                print('# We sent the data to the clients.')
            else:
                print('# We performed the epoch : ' + str(self.epoch) + '.')
                if (vector == "stop"):
                    print("# The vector that achieve the convergence is : " + str(self.paramVector))
                    # Plot the error on the training set
                    plt.figure(1)
                    plt.plot([i for i in range(self.epoch - 1)], self.testingErrors, 'b')
                    plt.plot([i for i in range(self.epoch - 1)], self.trainingErrors, 'r')
                    plt.show()
                    # Plot the training set and the hyperplan
                    plt.figure(2)
                    plt.scatter(trainaA, trainoA, s=10, c='r', marker='*')
                    plt.scatter(trainaB, trainoB, s=10, c='b', marker='o')
                    plt.plot([-5, 5], [5, -5], 'orange')
                    w1 = self.paramVector.get(1, 0)
                    w2 = self.paramVector.get(2, 0)
                    b = self.paramVector.get(hypPlace, 0)
                    i1 = (5 * w1 - b) / w2
                    i2 = (-5 * w1 - b) / w2
                    plt.plot([-5, 5], [i1, i2], 'crimson')
                    plt.show()
                    print("We went out of the loop because : ")
                    if (normDiff <= 10 ** (-3) * normPrecW):
                        print("     normDiff <= 10 ** (-3) * normPrecW")
                    elif (normGradW <= 10 ** (-3) * normw0):
                        print("     normGradW <= 10 ** (-3) * normw0")
                    else:
                        print("     self.epoch > nbMaxCall")
            if (realComputation or (self.epoch == 1)):
                # Compute the error made with that vector of parameters on the testing set
                self.testingErrors.append(sgd.error(self.oldParam, 0.1, testingSet, nbTestingData, hypPlace))
                self.trainingErrors.append(sgd.error(self.oldParam, 0.1, trainingSet, nbExamples, hypPlace))
                print('# The merged vector is : ' + vector + '.')
            if (self.epoch == nbMaxCall):
                print('We performed the maximum number of iterations.')
                print('The descent has been stopped.')
            print('############################################################')
            print('')
            self.epoch += 1
            self.step = self.step * 0.9
        ############################### END OF PRINT ###########################

        ######################################################################

        ######################################################################
        # Section 4 : empty the storage list of the vectors, and wait for all
        # the threads.

        self.vectors = []
        waiting.wait(lambda: (self.vectors == []))

        ######################################################################

        # time.sleep(1)
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
