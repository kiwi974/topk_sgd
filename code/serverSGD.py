"""The Python implementation of the gRPC stochastic gradient descent server.
Note : it's necessary to download and install the package waiting of Python to execute that code. it permits to create
barriers for the threads at different point of the algorithm. while loops was not enough : we can explain the different
problems we had with that synchronization part during the meeting, and justify the use of the librairy. """

from concurrent import futures

import math
import time
import waiting

import grpc
import route_guide_pb2
import route_guide_pb2_grpc

import threading
import pickle

import sgd
import sparseToolsDict as std


_ONE_DAY_IN_SECONDS = 24*60*60

""" Define the number of clients you want to use."""
nbClients = 2


# Place of the constante 1 in each example : it
# permits to include the hyperplan constant to the
# vector of parameters.

hypPlace = 10**6


# Importation of the data

def treatData(data):
    for i in range(len(data)):
        if (data[i].get(-1,0) == [[1]]):
            data[i][-1] = 1
    return data

print("Starting of the server...")

dataType = "dense"

if (dataType == "dense"):
    path = '/home/kiwi974/cours/epfl/opti_ma/project/denseData/voice.csv'
    data = std.buildCSV2Database(path)
else:
    with open('/home/kiwi974/cours/epfl/opti_ma/project/sparseData/data6000new', 'rb') as f:
        data = treatData(pickle.load(f))

# Number of examples we want in our training set.
nbExamples = 200

# Number of samples we want for each training subset client
numSamples = 20

# Number of examples we want in our testing set.
nbTestingData = 30

print("Building of the training set...")

# Define the training set.
trainingSet = data[:nbExamples]

print("Training data pre-processing...")

trainingSet = std.dataPreprocessing(trainingSet,hypPlace)

print("Building of the testing set...")

# Define the testing set.
testingSet = data[nbExamples:nbExamples+nbTestingData]

print("Testing data pre-processing")

testingSet = std.dataPreprocessing(testingSet, hypPlace)

# The depreciation of the SVM norm cost
l = 0.01

# The step of the descent
step = 1

# Initial vector to process the stochastic gradient descent :
# random generated.
w0 = {1: 0.21, 2: 0.75, hypPlace: 0.011}  # one element, to start the computation
gW0 = sgd.der_error(w0,l,trainingSet,nbExamples)
normGW0 = math.sqrt(std.sparse_dot(gW0,gW0))
nbParameters = len(trainingSet[0]) - 1  # -1 because we don't count the label

# Maximum number of epochs we allow.
nbMaxCall = 100


# The depreciation of the SVM norm cost
l = 0.1

# Constants to test the convergence
c1 = 10**(-8)
c2 = 10**(-8)

print("Server ready....")

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
        self.step = step
        # Keep all the merged vectors
        self.merged = [w0]
        # Number of bytes send bu clients at each epoch
        self.bytesTab = {}

    def GetFeature(self, request, context):

        ######################################################################
        # Section 1 : wait for all the clients -> get their vectors and
        # appoint one of them as the printer.

        self.iterator += 1
        if (request.poids == "pret" or request.poids == "getw0"):
            self.vectors.append(request.poids)
        else:
            entry = request.poids.split("<bytes>")
            b = int(entry[1])
            if (self.epoch in self.bytesTab):
                self.bytesTab[self.epoch] += b
            else:
                self.bytesTab[self.epoch] = b
            self.vectors.append(std.str2dict(entry[0]))
        self.enter_condition = (self.iterator == nbClients)
        waiting.wait(lambda : self.enter_condition)

        self.printerThreadName = threading.current_thread().name

        ######################################################################

        ######################################################################
        # Section 2 : compute the new vector -> send the data, a merge of
        # all the vectors we got from the clients or the message 'stop' the
        # signal to the client that we converged.

        normDiff = 0
        normGradW = 0
        normPrecW = 0
        if (request.poids == 'pret'):
            vector = std.datadict2Sstr(trainingSet) + "<depre>" + str(l) + "<samples>" + str(numSamples)
        elif (request.poids == 'getw0'):
            vector = std.dict2str(w0)
        else :
            gradParam = std.mergeSGD(self.vectors)
            gradParam = std.sparse_mult(self.step, gradParam)
            vector = std.sparse_vsous(self.oldParam, gradParam)

            ######## NORMALIZATION OF THE VECTOR OF PARAMETERS #########
            normW = math.sqrt(std.sparse_dot(vector,vector))
            vector = std.sparse_mult(1./normW,vector)

            ############################################################
            diff = std.sparse_vsous(self.oldParam,vector)
            normDiff = math.sqrt(std.sparse_dot(diff,diff))
            normGradW = math.sqrt(std.sparse_dot(vector,vector))
            normPrecW = math.sqrt(std.sparse_dot(self.oldParam, self.oldParam))
            if ((self.epoch > nbMaxCall) or (normGradW <= c2*normGW0) or (normDiff <= c1*normPrecW)):
                self.paramVector = vector
                vector = 'stop'
            else:
                vector = std.dict2str(vector)

        ######################################################################

        ######################################################################
        # Section 3 : wait that all the threads pass the computation area, and
        # store the new computed vector.

        realComputation = (request.poids != 'pret') and (request.poids != 'getw0') and (vector != 'stop')

        self.iterator -= 1

        self.exit_condition = (self.iterator == 0)
        waiting.wait(lambda : self.exit_condition)

        if (realComputation):
            self.oldParam = std.str2dict(vector)

        ######################################################################

        ###################### PRINT OF THE CURRENT STATE ######################
        ##################### AND DO CRITICAL MODIFICATIONS ####################
        if (threading.current_thread().name == self.printerThreadName):
            std.printTraceRecData(self.epoch, vector, self.paramVector, self.testingErrors, self.trainingErrors, normDiff, normGradW, normPrecW, normGW0, realComputation, self.oldParam,trainingSet, testingSet, nbTestingData, nbExamples, c1, c2, l)
            self.merged.append(self.oldParam)
            self.epoch += 1
            self.step *= 0.9


            dataTest = trainingSet[9]
            label = dataTest.get(-1,0)
            example = std.take_out_label(dataTest)
            print("label = " + str(label))
            print("SVM says = " + str(std.sparse_dot(self.oldParam,example)))


            ############################### END OF PRINT ###########################



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
