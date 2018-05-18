"""The Python implementation of the gRPC stochastic gradient descent server client."""

from __future__ import print_function


import grpc
import time

import route_guide_pb2
import route_guide_pb2_grpc

import sgd
import sparseToolsDict as std
import sys
import math



##############################################################################
# Contact the server to get the data set, the departure vector to compute the
# SGD until the server decides of the convergence of the algorithm.

# Input :
#       -stub : the stub associated with our server to call its methods.

# Remarks :
#       -the SGD that we implemented uses a constant descent step. It could be
#       improved for the milestone 2. The one we use is quite little, but we
#       choose to do more iterations but get a precise enough result, and avoid
#       to oscillate between the level curves of the SVM cost function.
#       -
##############################################################################


def guide_get_feature(stub):

    # A variable to count the number of iteration of the client, which must coincide with the epoch in the server.
    it = 1

    # We make a first call to the server to get the data : after that call, vect is the data set. Then we store it.
    vect = stub.GetFeature(route_guide_pb2.Vector(poids="pret"))
    dataInfo = vect.poids.split("<depre>")
    vect.poids = dataInfo[0]

    computeInfo = dataInfo[1].split("<samples>")

    # The depreciation of the SVM norm cost
    l = float(computeInfo[0])

    # Number of samples in each subtraining set
    numSamples = float(computeInfo[1])

    # We convert the set of data in the good format.
    dataSampleSet = std.str2datadict(vect.poids)

    # This second call serves to get the departure vector.
    vect = stub.GetFeature(route_guide_pb2.Vector(poids="getw0"))


    while (vect.poids != 'stop'):

        print("iteration : " + str(it))

        # We save the vector on which we base the computations
        wt = std.str2dict(vect.poids)

        # Gradient descent on the sample.
        nw = sgd.descent(dataSampleSet, std.str2dict(vect.poids), numSamples, l)

        # Normalization of the vector of parameters
        normnW = math.sqrt(std.sparse_dot(nw,nw))
        nw = std.sparse_mult(1/normnW,nw)

        # The result is sent to the server.
        strVect = std.dict2str(nw)

        vect.poids = strVect + "<bytes>" + str(sys.getsizeof(strVect))
        vect = stub.GetFeature(route_guide_pb2.Vector(poids=vect.poids))

        it += 1

        #time.sleep(1.7)

    print(vect)





def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = route_guide_pb2_grpc.RouteGuideStub(channel)
    print("-------------- GetFeature --------------")
    guide_get_feature(stub)


if __name__ == '__main__':
    run()
