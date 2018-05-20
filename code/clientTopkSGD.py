"""The Python implementation of the gRPC stochastic gradient descent server client."""

from __future__ import print_function


import grpc

import route_guide_pb2
import route_guide_pb2_grpc

import sgd
import sparseToolsDict as std
import sys


hypPlace = 4



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

    # We convert the set of data in the good format.
    data = vect.poids.split("<samples>")
    dataSampleSet = std.str2datadict(data[0])
    data2 = data[1].split("<#compo>")
    numSamples = int(data2[0])
    nbCompo = int(data2[1])


    # This second call serves to get the departure vector.
    vect = stub.GetFeature(route_guide_pb2.Vector(poids="getw0"))
    info = vect.poids.split("<<||>>")
    vect.poids = info[0]
    step = float(info[1])


    # Vector of the rests
    m = {}

    while (vect.poids != 'stop'):

        print("iteration : " + str(it))

        # Gradient descent on the sample.
        nw = sgd.descent(dataSampleSet, std.str2dict(vect.poids), numSamples, step)

        # Updating of m
        stepedGradient = std.sparse_mult(step,nw)
        m = std.sparse_vsum(m,stepedGradient)

        # Get the nbCompo biggest values in m, put it in g and remove them of m
        g = std.infiniteNormInd(m,nbCompo)

        # The result is sent to the server.
        strVect = std.dict2str(g)
        vect.poids = strVect + "<bytes>" + str(sys.getsizeof(strVect))
        vect = stub.GetFeature(route_guide_pb2.Vector(poids=vect.poids))

        if (vect.poids != 'stop'):
            info = vect.poids.split("<<||>>")
            vect.poids = info[0]
            step = float(info[1])

        it += 1

    print(vect)





def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = route_guide_pb2_grpc.RouteGuideStub(channel)
    print("-------------- GetFeature --------------")
    guide_get_feature(stub)


if __name__ == '__main__':
    run()
