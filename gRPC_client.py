import random
import time

import grpc
import numpy as np
import cv2

import image_service_pb2 as image_service_pb2
import image_service_pb2_grpc as image_service_pb2_grpc

files = ['data/cat1','data/cat2', 'data/cat3']

class gRPCClient():
    def __init__(self):
        channel = grpc.insecure_channel('localhost:50051')
        self.stub = image_service_pb2_grpc.ImageServiceStub(channel)

    def sendRequest(self, req): # Send requests
        return self.stub.process_image(req)


def generateRequests():
    reqs = []
    for name, file in zip(['asta','malte','nisse'],files):
        im = cv2.imread(file+'.png').tostring() 
        reqs.append(image_service_pb2.ImageRequest(name=name, image = im))
    for req in reqs:
        yield req
        time.sleep(random.uniform(1, 3))


def main():
    client = gRPCClient()

    response = client.sendRequest(generateRequests())

    for re in response:
        print(re)


if __name__ == '__main__':
    main()
