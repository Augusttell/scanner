import random
import time

import grpc
import numpy as np
import cv2

import image_service_pb2 as image_service_pb2
import image_service_pb2_grpc as image_service_pb2_grpc


class gRPCClient():
    def __init__(self):
        options=[
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024)
                ]
        channel = grpc.insecure_channel('localhost:50051',options = options)
        self.stub = image_service_pb2_grpc.ImageServiceStub(channel)

    def sendRequest(self, req): # Send requests
        return self.stub.process_image(req)


def generateRequests():
    name = 'milk'
    file = '../classifier/mjolkny.jpg'
    im = cv2.imread(file)
    print(im.shape)
    w,h,c = im.shape
    byte_im = im.tostring()
    req = image_service_pb2.ImageRequest(name=name, 
                                         image = byte_im,
                                         width = w,
                                         height= h,
                                         channels = c)
    return req

def main():
    client = gRPCClient()

    response = client.sendRequest(generateRequests())

    print(response)

if __name__ == '__main__':
    main()

