import time
from concurrent import futures
# import matplotlib.pyplot as plt

import PIL

import cv2
import grpc
import numpy as np
import image_service_pb2 as image_service_pb2
import image_service_pb2_grpc as image_service_pb2_grpc
from model import Classifier



sleep = 60 * 60 * 24


#-m mjolkny.jpg -t image -show edited -bwl 70 -bwr 15 -bht 0 -bhb 40 -binarization yes -greyscale yes -b1 145 -b2 255 -morph erosion -morphH 3 -morphW 3 -blur no -oem 2 -psm 3


class gRPCServer(image_service_pb2_grpc.ImageServiceServicer):
    def __init__(self):
        print('initializing server and model...')
#         self.model = Classifier()

    def process_image(self, request_iterator, context):
        for req in request_iterator:
            print(req.name)
            im_vec = np.frombuffer(req.image, dtype=np.uint8)
            im = im_vec.reshape(req.width, req.height, req.channels)
            self.segmenting_image(im)

            classifier = Classifier()
            pred = classifier.predict(im)
            print(pred)

            yield image_service_pb2.ImageResponse(name=req.name, price=np.random.randint(100))

    def segmenting_image(self, im):
#         print(im) # do some image processing
        print(im.shape)
#         plt.imshow(im)
#         plt.show()
#         cv2.imshow('image',im)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        
def serve():
    server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),  
            options=[
                        ('grpc.max_send_message_length', 100 * 1024 * 1024),
                        ('grpc.max_receive_message_length', 100 * 1024 * 1024)
                    ]
            )
#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    image_service_pb2_grpc.add_ImageServiceServicer_to_server(gRPCServer(), server)
    
    server.add_insecure_port('[::]:50051')
    
    server.start()
    try:
        while True:
            time.sleep(sleep)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
