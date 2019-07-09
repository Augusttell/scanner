import time
from concurrent import futures
# import matplotlib.pyplot as plt
import cv2
import grpc
import numpy as np
import image_service_pb2 as image_service_pb2
import image_service_pb2_grpc as image_service_pb2_grpc

sleep = 60 * 60 * 24


class gRPCServer(image_service_pb2_grpc.ImageServiceServicer):
    def __init__(self):
        print('initialization')

    def process_image(self, request_iterator, context):
        for req in request_iterator:
            print(req.name)
            self.segmenting_image(req.image)
            yield image_service_pb2.ImageResponse(name=req.name, price=np.random.randint(100))

    def segmenting_image(self, im):
#         print(im) # do some image processing
        im = np.frombuffer(im, dtype="float32")
        print(im.shape)
#         plt.imshow(im)
#         plt.show()
#         cv2.imshow('image',im)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    image_service_pb2_grpc.add_ImageServiceServicer_to_server(
        gRPCServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(sleep)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
