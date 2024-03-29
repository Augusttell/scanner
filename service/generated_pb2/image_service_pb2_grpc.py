# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import image_service_pb2 as image__service__pb2


class ImageServiceStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.process_image = channel.unary_unary(
        '/svinn_package.ImageService/process_image',
        request_serializer=image__service__pb2.ImageRequest.SerializeToString,
        response_deserializer=image__service__pb2.ImageResponse.FromString,
        )


class ImageServiceServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def process_image(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_ImageServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'process_image': grpc.unary_unary_rpc_method_handler(
          servicer.process_image,
          request_deserializer=image__service__pb2.ImageRequest.FromString,
          response_serializer=image__service__pb2.ImageResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'svinn_package.ImageService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
