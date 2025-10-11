import grpc
import numpy as np
import cv2
import os
from ..protos import siphon_service_pb2, siphon_service_pb2_grpc
from time import sleep


class SiphonClient:
    """
    Client for the Siphon service.
    """
    def __init__(self, host='localhost:50051', max_receive_message_length=100 * 1024 * 1024, max_send_message_length=100 * 1024 * 1024):
        """
        Args:
            host: string, host of the server
            max_receive_message_length: int, maximum length of the receive message
            max_send_message_length: int, maximum length of the send message
        """
        self.channel = grpc.insecure_channel(host, options=[
            ('grpc.max_receive_message_length', max_receive_message_length),
            ('grpc.max_send_message_length', max_send_message_length)
        ])
        self.stub = siphon_service_pb2_grpc.SiphonServiceStub(self.channel)
 
    
    def send_key(self, keys, hold_time, delay_time=0):
        """
        Send a key to the server.
        Args:
            keys: list of strings, keys to press, e.g., ['w', 'space']
            hold_time: string, time to hold the key in milliseconds
            delay_time: string, time to delay between keys in milliseconds
        """
        request = siphon_service_pb2.InputKeyRequest(keys=keys, hold_ms=hold_time, delay_ms=delay_time)
        return self.stub.InputKey(request)
    
    def get_attribute(self, attributeName):
        """
        Read memory value for a single attribute.
        Args:
            attributeName: string, name of the attribute
        """
        request = siphon_service_pb2.GetSiphonRequest(attributeName=attributeName)
        response = self.stub.GetAttribute(request)
        
        # Handle oneof value field
        if response.HasField('int_value'):
            return response.int_value
        elif response.HasField('float_value'):
            return response.float_value
        elif response.HasField('array_value'):
            return response.array_value
        else:
            return None

    def set_attribute(self, attributeName, value):
        """
        Set the value of an attribute.
        Args:
            attributeName: string, name of the attribute
            value: int, float, or bytes - the value to set
        """
        request = siphon_service_pb2.SetSiphonRequest(attributeName=attributeName)
        
        # Handle oneof value field based on type
        if isinstance(value, int):
            request.int_value = value
        elif isinstance(value, float):
            request.float_value = value
        elif isinstance(value, bytes):
            request.array_value = value
        else:
            raise ValueError(f"Unsupported value type: {type(value)}. Must be int, float, or bytes.")
        
        return self.stub.SetAttribute(request)
    
    def get_frame(self):
        """
        Get current frame as numpy array.
        """
        request = siphon_service_pb2.CaptureFrameRequest()
        response = self.stub.CaptureFrame(request)
        # Server sends BGRA format (32 bits per pixel = 4 bytes per pixel)
        frame = np.frombuffer(response.frame, dtype=np.uint8)
        frame = frame.reshape(response.height, response.width, 4)  # BGRA format
        
        # Convert BGRA to BGR for OpenCV (remove alpha channel)
        frame = frame[:, :, :3]  # Remove alpha channel
        
        # save_path = os.path.join(os.path.dirname(__file__), 'frame.png')
        # cv2.imwrite(save_path, frame)
        return frame

    def move_mouse(self, delta_x, delta_y, steps=1):
        """
        Move the mouse by the given delta.
        """
        request = siphon_service_pb2.MoveMouseRequest(delta_x=delta_x, delta_y=delta_y, steps=steps)
        return self.stub.MoveMouse(request)

    def close(self):
        """
        Close the channel.
        """
        self.channel.close()

        


    