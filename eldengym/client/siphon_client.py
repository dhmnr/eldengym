import grpc
import numpy as np
from ..protos import siphon_service_pb2, siphon_service_pb2_grpc


class SiphonClient:
    """
    Client for the Siphon service.
    """

    def __init__(
        self,
        host="localhost:50051",
        max_receive_message_length=100 * 1024 * 1024,
        max_send_message_length=100 * 1024 * 1024,
    ):
        """
        Args:
            host: string, host of the server
            max_receive_message_length: int, maximum length of the receive message
            max_send_message_length: int, maximum length of the send message
        """
        self.channel = grpc.insecure_channel(
            host,
            options=[
                ("grpc.max_receive_message_length", max_receive_message_length),
                ("grpc.max_send_message_length", max_send_message_length),
            ],
        )
        self.stub = siphon_service_pb2_grpc.SiphonServiceStub(self.channel)

    def send_key(self, keys, hold_time, delay_time=0):
        """
        Send a key to the server.
        Args:
            keys: list of strings, keys to press, e.g., ['w', 'space']
            hold_time: string, time to hold the key in milliseconds
            delay_time: string, time to delay between keys in milliseconds
        """
        request = siphon_service_pb2.InputKeyTapRequest(
            keys=keys, hold_ms=hold_time, delay_ms=delay_time
        )
        return self.stub.InputKeyTap(request)

    def get_attribute(self, attributeName):
        """
        Read memory value for a single attribute.
        Args:
            attributeName: string, name of the attribute
        """
        request = siphon_service_pb2.GetSiphonRequest(attributeName=attributeName)
        response = self.stub.GetAttribute(request)

        # Handle oneof value field
        if response.HasField("int_value"):
            return response.int_value
        elif response.HasField("float_value"):
            return response.float_value
        elif response.HasField("array_value"):
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
            raise ValueError(
                f"Unsupported value type: {type(value)}. Must be int, float, or bytes."
            )

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
        request = siphon_service_pb2.MoveMouseRequest(
            delta_x=delta_x, delta_y=delta_y, steps=steps
        )
        return self.stub.MoveMouse(request)

    def toggle_key(self, key, toggle):
        """
        Press or release a key.
        Args:
            key: string, key to toggle, e.g., 'w'
            toggle: bool, True to press, False to release
        """
        request = siphon_service_pb2.InputKeyToggleRequest(key=key, toggle=toggle)
        return self.stub.InputKeyToggle(request)

    def execute_command(
        self,
        command,
        args=None,
        working_directory="",
        timeout_seconds=30,
        capture_output=True,
    ):
        """
        Execute a command on the server.
        Args:
            command: string, command to execute
            args: list of strings, command arguments
            working_directory: string, working directory for the command
            timeout_seconds: int, timeout in seconds
            capture_output: bool, whether to capture output
        Returns:
            ExecuteCommandResponse with fields: success, message, exit_code,
            stdout_output, stderr_output, execution_time_ms
        """
        if args is None:
            args = []
        request = siphon_service_pb2.ExecuteCommandRequest(
            command=command,
            args=args,
            working_directory=working_directory,
            timeout_seconds=timeout_seconds,
            capture_output=capture_output,
        )
        return self.stub.ExecuteCommand(request)

    def set_process_config(self, process_name, process_window_name, attributes):
        """
        Set the process configuration.
        Args:
            process_name: string, name of the process
            process_window_name: string, window name of the process
            attributes: list of dicts with keys: name, pattern, offsets, type, length, method
                Example: [
                    {
                        'name': 'health',
                        'pattern': 'AB CD EF',
                        'offsets': [0x10, 0x20],
                        'type': 'int',
                        'length': 4,
                        'method': ''
                    }
                ]
        """
        request = siphon_service_pb2.SetProcessConfigRequest(
            process_name=process_name, process_window_name=process_window_name
        )

        for attr in attributes:
            proto_attr = request.attributes.add()
            proto_attr.name = attr.get("name", "")
            proto_attr.pattern = attr.get("pattern", "")
            proto_attr.offsets.extend(attr.get("offsets", []))
            proto_attr.type = attr.get("type", "")
            proto_attr.length = attr.get("length", 0)
            proto_attr.method = attr.get("method", "")

        return self.stub.SetProcessConfig(request)

    def initialize_memory(self):
        """
        Initialize the memory subsystem.
        Returns:
            InitializeMemoryResponse with fields: success, message, process_id
        """
        request = siphon_service_pb2.InitializeMemoryRequest()
        return self.stub.InitializeMemory(request)

    def initialize_input(self, window_name=""):
        """
        Initialize the input subsystem.
        Args:
            window_name: string, window name to target (optional)
        Returns:
            InitializeInputResponse with fields: success, message
        """
        request = siphon_service_pb2.InitializeInputRequest(window_name=window_name)
        return self.stub.InitializeInput(request)

    def initialize_capture(self, window_name=""):
        """
        Initialize the capture subsystem.
        Args:
            window_name: string, window name to target (optional)
        Returns:
            InitializeCaptureResponse with fields: success, message, window_width, window_height
        """
        request = siphon_service_pb2.InitializeCaptureRequest(window_name=window_name)
        return self.stub.InitializeCapture(request)

    def get_server_status(self):
        """
        Get the server initialization status.
        Returns:
            GetServerStatusResponse with fields: success, message, config_set,
            memory_initialized, input_initialized, capture_initialized,
            process_name, window_name, process_id
        """
        request = siphon_service_pb2.GetServerStatusRequest()
        return self.stub.GetServerStatus(request)

    def close(self):
        """
        Close the channel.
        """
        self.channel.close()
