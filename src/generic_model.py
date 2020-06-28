from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np

class GenericModel:

    core = IECore()

    """
    The parent class for various object detection and recognition models.
    """
    def __init__(self, model_name, device='CPU', extensions=None):
        """
        Initializes the generic model.
        """

        # OpenVINO 2020.1 loads extensions automatically
        if extensions and device.lower() == 'cpu':
            self.core.add_extensions(extensions)


        self.model_xml = model_name + '.xml'
        self.model_bin = model_name + '.bin'

        #self.network = self.core.read_network(model=self.model_xml, weights=self.model_bin)
        self.network = IENetwork(model=self.model_xml, weights=self.model_bin)

        # Check for unsupported layers
        supported_layers = set(self.core.query_network(network=self.network, device_name=device).keys())
        net_layers = set(self.network.layers.keys())
        unsupported_layers = net_layers.difference(supported_layers)
        if unsupported_layers:
            raise Exception('Unsupported layers: ' + ','.join(unsupported_layers))


        self.input_name = next(iter(self.network.inputs))
        self.output_name = next(iter(self.network.outputs))

        self.exe_network = self.core.load_network(network=self.network, device_name=device, num_requests=1)


    def _infer(self, image):
        """
        Feeds an input image to the model for inference.
        """
        input_dict = { self.input_name : image }
        output_dict = self.exe_network.infer(input_dict)
        return output_dict[self.output_name]

    
    def _preprocess_input(self, image, width, height):
        """
        Performs default preprocessing of the input:
        1) Resizes the image to the desired resolution
        2) Converts the channel order from HWC to CHW
        3) Adds the batch dimension
        """

        frame_resized = cv2.resize(src=image, dsize=(width, height))
        frame_resized = np.moveaxis(frame_resized, -1, 0)[None,...] # HWC -> BCHW
        return frame_resized
    
