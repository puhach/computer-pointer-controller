import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class FaceDetector:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.core = IECore()
        if extensions and device.lower() == 'cpu':
            self.core.add_extensions(extensions)


        self.model_xml = model_name + '.xml'
        self.model_bin = model_name + '.bin'

        #self.network = self.core.read_network(model=self.model_xml, weights=self.model_bin)
        self.network = IENetwork(model=self.model_xml, weights=self.model_bin)

        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_name = next(iter(self.network.outputs))
        #print(self.network.inputs.items())
        #print(self.network.inputs['input.1'].shape)

        #print(self.network.outputs)

        # TODO: check unsupported layers

        self.exe_network = self.core.load_network(network=self.network, device_name=device, num_requests=1)


    def detect(self, frame, confidence=0.5):
        input_frame_preprocessed = self._preprocess_input(frame)
        input_dict = { self.input_name : input_frame_preprocessed }
        output = self.exe_network.infer(input_dict)
        return self._preprocess_output(output[self.output_name], frame, confidence)

    #def load_model(self):
    #    '''
    #    TODO: You will need to complete this method.
    #    This method is for loading the model to the device specified by the user.
    #    If your model requires any Plugins, this is where you can load them.
    #    '''
    #    raise NotImplementedError

    #def predict(self, image):
    #    '''
    #    TODO: You will need to complete this method.
    #    This method is meant for running predictions on the input image.
    #    '''
    #    raise NotImplementedError

    #def check_model(self):
    #    raise NotImplementedError

    def _preprocess_input(self, frame):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        w = self.input_shape[3]
        h = self.input_shape[2]
        #c = self.input_shape[1]
        
        #frames_preprocessed = np.empty(shape=(len(batch), c, h, w))
        #frame_buf = np.empty(shape=(h, w, c))

        #for index, frame in enumerate(batch):
        #    cv2.resize(src=frame, dst=frame_buf, dsize=(w, h))        
        #    frames_preprocessed[index] = np.moveaxis(frame_buf, -1, 0)  # HWC -> CHW

        #return frames_preprocessed

        frame_resized = cv2.resize(src=frame, dsize=(w, h))
        frame_resized = np.moveaxis(frame_resized, -1, 0)[None,...] # HWC -> BCHW
        return frame_resized
            

    def _preprocess_output(self, outputs, frame, confidence):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # The net outputs a blob with shape: [1, 1, N, 7], where N is the number of detected bounding boxes. 
        # For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        h,w = frame.shape[:2]   # H, W, C
        for detection in outputs[0,0,:,:]:
            if detection[2] >= confidence:
                x_min, y_min, x_max, y_max = detection[3:]
                #res = cv2.rectangle(frame, (int(x_min*w),int(y_min*h)), (int(x_max*w), int(y_max*h)), color=(0,255,0))
                return frame[int(y_min*h):int(y_max*h)+1, int(x_min*w):int(x_max*w)+1, :]

        return None



