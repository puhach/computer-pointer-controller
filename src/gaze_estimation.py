from generic_model import GenericModel
import numpy as np
import math

class GazeEstimator(GenericModel):
    """
    A class for gaze direction estimation.
    """

    def __init__(self, precision, concurrency, device='CPU', extensions=None):
        """
        Initializes a new instance of the gaze direction estimation model.
        """
        super().__init__(
            model_name=f'../models/intel/gaze-estimation-adas-0002/{precision}/gaze-estimation-adas-0002', 
            concurrency=concurrency, device=device, extensions=extensions)

        self.left_eye_input_name = 'left_eye_image'
        self.left_eye_input_shape = self.network.inputs[self.left_eye_input_name].shape
        self.right_eye_input_name = 'right_eye_image'
        self.right_eye_input_shape = self.network.inputs[self.right_eye_input_name].shape
        self.head_pose_input_name = 'head_pose_angles'
        self.head_pos_input_shape = self.network.inputs[self.head_pose_input_name].shape

        self.output_name = 'gaze_vector'


    def feed_input(self, left_eye, right_eye, head_pose_angles):
        if left_eye is not None and right_eye is not None and left_eye.size>0 and right_eye.size>0 and head_pose_angles:
            left_eye_preprocessed = self._preprocess_input(left_eye, width=self.left_eye_input_shape[3], height=self.left_eye_input_shape[2])
            right_eye_preprocessed = self._preprocess_input(right_eye, width=self.right_eye_input_shape[3], height=self.right_eye_input_shape[2])
        
            input_dict = {
                self.left_eye_input_name : left_eye_preprocessed,
                self.right_eye_input_name : right_eye_preprocessed,
                self.head_pose_input_name : head_pose_angles
            }
        else:
            input_dict = None

        super().feed_input_dict(input_dict)


    def consume_output(self, wait):
        consumed, outputs = super().consume_output(wait)
        if consumed and outputs is not None:
            gaze_vector = tuple(outputs[0])
        else:
            gaze_vector = (0, 0, 0)
        return consumed, gaze_vector


