from generic_model import GenericModel
import numpy as np

class GazeEstimator(GenericModel):
    """
    A class for gaze direction estimation.
    """

    def __init__(self, device='CPU', extensions=None):
        """
        Initializes a new instance of the gaze direction estimation model.
        """
        super().__init__(model_name='../models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002', device=device, extensions=extensions)

        print(self.network.inputs)

        self.left_eye_input_name = 'left_eye_image'
        self.left_eye_input_shape = self.network.inputs[self.left_eye_input_name].shape
        self.right_eye_input_name = 'right_eye_image'
        self.right_eye_input_shape = self.network.inputs[self.right_eye_input_name].shape
        self.head_pose_input_name = 'head_pose_angles'
        self.head_pos_input_shape = self.network.inputs[self.head_pose_input_name].shape

        self.output_name = 'gaze_vector'


    def estimate(self, left_eye, right_eye, head_pose_angles):
        """
        The model takes three inputs: crop of left eye image, crop of right eye image, and three head pose angles – (yaw, pitch, and roll). 
        Returns a 3-D vector corresponding to the direction of a person’s gaze in a Cartesian coordinate system in which z-axis is directed from person’s eyes (mid-point between left and right eyes’ centers) to the camera center, 
        y-axis is vertical, and x-axis is orthogonal to both z,y axes so that (x,y,z) constitute a right-handed coordinate system.
        """
        left_eye_preprocessed = self._preprocess_input(left_eye, width=self.left_eye_input_shape[3], height=self.left_eye_input_shape[2])
        right_eye_preprocessed = self._preprocess_input(right_eye, width=self.right_eye_input_shape[3], height=self.right_eye_input_shape[2])
        #head_pose_angles_preprocessed = np.asarray(head_pose_angles)
     
        input_dict = {
            self.left_eye_input_name : left_eye_preprocessed,
            self.right_eye_input_name : right_eye_preprocessed,
            self.head_pose_input_name : head_pose_angles
        }
        
        output_dict = self.exe_network.infer(input_dict)
        gaze_vector = output_dict[self.output_name]
        return tuple(gaze_vector[0])


        