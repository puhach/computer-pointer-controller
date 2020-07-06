from generic_model import GenericModel

class HeadPoseEstimator(GenericModel):
    """
    A class for head pose estimation.
    """

    def __init__(self, precision, concurrency, device='CPU', extensions=None):
        """
        Initializes a new head pose estimation model instance.
        """

        super().__init__(
            model_name=f'../models/intel/head-pose-estimation-adas-0001/{precision}/head-pose-estimation-adas-0001', 
            concurrency=concurrency, device=device, extensions=extensions)

        self.input_shape = self.network.inputs[self.input_name].shape

        self.yaw_name = "angle_y_fc"
        self.pitch_name =  "angle_p_fc"
        self.roll_name = "angle_r_fc"


    def feed_input(self, face_image):
        """
        Takes in the face image, preprocesses it, and feeds to the model for inference.
        Depending on the inference mode, the call may be blocking or not.
        """
        face_image_preprocessed = self._preprocess_input(face_image, 
            width=self.input_shape[3], height=self.input_shape[2])
        super().feed_input(face_image_preprocessed)

    def consume_output(self, wait):
        """
        Retrieves the head pose angles from the estimation results. Returns a tuple. 
        The first value indicates whether the result was retrieved. The second item
        is a vector containing the yaw, pitch, and roll angles in degrees. 
        The wait parameter specifies whether the function has to wait for the current
        inference request to finish in case no result is available at the moment of the call.
        """
        consumed, output_dict = super().consume_output_dict(wait)
        if consumed and output_dict:
            head_pose = ( output_dict[self.yaw_name][0,0], output_dict[self.pitch_name][0,0], output_dict[self.roll_name][0,0] )
        else:
            head_pose = None
        return consumed, head_pose
