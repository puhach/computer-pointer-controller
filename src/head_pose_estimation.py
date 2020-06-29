from generic_model import GenericModel

class HeadPoseEstimator(GenericModel):
    """
    A class for head pose estimation.
    """

    def __init__(self, device='CPU', extensions=None):
        """
        Initializes a new head pose estimation model instance.
        """

        super().__init__(model_name='../models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001', device=device, extensions=extensions)

        self.input_shape = self.network.inputs[self.input_name].shape

        self.yaw_name = "angle_y_fc"
        self.pitch_name =  "angle_p_fc"
        self.roll_name = "angle_r_fc"



    def estimate(self, face_image):
        """
        Takes in a face image and returns the yaw, pitсh, and roll angles in degrees.
        """

        if face_image is None or face_image.size < 1:
            print('Skipping head pose estimation: the face image is empty')
            return None

        face_image_preprocessed = self._preprocess_input(face_image, width=self.input_shape[3], height=self.input_shape[2])
        #head_pose = self._infer(face_image_preprocessed)
        input_dict = { self.input_name : face_image_preprocessed }
        output_dict = self.exe_network.infer(input_dict)
        head_pose = ( output_dict[self.yaw_name][0,0], output_dict[self.pitch_name][0,0], output_dict[self.roll_name][0,0] )
        return head_pose