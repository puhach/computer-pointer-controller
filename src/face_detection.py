from generic_model import GenericModel
import helpers

class FaceDetector(GenericModel):
    """
    A class for the Face Detection Model.
    """

    def __init__(self, precision, concurrency, device='CPU', extensions=None):
        """
        Initializes a new instance of the face detection model.
        """

        super().__init__(
            model_name=f'../models/intel/face-detection-retail-0005/{precision}/face-detection-retail-0005', 
            concurrency=concurrency,
            device=device, 
            extensions=extensions)

        self.input_shape = self.network.inputs[self.input_name].shape


    def feed_input(self, image):
        image_preprocessed = self._preprocess_input(image, width=self.input_shape[3], height=self.input_shape[2])
        super().feed_input(image_preprocessed)


    def consume_output(self, wait, confidence=0.5):
        consumed, detections = super().consume_output(wait)
        if consumed and detections is not None:
            face_box = self._get_bounding_box(detections, confidence=confidence)            
        else:
            face_box = None
        return consumed, face_box


    def preprocess_output(self, face_box, frame):
        if face_box is None or frame is None or frame.size < 1:
            return None, None
        else:
            face_box = helpers.fit(face_box, frame)
            face_img = helpers.crop(frame, face_box)
            return face_img, face_box


    def _get_bounding_box(self, detections, confidence):
        """
        An auxiliary function which finds the first face bounding box with sufficient detection confidence.
        """
        for detection in detections[0,0,:,:]:
            if detection[2] >= confidence:
                return detection[3:]
        
        return None

