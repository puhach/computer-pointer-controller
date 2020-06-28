from generic_model import GenericModel
import helpers

class FaceDetector(GenericModel):
    """
    A class for the Face Detection Model.
    """

    def __init__(self, device='CPU', extensions=None):
        """
        Initializes a new instance of the face detection model.
        """

        super().__init__(model_name='../models/intel/face-detection-retail-0005/FP32/face-detection-retail-0005', device=device, extensions=extensions)

        self.input_shape = self.network.inputs[self.input_name].shape


    def detect(self, image, confidence=0.5):
        """
        Returns a bounding box of the face if detected. Otherwise, returns None.
        """
        image_preprocessed = self._preprocess_input(image, width=self.input_shape[3], height=self.input_shape[2])
        outputs = super()._infer(image_preprocessed)
        box = self._get_bounding_box(outputs, confidence)
        #return self._fit(box, image)
        return helpers.fit(box, image)
        


    def extract(self, image, confidence=0.5):
        """
        Crops the image of face from the original image. 
        """
        box = self.detect(image, confidence)
        #return self._crop(image, box)
        return helpers.crop(image, box), box


    def _get_bounding_box(self, detections, confidence):
        """
        An auxiliary function which finds the first face bounding box with sufficient detection confidence.
        """
        for detection in detections[0,0,:,:]:
            if detection[2] >= confidence:
                return detection[3:]
        
        return None

    #def _preprocess_input(self, frame):
    #    '''
    #    Before feeding the data into the model for inference,
    #    you might have to preprocess it. This function is where you can do that.
    #    '''
    #    w = self.input_shape[3]
    #    h = self.input_shape[2]
    #    #c = self.input_shape[1]
    #    
    #    #frames_preprocessed = np.empty(shape=(len(batch), c, h, w))
    #    #frame_buf = np.empty(shape=(h, w, c))
    #
    #    #for index, frame in enumerate(batch):
    #    #    cv2.resize(src=frame, dst=frame_buf, dsize=(w, h))        
    #    #    frames_preprocessed[index] = np.moveaxis(frame_buf, -1, 0)  # HWC -> CHW
    #
    #    #return frames_preprocessed
    #
    #    frame_resized = cv2.resize(src=frame, dsize=(w, h))
    #    frame_resized = np.moveaxis(frame_resized, -1, 0)[None,...] # HWC -> BCHW
    #    return frame_resized
            

    #def _preprocess_output(self, outputs, frame, confidence):
    #    '''
    #    Before feeding the output of this model to the next model,
    #    you might have to preprocess the output. This function is where you can do that.
    #    '''
    #    # The net outputs a blob with shape: [1, 1, N, 7], where N is the number of detected bounding boxes. 
    #    # For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
    #    h,w = frame.shape[:2]   # H, W, C
    #    for detection in outputs[0,0,:,:]:
    #        if detection[2] >= confidence:
    #            x_min, y_min, x_max, y_max = detection[3:]
    #            #res = cv2.rectangle(frame, (int(x_min*w),int(y_min*h)), (int(x_max*w), int(y_max*h)), color=(0,255,0))
    #            return frame[int(y_min*h):int(y_max*h)+1, int(x_min*w):int(x_max*w)+1, :]
    #
    #    return None



