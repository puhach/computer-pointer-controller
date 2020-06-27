from generic_model import GenericModel
import cv2 # for drawing landmarks


#class FacialLandmarkDetector(GenericModel):
class EyeDetector(GenericModel):
    """
    A class for eye detection and extraction.
    """
    def __init__(self, device='CPU', extensions=None):
        """
        Initializes a facial landmark detection model instance.
        """

        super().__init__(model_name='../models/intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002', device=device, extensions=extensions)

        self.input_shape = self.network.inputs[self.input_name].shape

    def detect(self, image):
        image_preprocessed = self._preprocess_input(image, self.input_shape[3], self.input_shape[2])
        landmarks = super().detect(image_preprocessed)
        #testimg = self._draw_landmarks(image, landmarks)
        left_eye_box, right_eye_box = self._get_eye_boxes(landmarks)        
        return self._fit(left_eye_box, image), self._fit(right_eye_box, image)
        #return self._adjust(self.fit(left_eye_box, image), True), self._adjust(self.fit(right_eye_box, image), False), testimg

    def extract(self, image):
        left_eye_box, right_eye_box = self.detect(image)
        return self._crop(image, left_eye_box), self._crop(image, right_eye_box)

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

    #def preprocess_input(self, image):
    #    '''
    #    Before feeding the data into the model for inference,
    #    you might have to preprocess it. This function is where you can do that.
    #    '''
    #    raise NotImplementedError

    def _get_eye_boxes(self, landmarks):
        """
        Finds the bounding boxes of eyes given the coordinates of facial landmarks.        
        """
        
        # The landmarks contain a row-vector of 70 floating point values for 35 landmarks' normed coordinates in the form (x0, y0, x1, y1, ..., x34, y34)

        # [Left Eye] p0, p1: corners of the eye, located on the boundary of the eyeball and the eyelid.
        # [Right Eye] p2, p3: corners of the eye, located on the boundary of the eyeball and the eyelid.        
        # [Left Eyebrow] p12: starting point of the upper boundary of the eyebrow; p13: mid-point of the upper arc of the eyebrow; p14: ending point of the upper boundary of the eyebrow.
        # [Right Eyebrow] p15: starting point of the upper boundary of the eyebrow; p16: mid-point of the upper arc of the eyebrow; p17: ending point of the upper boundary of the eyebrow.

        # Left eye
        xmin_l = landmarks[0, 12*2]  # p12: starting point of the upper boundary of the eyebrow        
        #xmax_l = landmarks[0, 0*2]  # p0: corner of the eye, located on the boundary of the eyeball and the eyelid
        xmax_l = landmarks[0, 14*2]  # p14: ending point of the upper boundary of the eyebrow
        ymin_l = landmarks[0, 13*2+1]   # p13: mid-point of the upper arc of the eyebrow
        #ymax_l = ymin_l + abs(xmax_l - xmin_l) # make a square bounding box
        #ymax_l = landmarks[0, 1*2+1] # p1: corner of the eye, located on the boundary of the eyeball and the eyelid
        # p0: corner of the eye, located on the boundary of the eyeball and the eyelid; p14: ending point of the upper boundary of the eyebrow
        ymax_l = self._clamp(2*landmarks[0, 0*2+1] - landmarks[0, 14*2+1], 0, 1)

        # Right eye
        xmin_r = landmarks[0, 15*2] # p15: starting point of the upper boundary of the eyebrow
        xmax_r = landmarks[0, 17*2] # p17: ending point of the upper boundary of the eyebrow
        ymin_r = landmarks[0, 16*2+1]   # p16: mid-point of the upper arc of the eyebrow
        #ymax_r = ymin_r + abs(xmax_r - xmax_l)  # make a square bounding box
        #ymax_r = landmarks[0, 3*2+1] # p3: corner of the eye, located on the boundary of the eyeball and the eyelid
        # p2: corners of the eye, located on the boundary of the eyeball and the eyelid; p15: starting point of the upper boundary of the eyebrow
        ymax_r = self._clamp(2*landmarks[0, 2*2+1] - landmarks[0, 15*2+1], 0, 1)

        return (xmin_l, ymin_l, xmax_l, ymax_l), (xmin_r, ymin_r, xmax_r, ymax_r)


    # TODO: consider moving this function to helpers
    def _clamp(self, value, range_min, range_max):
        return max(range_min, min(range_max, value))

    #def _adjust(self, box, is_left):
    #    xmin, ymin, xmax, ymax = box
    #
    #    if xmax < xmin:
    #        xmin, xmax = xmax, xmin
    #
    #    if ymax < ymin:
    #        ymin, ymax = ymax, ymin
    #
    #    if xmax-xmin < 0.7*(ymax-ymin):
    #        print('Eye width is too small')
    #        if is_left: # xmin may be occluded for the left eye
    #            xmin = max(0, xmax - ymax + ymin)
    #        else:   # xmax may be occluded for the right eye
    #            xmax = xmin + ymax - ymin
    #
    #    elif ymax-ymin < 0.7*(xmax-xmin):
    #        print('Eye height is too small')
    #        ymax = ymin + xmax-xmin
    #
    #    return (xmin, ymin, xmax, ymax)



    def _draw_landmarks(self, image, landmarks):
        """
        An auxiliary function for debugging.
        """
        h,w = image.shape[:-1]
        res_img = image.copy()  
        for i in range(0, landmarks.shape[1], 2):
            x = int(w*landmarks[0, i])
            y = int(h*landmarks[0, i+1])
            res_img = cv2.circle(res_img, (x,y), radius=2, color=(0,255,0))

        return res_img