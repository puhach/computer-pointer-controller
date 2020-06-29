from generic_model import GenericModel
import helpers

class EyeDetector(GenericModel):
    """
    A class for eye detection and extraction.
    """
    def __init__(self, precision, device='CPU', extensions=None):
        """
        Initializes a facial landmark detection model instance.
        """

        super().__init__(
            model_name=f'../models/intel/facial-landmarks-35-adas-0002/{precision}/facial-landmarks-35-adas-0002', 
            device=device, 
            extensions=extensions)

        self.input_shape = self.network.inputs[self.input_name].shape

    def detect(self, face_image):
        """
        Takes in a face image and returns bounding boxes of the left and right eyes.
        """

        if face_image is None or face_image.size<1:
            print('Skipping eye detection: the face image is empty')
            return None, None

        image_preprocessed = self._preprocess_input(face_image, self.input_shape[3], self.input_shape[2])
        landmarks = super()._infer(image_preprocessed)
        left_eye_box, right_eye_box = self._get_eye_boxes(landmarks)        
        return helpers.fit(left_eye_box, face_image), helpers.fit(right_eye_box, face_image)

    def extract(self, face_image):
        """
        Extracts the images of eyes from a face image.
        """
        left_eye_box, right_eye_box = self.detect(face_image)
        return helpers.crop(face_image, left_eye_box), helpers.crop(face_image, right_eye_box)



    def _get_eye_boxes(self, landmarks):
        """
        Finds the bounding boxes of eyes given the coordinates of facial landmarks.        
        """
        
        # The landmarks contain a row-vector of 70 floating point values for 35 landmarks' 
        # normed coordinates in the form (x0, y0, x1, y1, ..., x34, y34)

        # Left eye
        xmin_l = landmarks[0, 12*2]  # p12: starting point of the upper boundary of the eyebrow        
        #xmax_l = landmarks[0, 0*2]  # p0: corner of the eye, located on the boundary of the eyeball and the eyelid
        xmax_l = landmarks[0, 14*2]  # p14: ending point of the upper boundary of the eyebrow
        ymin_l = landmarks[0, 13*2+1]   # p13: mid-point of the upper arc of the eyebrow
        #ymax_l = ymin_l + abs(xmax_l - xmin_l) # make a square bounding box
        #ymax_l = landmarks[0, 1*2+1] # p1: corner of the eye, located on the boundary of the eyeball and the eyelid
        # p0: corner of the eye, located on the boundary of the eyeball and the eyelid; p14: ending point of the upper boundary of the eyebrow
        ymax_l = helpers.clamp(2*landmarks[0, 0*2+1] - landmarks[0, 14*2+1], 0, 1)

        # Right eye
        xmin_r = landmarks[0, 15*2] # p15: starting point of the upper boundary of the eyebrow
        xmax_r = landmarks[0, 17*2] # p17: ending point of the upper boundary of the eyebrow
        ymin_r = landmarks[0, 16*2+1]   # p16: mid-point of the upper arc of the eyebrow
        #ymax_r = ymin_r + abs(xmax_r - xmax_l)  # make a square bounding box
        #ymax_r = landmarks[0, 3*2+1] # p3: corner of the eye, located on the boundary of the eyeball and the eyelid
        # p2: corners of the eye, located on the boundary of the eyeball and the eyelid; p15: starting point of the upper boundary of the eyebrow
        ymax_r = helpers.clamp(2*landmarks[0, 2*2+1] - landmarks[0, 15*2+1], 0, 1)

        return (xmin_l, ymin_l, xmax_l, ymax_l), (xmin_r, ymin_r, xmax_r, ymax_r)



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

