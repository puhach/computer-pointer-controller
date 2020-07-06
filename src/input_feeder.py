import cv2
from numpy import ndarray

class InputFeeder:

    def __init__(self, input):
        """
        Creates an input feed from an image or a video file. In order to use a webcam feed, pass in 'cam' as input.        
        """
                        
        self.input = input.lower()

        if self.input == 'cam':            
            self.cap = cv2.VideoCapture(0)
        else:
            # Restore the original letter case (important for file names)
            self.input = input

            # cv2.VideoCapture can handle both images and video files
            self.cap = cv2.VideoCapture(self.input)

        if not self.cap.isOpened():
            raise Exception('Failed to open the input: ' + self.input)
        
    
    def read_next(self):
        """
        Reads the next frame from the input stream. If the next frame is not available, returns None.
        In case the source is a webcam, the frame will be automatically reflected.
        """
        read, frame = self.cap.read()
        if read:
            return frame[:,::-1,:] if self.input=='cam' else frame
        else:
            return None


    def close(self):
        """
        Closes the VideoCapture.
        """
        self.cap.release()

