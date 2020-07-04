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
        read, frame = self.cap.read()
        if read:
            return frame[:,::-1,:] if self.input=='cam' else frame
        else:
            print('Reached the end of the input stream')
            return None

#    def next_frame(self):
#        """
#        Yields a new frame read from the source (if available). 
#        In case the source is a webcam, the frame will be automatically reflected.
#        """
#
#        while True:
#            read, frame = self.cap.read()
#            if read:                                
#                # What is left from our point of view is right from the camera viewpoint
#                yield frame[:,::-1,:] if self.input=='cam' else frame
#            else:
#                print('Reached the end of the input stream')
#                break

    #def next_batch(self):
    #    '''
    #    Returns the next image from either a video file or webcam.
    #    If input_type is 'image', then it returns the same image.
    #    '''
    #    while True:
    #        for _ in range(10):
    #            _, frame=self.cap.read()
    #        yield frame


    def close(self):
        """
        Closes the VideoCapture.
        """
        self.cap.release()

