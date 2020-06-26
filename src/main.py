
import cv2
from input_feeder import InputFeeder
from face_detection import FaceDetector



feed = InputFeeder('video', input_file='./bin/demo.mp4')
#feed = InputFeeder('image', input_file='./bin/image.png')
#feed = InputFeeder('cam')

feed.load_data()

faceDetector = FaceDetector('./models/intel/face-detection-retail-0005/FP32/face-detection-retail-0005', device='CPU', extensions=None)

#for batch in feed.next_batch():
for frame in feed.next_frame():
    #print('batch:', batch.shape)
    output = faceDetector.detect(frame)
    cv2.imshow('face', output)
    cv2.waitKey()

feed.close()
