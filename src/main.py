
import cv2
from input_feeder import InputFeeder
from face_detection import FaceDetector
from facial_landmarks_detection import EyeDetector


feed = InputFeeder('video', input_file='../bin/demo.mp4')
#feed = InputFeeder('image', input_file='./bin/image.png')
#feed = InputFeeder('cam')

feed.load_data()

faceDetector = FaceDetector(device='CPU', extensions=None)

eyeDetector = EyeDetector(device='CPU', extensions=None)

#for batch in feed.next_batch():
for frame in feed.next_frame():
    #print('batch:', batch.shape)
    face = faceDetector.extract(frame, confidence=0.5)
    left_eye, right_eye = eyeDetector.extract(face)
    cv2.imshow('face', face)
    cv2.imshow('left eye', left_eye)
    cv2.imshow('right eye', right_eye)
    cv2.waitKey()

feed.close()
