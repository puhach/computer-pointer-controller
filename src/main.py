
import cv2
import math
from input_feeder import InputFeeder
from face_detection import FaceDetector
from facial_landmarks_detection import EyeDetector
from head_pose_estimation import HeadPoseEstimator
from gaze_estimation import GazeEstimator


def draw_vector(gaze_vector, face_image):
    gx,gy,gz = gaze_vector
    mag = math.sqrt(gx*gx + gy*gy + gz*gz) + 1e-4    
    gx *= 100/mag
    gy *= 100/mag    
    p1 = (face.shape[1]//2, face.shape[0]//2)
    p2 = (int(face.shape[1]/2+gx), int(face.shape[0]/2-gy))
    res_img = face_image.copy()
    #cv2.line(res, p1, p2, color=(0,255,0))
    #cv2.circle(res, p2, radius=5, color=(0,0,255))
    cv2.arrowedLine(res_img, p1, p2, color=(0,255,0))
    return res_img

feed = InputFeeder('video', input_file='../bin/demo.mp4')
#feed = InputFeeder('image', input_file='./bin/image.png')
#feed = InputFeeder('cam')

feed.load_data()

faceDetector = FaceDetector(device='CPU', extensions=None)
eyeDetector = EyeDetector(device='CPU', extensions=None)
headPoseEstimator = HeadPoseEstimator(device='CPU', extensions=None)
gazeEstimator = GazeEstimator(device='CPU', extensions=None)

#for batch in feed.next_batch():
for frame in feed.next_frame():
    #print('batch:', batch.shape)
    face = faceDetector.extract(frame, confidence=0.5)
    left_eye, right_eye = eyeDetector.extract(face)
    head_pose_angles = headPoseEstimator.estimate(face)
    gaze_vector = gazeEstimator.estimate(left_eye, right_eye, head_pose_angles)
    print(gaze_vector)
    face = draw_vector(gaze_vector, face)

    cv2.imshow('face', face)
    cv2.imshow('left eye', left_eye)
    cv2.imshow('right eye', right_eye)
    cv2.waitKey()

feed.close()
