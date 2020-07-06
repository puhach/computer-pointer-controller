from input_feeder import InputFeeder
from face_detection import FaceDetector
from facial_landmarks_detection import EyeDetector
from head_pose_estimation import HeadPoseEstimator
from gaze_estimation import GazeEstimator
from mouse_controller import MouseController
from collections import deque
import helpers
import cv2
import argparse
import time

print('Initialization...')

parser = argparse.ArgumentParser(description="Computer Pointer Controller")
parser.add_argument('--input', type=str, required=True,
                    help="An input file name or 'cam' to capture input from a webcam.")
parser.add_argument('--device', type=str, default='cpu', 
                    help="Device name to perform inference on. Defaults to CPU.")
parser.add_argument('--ext', type=str, default=None,
                    help="Specifies the extension to use with the device.")
parser.add_argument('--precision', type=str, default='FP32',
                    help="Specifies the model precision to use: FP32, FP16, or FP32-INT8. Default is FP32.")
parser.add_argument('--concurrency', type=int, default=1, 
                    help="Defines the number of concurrent requests each model can execute. "
                        "Pass zero for synchronous inference. Default is 1.")
parser.add_argument('--confidence', type=float, default=0.5,
                    help="Specifies face detection probability threshold. Must be in range from 0 to 1. "
                        "Default is 0.5.")
parser.add_argument('--failsafe', action='store_true', default=False, 
                    help="Enables the fail-safe feature of PyAutoGUI. By default, it's disabled.")
parser.add_argument('--clean', action='store_true', default=False,
                    help="Enables visualization of intermediate model outputs. Active by default.")
parser.add_argument('--stats', action='store_true', default=False,
                    help="Prints per-layer performance statistics. Disabled by default.")
parser.add_argument('--silent', action='store_true', default=False, 
                    help="Enables the silent mode when video output and the mouse control feature "
                        "are disabled. Useful for performance measurement. Disabled by default.")
# TODO: add more arguments (mouse controller speed and precision)

args = parser.parse_args()

feed = InputFeeder(args.input)

t = -time.time()
faceDetector = FaceDetector(precision=args.precision, concurrency=args.concurrency, device=args.device, extensions=args.ext)
eyeDetector = EyeDetector(precision=args.precision, concurrency=args.concurrency, device=args.device, extensions=args.ext)
headPoseEstimator = HeadPoseEstimator(precision=args.precision, concurrency=args.concurrency, device=args.device, extensions=args.ext)
gazeEstimator = GazeEstimator(precision=args.precision, concurrency=args.concurrency, device=args.device, extensions=args.ext)
mouseController = MouseController(precision='medium', speed='medium', failsafe=args.failsafe)
t += time.time()

print(f'Model Loading Time: {t:.2} s')

print('Running...')

q = deque()
faces_produced = 0
head_poses_produced = 0
eyes_produced = 0
hpae_consumed = 0
wait_needed = False
done = False
t = -time.time()

while not done:

    gaze_vector_consumed, gaze_vector = gazeEstimator.consume_output(wait_needed)
    if gaze_vector_consumed:
        frame, face_box, _, _ = q.popleft()
        faces_produced -= 1
        eyes_produced -= 1
        head_poses_produced -= 1
        hpae_consumed -= 1

        if not args.silent:
            if gaze_vector:
                gx, gy, _ = gaze_vector            
                mouseController.move(gx, gy)

                if args.clean:
                    output_frame = frame
                else:
                    output_frame = helpers.draw_gaze_vector(frame, face_box, gx, gy)

            else:   # gaze vector is None
                output_frame = frame    # show the original frame

            cv2.imshow(args.input, output_frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    if hpae_consumed < min(head_poses_produced, eyes_produced):
        left_eye, right_eye = q[hpae_consumed][2]
        head_pose_angles = q[hpae_consumed][3]
        hpae_consumed += 1
        gazeEstimator.feed_input(left_eye, right_eye, head_pose_angles)

    head_pose_consumed, head_pose_angles = headPoseEstimator.consume_output(wait_needed)
    if head_pose_consumed:
        q[head_poses_produced][3] = head_pose_angles
        head_poses_produced += 1

    eyes_consumed, eye_boxes = eyeDetector.consume_output(wait_needed)
    if eyes_consumed:
        frame, face_box, _, _ = q[eyes_produced]
        eyes = eyeDetector.preprocess_output(eye_boxes, face_image=helpers.crop(frame, face_box))
        q[eyes_produced][2] = eyes
        eyes_produced += 1

    face_consumed, face_box = faceDetector.consume_output(confidence=args.confidence, wait=wait_needed)
    if face_consumed:        
        face_img, face_box = faceDetector.preprocess_output(face_box, frame=q[faces_produced][0])
        q[faces_produced][1] = face_box
        faces_produced += 1
        eyeDetector.feed_input(face_img)
        headPoseEstimator.feed_input(face_img)

    frame = feed.read_next()
    if frame is not None:
        # [original frame, face box, eyes, head pose]
        q.append([frame, None, None, None])
        faceDetector.feed_input(frame)
    else:
        # When we reached the end of the input stream we have to wait for all frames to finish processing.
        # To avoid idle running the loop, blocking wait is used when no output is available from any model.
        wait_needed = not gaze_vector_consumed and not head_pose_consumed and not eyes_consumed and not face_consumed
        done = len(q)<1     
    
t += time.time()
feed.close()

print(f'Done\n\nTotal Processing Time: {t:.4} s')

if args.stats:
    print('\nLayer-wise Execution Time')
    faceDetector.print_stats(title="\nFace Detector")
    eyeDetector.print_stats(title="\nEye Detector")
    headPoseEstimator.print_stats(title="\nHead Pose Estimator")
    gazeEstimator.print_stats(title="\nGaze Direction Estimator")
