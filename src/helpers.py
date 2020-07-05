import cv2
import math

def crop(image, box):
    """
    Extracts a part of the image specified by the bounding box.
    """
    if box and image is not None:
        xmin,ymin,xmax,ymax = box
        return image[ymin:ymax+1, xmin:xmax+1, :]
    else:
        return None

def fit(box, image):
    """
    Scales the bounding box to the image size.
    """
    if box is None or image is None:            
        return None
    else:
        h,w = image.shape[:-1]
        return tuple([int(coord*dim) for coord, dim in zip(box, [w, h, w, h])])

def normalize(v):
    """
    Normalizes a 3D vector.
    """
    x,y,z = v
    mag = math.sqrt(x*x + y*y + z*z) + 1e-4
    x /= mag
    y /= mag
    z /= mag
    return (x,y,z)

def clamp(value, range_min, range_max):
    """
    Clamps a value into a specified range.
    """
    return max(range_min, min(range_max, value))



def draw_gaze_vector(image, box, gx, gy):
    """
    Draws a gaze vector and a face bounding box.
    """

    if box is None:
        return image
    
    length = min(box[2]-box[0], box[3]-box[1])
    gx *= length
    gy *= length

    p1 = ((box[0]+box[2])//2, (box[1]+box[3])//2)
    p2 = (int(p1[0]+gx), int(p1[1]-gy))     # OpeCV's y axis is directed to down

    output_image = image.copy()
    cv2.arrowedLine(output_image, p1, p2, color=(0,255,0))
    cv2.rectangle(output_image, (box[0], box[1]), (box[2], box[3]), color=(0,0,255))
    return output_image


def draw_landmarks(face_image, landmarks):
    """
    An auxiliary function which draws facial landmarks on a face image (used for debugging).
    """
    h,w = face_image.shape[:-1]
    res_img = face_image.copy()  
    for i in range(0, landmarks.shape[1], 2):
        x = int(w*landmarks[0, i])
        y = int(h*landmarks[0, i+1])
        res_img = cv2.circle(res_img, (x,y), radius=2, color=(0,255,0))

    return res_img