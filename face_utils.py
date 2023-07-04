import numpy as np
import cv2


LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
NOSE_INDICES = [30]
LEFT_MOUTH_INDICES = [48]
RIGHT_MOUTH_INDICES = [54]

def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom

def extract_point(shape, indices):
    points = map(lambda i: shape.part(i), indices)
    return list(points)[0]

def extract_eye(shape, eye_indices):
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)

def extract_eye_center(shape, eye_indices):
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6

def extract_left_eye_center(shape):
    return extract_eye_center(shape, LEFT_EYE_INDICES)

def extract_right_eye_center(shape):
    return extract_eye_center(shape, RIGHT_EYE_INDICES)

def extract_right_mouth(shape):
    return extract_point(shape, RIGHT_MOUTH_INDICES)

def extract_left_mouth(shape):
    return extract_point(shape, LEFT_MOUTH_INDICES)

def extract_nose(shape):
    return extract_point(shape, NOSE_INDICES)

def extract_lankmarks5(shape):
    left_eye = extract_left_eye_center(shape)
    right_eye = extract_right_eye_center(shape)
    right_mouth = extract_right_mouth(shape)
    left_mouth = extract_left_mouth(shape)
    nose = extract_nose(shape)
    return [(left_eye), (right_eye), (nose.x, nose.y), (left_mouth.x, left_mouth.y), (right_mouth.x, right_mouth.y)]

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M

def crop_image(image, det):
    left, top, right, bottom = rect_to_tuple(det)
    return image[top:bottom, left:right]
