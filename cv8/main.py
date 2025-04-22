import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt

src_img = cv2.imread("hh.jpg")
dst_img = cv2.imread("ronaldo.jpg")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib_shape_predictor_68_face_landmarks.dat")

src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
src_faces = detector(src_gray)
src_landmarks = predictor(src_gray, src_faces[0])
src_points = np.array([[p.x, p.y] for p in src_landmarks.parts()])

src_img_landmarks = src_img.copy()
for (x, y) in src_points:
    cv2.circle(src_img_landmarks, (x, y), 2, (0, 255, 0), -1)
cv2.imshow("first", src_img_landmarks)
cv2.waitKey(0)

src_hull = cv2.convexHull(src_points)
src_mask = np.zeros_like(src_gray)
cv2.fillConvexPoly(src_mask, src_hull, 255)

src_rect = cv2.boundingRect(src_hull)
src_face_w, src_face_h = src_rect[2], src_rect[3]


dst_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)
dst_faces = detector(dst_gray)
dst_landmarks = predictor(dst_gray, dst_faces[0])
dst_points = np.array([[p.x, p.y] for p in dst_landmarks.parts()])

dst_img_landmarks = dst_img.copy()
for (x, y) in dst_points:
    cv2.circle(dst_img_landmarks, (x, y), 2, (0, 0, 255), -1)
cv2.imshow("second", dst_img_landmarks)
cv2.waitKey(0)

dst_hull = cv2.convexHull(dst_points)
dst_rect = cv2.boundingRect(dst_hull)
dst_face_w, dst_face_h = dst_rect[2], dst_rect[3]

src_face = src_img[src_rect[1]:src_rect[1]+src_face_h, src_rect[0]:src_rect[0]+src_face_w]
src_face_resized = cv2.resize(src_face, (dst_face_w, dst_face_h))
src_mask_resized = cv2.resize(src_mask[src_rect[1]:src_rect[1]+src_face_h, src_rect[0]:src_rect[0]+src_face_w], (dst_face_w, dst_face_h))

center_face = (dst_rect[0] + dst_face_w // 2, dst_rect[1] + dst_face_h // 2)

output = cv2.seamlessClone(src_face_resized, dst_img, src_mask_resized, center_face, cv2.NORMAL_CLONE)

cv2.imshow("Face Swap Result", output)
cv2.waitKey(0)
cv2.destroyAllWindows()