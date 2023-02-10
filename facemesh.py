import cv2
import mediapipe as mp
import numpy as np
from draw_pupil import draw_pupil


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

vid_cap = cv2.VideoCapture(0)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


def get_landmark(image, frame):
    height, width, _ = frame.shape
    landmarks = []
    results = face_mesh.process(image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = face_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                landmarks.append([x, y])
    return np.array(landmarks, np.int32)


def draw_face(frame, landmark):
    cv2.drawContours(frame, [landmark], -1, (0, 255, 0), 1)
    for p in landmark:
        frame = cv2.circle(frame, (p[0], p[1]), radius=3,
                           color=(0, 255, 0), thickness=-1)
    convexhull = cv2.convexHull(landmark)
    frame = cv2.polylines(
        frame, [convexhull], True, color=(0, 255, 0), thickness=1)
    return frame


while vid_cap.isOpened():
    ret, frame = vid_cap.read()
    if not ret:
        break
    f = frame.copy()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmark = get_landmark(image=image, frame=f)
    if len(landmark) != 0:
        """
        uncomment the following function to draw green dots on all 468 facial landmarks and connect them
        """
        # frame = draw_face(frame,landmark)

        """
        uncomment the following function to draw red dot on the pupil of the face detected
        """
        frame = draw_pupil(frame, landmark)

    cv2.imshow("", cv2.flip(frame, 1))
    key = cv2.waitKey(10)
    if key == 27:
        break

vid_cap.release()
