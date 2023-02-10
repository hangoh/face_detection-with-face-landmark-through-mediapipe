import cv2


def find_pupil(landmark):
    # for all the face point by mediapipe visit: https://raw.githubusercontent.com/google/mediapipe/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    FACE_point = [33, 133, 145, 159, 263, 362, 374, 386]
    landmark_coor = []
    for fp in FACE_point:
        p = [landmark[fp][0], landmark[fp][1]]
        landmark_coor.append(p)
    pupil = []
    i = 1
    while i <= 2:
        pupil.append(calculate_pupil_position(
            landmark_coor[(-1+i)*(i*i):4*(i*i)]))
        i = i+1
    return landmark_coor, pupil


def calculate_pupil_position(landmark_coor):
    # for the equation please visit: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    # the idea was to find the intersection between two line to get the location of the pupil
    x1, y1 = landmark_coor[0]
    x2, y2 = landmark_coor[1]
    x3, y3 = landmark_coor[2]
    x4, y4 = landmark_coor[3]

    p_xA = ((x1*y2-y1*x2)*(x3-x4))-((x1-x2)*(x3*y4-y3*x4))
    p_xB = ((x1-x2)*(y3-y4))-((y1-y2)*(x3-x4))
    p_x = p_xA/p_xB

    p_yA = ((x1*y2-y1*x2)*(y3-y4))-((y1-y2)*(x3*y4-y3*x4))
    p_yB = ((x1-x2)*(y3-y4))-((y1-y2)*(x3-x4))
    p_y = p_yA/p_yB

    return [p_x, p_y]


def draw_pupil(frame, landmark):
    lm_c, pupil = find_pupil(landmark)
    line = []
    for c in lm_c:
        line.append(c)
        frame = cv2.circle(frame, (c[0], c[1]), radius=3,
                           color=(0, 255, 0), thickness=-1)
        if len(line) == 2:
            frame = cv2.line(
                frame, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (0, 255, 0), 1)
            line.clear()
    for p in pupil:
        frame = cv2.circle(frame, (round(p[0]), round(p[1])), radius=7,
                           color=(0, 0, 255), thickness=-1)
    return frame
