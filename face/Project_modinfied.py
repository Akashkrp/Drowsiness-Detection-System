import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame

cap = cv2.VideoCapture(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

EAR_THRESH_SLEEP = 0.21
EAR_THRESH_DROWSY = 0.25

sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

TEAM= "TEAM: F.R.I.D.A.Y"

def eye_aspect_ratio(eye):
    # Euclidean distances between the two vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])

    # Eye Aspect Ratio
    ear = (A + B) / (2.0 * C)
    return ear

face_frame = None

pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        EAR_Value = (left_ear + right_ear) / 2
        values = "E.A.R =" + str(EAR_Value)

        if EAR_Value < EAR_THRESH_SLEEP:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                alarm_sound.play(-1)
        elif EAR_Value < EAR_THRESH_DROWSY:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy :("
                color = (0, 0, 255)
                alarm_sound.stop()
        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)
                alarm_sound.stop()

        cv2.putText(frame, status, (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, values, (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 3)

        for (x, y) in np.concatenate((left_eye, right_eye), axis=0):
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    if face_frame is not None:
        cv2.putText(frame, TEAM, (20, 450), cv2.FONT_ITALIC, 1, (0, 165, 255), 3)
        cv2.imshow("Result of detector", face_frame)
        

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
