import datetime
import os

import numpy as np

import cv2

if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier(cv2.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Face", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        frame_save = np.copy(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        face = np.zeros([1, 1])
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = frame[y:y+h, x:x+w]

        cv2.imshow("Face", face)
        cv2.imshow("Frame", frame)
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break
        elif keypress == ord(" "):
            cv2.imwrite("face-camera/" + datetime.datetime.now().isoformat() + ".png", frame_save)
            print(os.getcwd() + " : " + "face-camera/pics/" + datetime.datetime.now().isoformat() + ".png")
    cap.release()
    cv2.destroyAllWindows()
