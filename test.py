#!/usr/bin/env python3
# Author: axl1439

import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import os

import time

from pulse_calculator import PulseCalculator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_CASCADE = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml'))

def process_pulse(video):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    t_interval = 1000 / fps
    if not cap.isOpened():
        raise ValueError('Unable to opened specified video file')

    calculator = PulseCalculator()
    t = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray)
        if len(faces) != 1:
            continue
        x, y, w, h = faces[0]
        xmin, xmax = x + w // 4, x + 3 * w // 4
        ymin, ymax = y, y + h // 3
        roi_green = frame[ymin:ymax, xmin:xmax,1]
        calculator.add_observation(np.mean(roi_green), t)
        pulse_text = '{:2.2f}'.format(calculator.get_pulse())
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(frame, pulse_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)
        cv2.imshow('Face', frame)
        cv2.imshow('Green', roi_green)
        t += t_interval
        if (cv2.waitKey(1) & 255) == ord('q'):
            break
    calculator.plot_pulse()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    face2 = os.path.join(BASE_DIR, 'data/face2.mp4')
    process_pulse(face2)
