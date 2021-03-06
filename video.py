#!/usr/bin/env python3
# Author: Alvin Lin (omgimanerd)

import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import os
import time

from pulse_calculator import PulseCalculator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_CASCADE = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

def get_intensity(frame):
    b = np.mean(frame[:,:,0])
    g = np.mean(frame[:,:,1])
    r = np.mean(frame[:,:,2])
    return (b + g + r) / 3

def process_pulse(video, transform=None):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError('Unable to open specified video file')
    fps = cap.get(cv2.CAP_PROP_FPS)
    face_detector = cv2.CascadeClassifier(FACE_CASCADE)
    calculator = PulseCalculator()

    t = 0
    while cap.isOpened():
        valid, frame = cap.read()
        if not valid:
            break
        if transform:
            frame = transform(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray)
        if len(faces) != 1:
            continue
        x, y, w, h = faces[0]
        xmin, xmax = x + w // 4, x + 3 * w // 4
        ymin, ymax = y, y + h // 3
        forehead = frame[ymin:ymax, xmin:xmax]
        intensity = get_intensity(forehead)
        calculator.add_observation(intensity, t)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        pulse_text = '{:2.2f}'.format(calculator.get_pulse())
        cv2.putText(frame, pulse_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)
        cv2.imshow('Face', frame)
        t += 1 / fps
        if (cv2.waitKey(1) & 255) == ord('q'):
            break
    calculator.plot_observations()
    calculator.plot_pulse()
    plt.show()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    alvin = os.path.join(BASE_DIR, 'data/face2.mp4')
    process_pulse(alvin)
