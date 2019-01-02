#!/usr/bin/env python3
# Author: omgimanerd

import cv2
import matplotlib.pyplot as plt
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

def plot_fft(data):
    pass

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError('Unable to access webcam')
    face_detector = cv2.CascadeClassifier(FACE_CASCADE)
    calculator = PulseCalculator()

    t_start = time.time()
    delta = 0
    while True:
        valid, frame = cap.read()
        if not valid:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray)
        if len(faces) != 1:
            continue
        x, y, w, h = faces[0]
        xmin, xmax = x + w // 4, x + 3 * w // 4
        ymin, ymax = y, y + h // 3
        forehead = frame[ymin:ymax, xmin:xmax]
        intensity = get_intensity(forehead)
        calculator.add_observation(intensity, delta)
        delta = time.time() - t_start

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        pulse_text = '{:2.2f}'.format(calculator.get_pulse())
        cv2.putText(frame, pulse_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)
        cv2.imshow('Face', frame)
        if (cv2.waitKey(1) & 255) == ord('q'):
            break
    calculator.plot_observations()
    plt.show()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
