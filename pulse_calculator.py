#!/usr/bin/env python3
# Author: Alvin Lin

import math
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

THRESHOLD = 10

class PulseCalculator:
    def __init__(self, t_start=0):
        self.t_start = t_start
        self.observations = []
        self.pulse = []

    def add_observation(self, value, time):
        self.observations.append([value, time])
        pulse = self.calculate_pulse()
        self.pulse.append([pulse, time])

    def get_observations(self, window):
        if len(self.observations) < 20:
            return None
        observations = np.array(self.observations)
        last_observation_t = observations[-1,1]
        selected = observations[:,1] > (last_observation_t - window)
        return observations[selected]

    def calculate_pulse(self, window=6000):
        if len(self.observations) < 20:
            return 0
        observations = np.array(self.observations)
        start_t = observations[-1,1] - window
        observations = observations[observations[:,1] > start_t]
        values, times = observations[:,0], observations[:,1]
        n = len(observations)
        duration = (times[-1] - times[0]) / 1000
        fps = float(n) / duration

        spaced_time = np.linspace(times[0], times[-1], n)
        interpolated = np.interp(spaced_time, times, values) * np.hamming(n)
        fft = np.abs(np.fft.rfft(interpolated - np.mean(interpolated)))
        freqs = float(fps) / n * np.arange(n / 2 + 1) * 60.0
        selected_index = np.where((freqs > 50) & (freqs < 180))
        pruned = fft[selected_index]
        if len(pruned) == 0:
            return 0
        return freqs[selected_index][np.argmax(pruned)]

    def get_pulse(self):
        return self.pulse[-1][0]

    def plot_observations(self):
        observations = np.array(self.observations)
        values, times = observations[:,0], observations[:,1]
        plt.figure(1)
        plt.xlabel('Time')
        plt.ylabel('Optical Intensity')
        plt.plot(times, values)

    def plot_pulse(self):
        pulse = np.array(self.pulse)
        values, times = pulse[:,0], pulse[:,1]
        plt.figure(2)
        plt.xlabel('Time')
        plt.ylabel('Calculated BPM')
        plt.plot(times, values)

    def serialize(self, name):
        np.savetxt(name, self.observations, delimiter=',')

if __name__ == '__main__':
    calculator = PulseCalculator()
    observations = np.loadtxt('pulse.txt', delimiter=',')
    calculator.observations = observations[:100]
    print(calculator.get_pulse())
