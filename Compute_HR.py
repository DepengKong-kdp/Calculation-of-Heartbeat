
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import smooth_filter, HighPass_filter, LowPass_filter, Envelope


class Compute_HR:
    def __init__(self, path, fs=125, fc_low=45, fc_high=10, up_thresh=0.00015, low_thresh=0):
        '''
        Convert the voltage measurement to heart beat
        :param path: path of the input data
        :param fs: sampling rate of input data
        :param fc_low: cut-off frequency of the Low-pass filter
        :param fc_high: cut-off frequency of the High-pass filter
        :param up_thresh:    threshold of determining if it's a maximum
        :param down_thresh:  threshold of determining if it's a minimum
        '''
        self.data = pd.read_excel(path, header=None)
        self.t = np.asarray(self.data[0][1:]).astype(float)
        self.v = np.asarray(self.data[1][1:]).astype(float)
        self.fs = fs
        self.fc_low = fc_low
        self.fc_high = fc_high
        self.up_thresh = up_thresh
        self.low_thresh = low_thresh

        self.v_ = self.preproc()
        self.hr = []
        self.hr_t = []
        self.hr_avg = None

    def preproc(self):
        # filter
        v_ = HighPass_filter(self.v, fs=self.fs, fc=self.fc_high)
        v_ = LowPass_filter(v_, fs=self.fs, fc=self.fc_low)
        return v_

    def compute_HR(self):
        # find peaks (local maximum)
        (x_up, data_up), (_, _) = Envelope(index=self.t, data=self.v_, kernel=1, up_thresh=self.up_thresh, low_thresh=self.low_thresh)
        # compute the HR
        for i in range(1, len(x_up)-2):
            inter_points = x_up[i+1] - x_up[i]      # interval between heartbeats
            hr_          = 60 / inter_points        # heartbeats (bpm)
            self.hr.append(hr_)
            self.hr_t.append(x_up[i])
        # smooth the calculated heartbeat curve
        self.hr_avg = smooth_filter(np.asarray(self.hr), kernel=50).reshape(-1)

    def visulization(self):
        plt.subplot(3, 1, 1)
        plt.plot(self.t, self.v)
        plt.ylabel('Raw signal (V)')
        plt.subplot(3, 1, 2)
        plt.plot(self.t, self.v_)
        plt.ylabel('Filtered signal (V)')
        plt.subplot(3, 1, 3)
        plt.plot(self.hr_t, self.hr)
        plt.plot(self.hr_t, self.hr_avg, 'r--', linewidth=1)
        plt.ylabel('HR (BPM)')
        plt.xlabel('Time (s)')
        plt.show()

    def save_results(self, name):
        data_frame1 = pd.DataFrame({
            'Time (s)'           : self.t,
            'Raw Signal (V)'     : self.v,
            'Filtered Signal (V)': self.v_,
        })
        data_frame2 = pd.DataFrame({
            'Time for HR (s)'    : self.hr_t,
            'HR (BPM)'           : self.hr,
            'Avg HR (BPM)'       : self.hr_avg
        })

        data_frame1.to_csv(name)
        data_frame2.to_csv(name)

if __name__ == '__main__':
    HR = Compute_HR(path='./data/ECG_example_data.xlsx')
    HR.compute_HR()
    HR.visulization()
    HR.save_results('./result/computed_HR.csv')