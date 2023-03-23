import numpy as np
from scipy import interpolate, signal


def smooth_filter(data,kernel=100):
    '''
    Used to filter the input data with Moving Average filter
    :param data: data need to be filtered
    :param kernel: the window size of the Moving Average filter
    :return: filtered data
    '''
    print("##################### start smoothing #####################")
    shape = data.shape

    print('data shape:', shape)
    if len(shape) == 1:
        data = np.append(data[:kernel // 2], data)
        data = np.append(data, data[-kernel // 2:])
        data_new = np.empty((shape[0],))
        print(data_new.shape)
        for i in range(shape[0]):
            data_new[i:i + kernel] = np.average(data[i:i + kernel])
    elif len(shape) == 2:
        data = np.append(data[:, :kernel // 2], data, axis=1)
        data = np.append(data, data[:, -kernel // 2:], axis=1)
        data_new = np.empty((shape[0], shape[1]))
        print(data_new.shape)
        for i in range(shape[1]):
            data_new[:, i] = np.average(data[:, i:i + kernel], axis=1)
    else:
        raise UserWarning('Please ensure the input data has shape dim <= 2')
    return data_new

def HighPass_filter(data,fs=1000,fc=2):
    '''
    High pass filter for data processing
    :param data: data to be processed
    :param fs: sampling frequency
    :param fc: cut frequency
    :return: data filtered
    '''
    b, a = signal.butter(4, 2.0*fc/fs, 'highpass')
    if len(data.shape) == 1:
        filtered = signal.filtfilt(b, a, data)
    elif len(data.shape) == 2:
        data_list = []
        for item in data:
            data_list.append(signal.filtfilt(b, a, item))
        filtered = np.asarray(data_list)
    else:
        raise UserWarning('Please ensure the input data has shape dim <= 2')
    return filtered

def LowPass_filter(data,fs=1000,fc=100):
    '''
    Low pass filter for data processing
    :param data: data to be processed
    :param fs: sampling frequency
    :param fc: cut frequency
    :return: data filtered
    '''
    b, a = signal.butter(4, 2.0*fc/fs, 'lowpass')
    if len(data.shape) == 1:
        filtered = signal.filtfilt(b,a,data)
    elif len(data.shape) == 2:
        data_list = []
        for item in data:
            data_list.append(signal.filtfilt(b, a, item))
        filtered = np.asarray(data_list)
    else:
        raise UserWarning('Please ensure the input data has shape dim <= 2')
    return filtered

def Envelope(index=None, data=None, kernel=5, up_thresh=None, low_thresh=None):
    '''
    Find the biggest and smallest points in the original data
    :param data:         data to be processed
    :param kernel:       window size when find the extreme values
    :param up_thresh:    threshold of determining if it's a maximum
    :param down_thresh:  threshold of determining if it's a minimum
    :return:             the extreme values and their corresponding indexes
    '''
    print("##################### searching extreme values #####################")
    if len(data.shape) == 1:
        x_up = [0]
        data_up = [data[0]]
        x_low = [0]
        data_low = [data[0]]
        for i in range(kernel, data.shape[0] - kernel):
            if np.sign(data[i] - data[i-kernel]) == 1 and np.sign(data[i] - data[i+kernel]) == 1 and data[i] >= up_thresh:
                x_up.append(i)
                data_up.append(data[i])
            if np.sign(data[i] - data[i-kernel]) == -1 and np.sign(data[i] - data[i+kernel]) == -1 and data[i] <= low_thresh:
                x_low.append(i)
                data_low.append(data[i])
        x_up.append(data.shape[0]-1)
        data_up.append(data[-1])
        x_low.append(data.shape[0]-1)
        data_low.append(data[-1])
    else:
        raise UserWarning('Please ensure the input data has shape dim == 1')
    if index is None:
        return (x_up, data_up), (x_low, data_low)
    else:
        return (index[x_up], data_up), (index[x_low], data_low)