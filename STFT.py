import numpy as np
from scipy import signal

def stft(signal, fs, window='hann', nperseg=256, noverlap=None, nfft=None):
    """
    进行STFT变换
    :param signal: 输入信号
    :param fs: 采样率
    :param window: 窗函数类型
    :param nperseg: 每个窗口的长度
    :param noverlap: 相邻两个窗口之间的重叠长度，默认为None表示无重叠
    :param nfft: FFT长度，默认值为None，即nfft=nperseg
    :return: STFT结果的频域矩阵和频率向量
    """
    if nfft is None:
        nfft = nperseg

    f, t, stft_result = signal.stft(signal, fs, window=window, nperseg=nperseg,
                                    noverlap=noverlap, nfft=nfft)
    return f, t, stft_result

import matplotlib.pyplot as plt
from scipy.io import wavfile

# 读取音频文件
fs, signal = wavfile.read('audio.wav')

# 计算STFT
f, t, stft_result = stft(signal, fs)

# 绘制STFT的时频图
plt.pcolormesh(t, f, np.abs(stft_result))
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()