import scipy.signal as signal
import pandas as pd
import numpy as np


# B, A : coefficients of the filter
# N: order of the filter; higher the order, more the number of coefficients & hence more computational complexity/ processing.
# Wn = (Filtering frequency threshold/ sampling frequency) * 2
# The 2 comes from considerations of Nyquist frequency.



# convert signal data from dataframe colunm to an array
signal_data_array = pd.np.array(df.iloc[:,[colnum]]).reshape(len(df))


## low pass filter - removes high freq noise
N  = 2                 # Filter order
Wn = (10/100)*2       # Cutoff frequency 0.1   ; (Filtering frequency threshold/ sampling frequency) * 2
B, A = signal.butter(N,Wn, output='ba',btype='lowpass')    # save Wn separately & put here, don’t put numeric value directly
filtered_signal=signal.filtfilt(B,A,signal_data_array)   



## high pass filter - removes low freq noise
N1 = 2
Wn1= 0.25 * 2 /100
B1,A1 = signal.butter(N1,Wn1,output='ba',btype='highpass')  # save Wn1 separately & put here, don’t put numeric value directly
filtered2 = signal.filtfilt(B1,A1,filtered_signal)



#################################################################
## how to find which filter order to use ?
import matplotlib.pyplot as plt
from scipy.signal import freqz

# Sample rate and desired cutoff frequencies (in Hz).
fs = 100.0
lowcut = 10.0
highcut = 10.0

# Plot the frequency response for a few different orders.
plt.figure(1)
plt.clf()
for order in [2, 6, 9]:
   # N=2
    Wn=(10/100)*2
    b, a = butter(order,Wn, output='ba',btype='highpass')
    w, h = freqz(b, a, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)


plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
         '--', label='sqrt(0.5)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.legend(loc='best')



