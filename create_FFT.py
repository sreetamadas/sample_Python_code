# https://www.researchgate.net/post/Can_someone_provide_me_the_Python_script_to_plot_FFT

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def plotSpectrum(y,Fs):
    n = int(len(y)) # length of the signal
    k = np.arange(n)
    T = n/Fs  # Fs is sampling freq. , T will be time duration of signal
    frq = k/T # two sides frequency range
    frq = frq[range(int(len(frq)/2))] # one side frequency range
    Y = np.fft.fftn(y)/n # fft computing and normalization
    Y = Y[range(int(len(Y)/2))]
    
    fig, ax = plt.subplots(figsize=(15,3))
    ax.scatter(frq,abs(Y),marker = 'o', s=10, alpha=0.4) #, cmap = cm.jet ; marker = 'o'\n",
    
    return frq[np.argmax(abs(Y))]

 
# np.argmax Returns the index(ices) 'I' of the maximum values of 'abs(Y)'
# frq[..] returns the value at index I in array frq - this corresponds to the no. of cycles in the signal y

# The FFT function will return a complex array : Amplitudes (Y-axis) & corresponding frequencies (X-axis). 
# This is 2-sided (w & -w for the same frequency), so take 1-sided frequency.
# Take absolute value to find Amplitude, & the index of the max amplitude in the frq array corresponds to the dominant frequency.

