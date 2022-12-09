"""
Document for finding the sensitivity of our program
==========================================================================================
By Arda Ölmez, Jelle Hoek and Laurens Zwart
"""


#importing modules
import numpy as np
import matplotlib.pyplot as plt
import JALS_functions as f

#initializing variables
srate = 5000 #Hz
D = srate*2 #template length
N = 50 #number of templates
tests = 2000 #number of tests
amplitude = np.logspace(-5,2,tests) #amplitude scaled logarithmic between 10^ -4 and 2

#generating masses
minmass , maxmass = 50 , 150 #creating mass range
m1_hidden, m2_hidden = f.gen_mass(minmass, maxmass, 'uniform', 1) #one random mass (dummy signal)
m1, m2 = f.gen_mass(minmass, maxmass, 'log', N) #logarithmic spaced masses for templates

#generating waveforms
hp_hidden, hc_hidden = f.gen_wave(m1_hidden, m2_hidden, 1, srate, D) #hidden waveform
hp, hc = f.gen_wave(m1, m2, N, srate, D) #template waveforms

#the tests
snr_array = np.ndarray(tests)
correlation_vector = np.ndarray((tests,N))
chances = np.ndarray(tests)
for i in range(0,tests):
    noise = np.random.normal(0,1e-17,D) #generating noise
    data = noise + amplitude[i]*hp_hidden[0,:] #signal is added to the noise
    correlation_vector[i,:] = np.sqrt(f.scorrelation(hp, data) + f.scorrelation(hc, data))
    snr_array[i] = 10*np.log10(f.SNR(amplitude[i]*hp_hidden[0,:], noise))
    chances[i] = f.calc_wave_chance_normal(correlation_vector[i,:])

#plotting
plt.figure()
plt.title("Correlation")
plt.xlabel("amplitude")
plt.ylabel("correlation")
plt.xscale("log")
plt.plot(amplitude, correlation_vector)
plt.figure()
plt.title("SNR")
plt.xlabel("amplitude")
plt.ylabel("SNR (dB)")
plt.xscale("log")
plt.plot(amplitude,snr_array)
plt.figure()
plt.title("chance of being detected")
plt.ylabel("chance")
plt.xlabel("SNR (dB)")
plt.plot(snr_array,chances)
plt.show()