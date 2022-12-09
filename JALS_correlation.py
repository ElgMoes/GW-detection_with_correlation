"""
Document for scanning data for GWs
==========================================================================================
By Arda Ölmez, Jelle Hoek and Laurens Zwart
"""


#importing modules
import numpy as np
import matplotlib.pyplot as plt

#importing custom defenitions
import JALS_functions as f

#initializing variables
srate = 5000 #Hz
D = srate*3 #template length
T = srate*10 #data length
N = 10 #templates amount
signal_amount = 1 #amount of signal plugged into noise
t = np.linspace(0,T/srate,T)
noise = np.random.normal(0,1e-17,T) #generating noise

#generating masses
minmass , maxmass = 50 , 150 #creating mass range
m1_hidden, m2_hidden = f.gen_mass(minmass, maxmass, 'uniform', 1)
m1, m2 = f.gen_mass(minmass, maxmass, 'linear', N)

#generating waveforms
hp_hidden, hc_hidden = f.gen_wave(m1_hidden, m2_hidden, 1, srate, D) #hidden waveform
hp, hc = f.gen_wave(m1, m2, N, srate, D) #template waveform

#setting signal in noise
signal = hp_hidden[0,:] #first hidden waveform is signal
random_id = np.random.randint(T//4, 3*T//4) 	
noise[random_id:random_id+D] += signal_amount*signal #signal is added to the noise

#correlation
correlation_vector = f.correlate_templates(N, T, D, hp, hc, noise)

#calculating starting point of wave
output_high_corr = []
output_high_corr_index = []
for i in range(0,N):
        output_high_corr_index.append(np.argmax(correlation_vector[i,:]))
        output_high_corr.append(max(correlation_vector[i,:]))

#printing starting point of wave
print("signal started at: ",random_id/srate) #known starting point
print("SNR of signal is:", f.SNR(signal, noise))
max_output_high_corr_index = np.argmax(output_high_corr)
found_signal_start = output_high_corr_index[max_output_high_corr_index]
print("signal found at: ",(found_signal_start)/srate) #calculated starting point

#calculating chance of being wave
f.calc_wave_chance_normal(output_high_corr)

#fetching signal with highest correlation
signal_found = hp[max_output_high_corr_index,:]

#plotting
plt.figure()
plt.plot(t, noise, label = 'signal+noise')
plt.plot(t[random_id:random_id+D], signal, label = 'signal')
plt.plot(t[found_signal_start:found_signal_start+D], signal_found, label = 'found signal')
plt.figure()
plt.title("Correlation")
for i in range(0,N):
    plt.plot(t[:T-D], correlation_vector[i,:], label = str(i))
plt.show()