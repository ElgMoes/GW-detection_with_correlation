"""
Program for generating the cumulative correlation distribution
==========================================================================================
By Arda Ölmez, Jelle Hoek and Laurens Zwart
"""

#Importing modules
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sys import path

#importing custom defenitions
import JALS_functions as f

#initiating variables
srate = 5000 #sampling rate in hertz
D = srate*2 #template length in seconds
T = srate*100 #data length in seconds
N = 50 #templates amount
C = np.linspace(0,1,10000)
t = np.linspace(0,T/srate,T)
normal_noise = np.random.normal(0, 1e-17, T)

#generating masses
minmass , maxmass = 50 , 150 #creating mass range
m1, m2 = f.gen_mass(minmass, maxmass, 'log', N)

#generating waveforms
hp, hc = f.gen_wave(m1, m2, N, srate, D) #template waveform

#correlation
correlation_vector = f.correlate_templates(N, T, D, hp, hc, normal_noise)

#calculating frequencies and probabilities
cum_correlation = np.zeros(C.size)
cum_chances = np.zeros(C.size)
for i in tqdm(range(0, C.size), desc = 'Doing frequency counting'):
    cum_correlation[i] = (C[i] <= correlation_vector[0,:]).sum()
cum_chances = 1-(cum_correlation/(T-D))

#save frequency and probability in files
np.save(path[0]+r"\\cum_correlation_distribution.npy", cum_correlation)
np.save(path[0]+r"\\chances_distribution.npy", cum_chances)

#plotting frequency and probability
plt.figure()
plt.plot(np.linspace(0, 1, np.load("normal_N_C.npy").size), np.load("normal_N_C.npy"), label="Frequency normal noise")
plt.legend()
plt.figure()
plt.plot(np.linspace(0, 1, np.load("chances_distribution.npy").size), np.load("chances_distribution.npy"), label="Probability normal noise")
plt.legend()
plt.show()