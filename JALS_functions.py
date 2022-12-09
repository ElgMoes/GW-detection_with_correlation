"""
Document with all functions
=============================================================================================
By Arda Ölmez, Jelle Hoek and Laurens Zwart              
"""

#importing modules
import mlgw.GW_generator as generator
import numpy as np
from tqdm import tqdm
from sys import path

#defining scalar product function
def scalar_product(signal_A, signal_B):
    result = signal_A * signal_B
    result = np.sum(result, axis =-1)
    return result

#defining squared correlation function
def scorrelation(signal_A, signal_B): 
    result = scalar_product(signal_A, signal_B)**2 
    result /= (scalar_product(signal_A, signal_A) * scalar_product(signal_B, signal_B))
    return result

#defining generating waveforms
def gen_wave(m1,m2,zero_amount, srate, D):
    gen = generator.GW_generator() #instance of the wavefirm generator
    zeros = np.zeros(zero_amount)
    theta = np.stack((m1,m2,zeros,zeros),axis=1)
    times = np.linspace(-D/srate,0.02, D)
    hp, hc = gen.get_WF(theta, times, modes=(2,2))
    return hp, hc #return amplitude and frequency

#defining signal-to-noise ratio
def SNR(signal, noise):
    snr = np.mean(signal**2)/np.mean(noise**2)
    return snr

#defining generating masses
def gen_mass(minmass, maxmass, type, amount):
    if type == 'uniform':
        m2 = np.random.uniform(minmass, maxmass, amount)
        m1 = np.random.uniform(0, maxmass, amount) + m2
    elif type == 'log':
        m2 = m1 = np.logspace(np.log10(minmass),np.log10(maxmass), amount)
    else:
        print("Choose between: 'uniform' or 'log'")
    return m1, m2

#defining corralating templates
def correlate_templates(N, T, D, hp, hc, signal):
    correlation_vector = np.zeros((N,T-D)) #empty matrix
    for i in tqdm(range(0, T-D, 1), desc = 'Doing Matched Filtering'):
        corr = scorrelation(hp, signal[i:i+D]) + scorrelation(hc, signal[i:i+D]) #correlation
        correlation_vector[:,i] = np.sqrt(corr) #correlation added to matrix
    return correlation_vector

#defining calculating wave chance with normal distribution noise
def calc_wave_chance_normal(input):
    normal_P_C = np.load(path[0]+r"\\chances_distribution.npy") #must include path to the program
    chance = 100*normal_P_C[int(max(input)*len(normal_P_C))]
    return chance