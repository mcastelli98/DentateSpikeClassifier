# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 17:02:19 2023

@author: trin3692
"""

from peakdetect import detect_peaks
import numpy as np
import os as os
import time as time
import sys as sys
import pandas as pd
from importlib import reload as reload
import scipy.signal as sig
from scipy import io
import scipy.ndimage as ndimage

sr = 1250
DTT = 1000/sr # sampling time (in ms) of eeg signal
lfp2mV = .196/1000

def readfoldernm(folder):
    '''Gets the folder name and reports the basefile and recording day '''
    if folder[-1] == '/':
        folder = folder[:-1]
    aux = np.max([i for (i, char_) in enumerate(folder) if char_ == '/'])
    recday = folder[1 + aux:]
    b = folder + '/' + recday

    return b, recday

def describe_data(dataset):
    '''Describeds dataset by counting how many recording days and unique mice it includes'''
    mice_list = []
    n_days = len(dataset)

    for folder in dataset:
        b,recday = readfoldernm(folder)
        mID = b2mouse(b)

        if mID not in mice_list:
            mice_list.append(mID)

    print('########################')
    print('DATASET DESCRIPTION:')
    print('\t-n. recdays = {}'.format(n_days))
    print('\t-n. mice = {}'.format(len(mice_list)))

def getEventsInWin(events,win_l,sdur):
    '''Checks which events fall within the session duration (sdur) when adding a trigger window (win_l)'''
    return np.logical_and(((events-(win_l))>=0),((events+(win_l))<sdur))

def bandpass(signal, lowf, hif, sr=sr, FilterOrder=4, axis=-1):
    '''Bandpass Butterworth filter using cut-off frequency lowf and hif'''
    lowf_ = (2. / sr) * lowf  # Hz * (2/SamplingRate)
    hif_ = (2. / sr) * hif  # Hz * (2/SamplingRate)

    # filtering
    b, a = sig.butter(FilterOrder, [lowf_, hif_], 'band')  # filter design
    output = sig.filtfilt(b, a, signal, axis=axis)
    return output


def lowpass(signal, fcut, sr=1250., FilterOrder=4, axis=-1):
    '''Low Pass Butterworth filter using cut-off frequency fcut'''
    lof_ = (2. / sr) * fcut  # Hz * (2/SamplingRate)

    # filtering
    b, a = sig.butter(FilterOrder, lof_, 'low')  # filter design
    output = sig.filtfilt(b, a, signal, axis=axis)

    return output

def triggeredAverage(sig2trig, trigger, taLen=500, sr=1250., average=True):
    '''

    Trigger sig2trig with timestamps given by trigger
    trigger: is the timestamps (in samples) used to trigger the signal
    taLen: is the window size used to trigger (in samples)
    sr: is the sampling rate
    average: True if you want to average across all events
    '''
    taLen = int(taLen)
    prepost = np.arange(taLen, dtype=int) - int(taLen / 2)

    if len(np.shape(sig2trig)) == 1:
        sig2trig = sig2trig[None, :]

    mask_trig = trigger < (np.size(sig2trig, 1) + np.min(prepost))
    trigger = trigger[mask_trig]

    if average:
        ta = np.zeros((np.size(sig2trig, 0), taLen))
        for t in trigger:
            ta += sig2trig[:, t + prepost]
        ta /= len(trigger)
        ta = ta.squeeze()
    else:
        ta = np.zeros((len(mask_trig), np.size(sig2trig, 0), taLen)) + np.nan
        tis = np.where(mask_trig)[0]
        for (tii, t) in enumerate(trigger):
            ti = tis[tii]
            ta[ti] = sig2trig[:, t + prepost]

    taxis = 1000. * prepost / sr
    return ta, taxis


### PLOT FUNCTIONS
def probe_yticks(layer,ax=None):
    '''Writes the corresponding layer name for each lfp trace'''
    if ax is None:
        ax = plt.gca()
    ## get layers inf
    layer_trode = []
    layer_lbls = []
    for region in list(layer):
        if np.any(layer[region]):
            tr = layer[region]
            if isinstance(layer[region],(list,np.ndarray)):
                tr = layer[region][0]
            layer_trode.append(tr)
            layer_lbls.append(region)
    ## label ###
    ax.set_yticks(layer_trode)
    ax.set_yticklabels(layer_lbls)

def plt_Mean_ErrArea(x,mu,sem,color='k',ls='-',alpha=.5,label='',ax=None):
    '''Plots mean and sem with shaded area'''
    if ax is None:
        ax = plt.gca()
    ax.plot(x,mu,ls,color=color,lw=2,label=label)
    ax.fill_between(x,mu-sem,mu+sem,color=color,alpha=alpha)


def AdjustBoxPlot(ax=None,alpha=0.75,color=gray2):
    '''Despines axis and adds a grid'''
    if ax is None:
        ax = plt.gca()
    ax.grid(True,which='major',color=color,alpha=alpha,linestyle='--')
    sns.despine(ax=ax)