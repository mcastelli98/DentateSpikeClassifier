"""
Functions used to use the dentate spikes classification model.
Created on Fri Oct 20 2023

Author: Manfredi Castelli - manfredi.castelli@ndcn.ox.ac.uk
"""
# Libraries import

import numpy as np
import seaborn as sns

import utils
from scipy import stats

####### PARAMS ###############################
winL_ms = 100
talen = int(2 * (winL_ms / utils.DTT))
talenh = int(talen / 2)
taxis = utils.DTT * (np.arange(-talenh, talenh, 1))
f_low = 50  # Hz

nclass = 2
PURPLE = sns.hls_palette(10,l=.45,s=1)[8]
GREEN = sns.hls_palette(8,l=.4,s=.8)[3]
colors_DSs = [GREEN,PURPLE]

## Load model that can predict dentate spike type from the LFP of the dentate gyrus
filename = './Data/dSpikesClassifier_py10.model'

model_saved = np.load(filename, allow_pickle=True)
model = model_saved['model']
model_info = model_saved['info']
refLFP_gr = model_info['refLFP']

####### FUNCTIONS ############################
def model_report():
    print('#### REPORT ON TRAINED MODEL ####')
    utils.describe_data(model_info['dataset'])
    print()
    print('Dataset contained {} dentate spikes which were used to trained the model.\n\n\
    The model reached an accuracy of {:.2f}% (k={} cross validation, chance level: 50%).' \
          .format(model_info['n_spikes'], 100 * model_info['score'], model_info['k']))


def matchLength(x, y):
    np.random.seed()
    if len(y) > len(x):
        return x, np.random.choice(y, len(x))

    elif len(y) < len(x):
        return np.random.choice(x, len(y)), y
    else:
        return x, y


def norm_traces(lfp_trig):
    return np.array([stats.zscore(lfp) for lfp in lfp_trig])


def triggerLFP(lfps, dspikes):
    lfp_ds = utils.triggeredAverage(lfps, dspikes, taLen=talen, average=False)[0]
    return lfp_ds


def preProcessing(X):
    '''
    The preProcessing consists into 2 steps:
    1) Low pass lfp with 4th Order Butterworth filter with cut-off 50 Hz
    2) z-score each LFP trace
    '''
    X = utils.lowpass(X, f_low)  # low pass filter lfp
    return norm_traces(X)  # zscore lfp


def predictionReport(class_pred):
    print('############################\nClassification report:\nTotal # spikes:  ' + str(len(class_pred)))
    for classi in range(nclass):
        print(' {:.2f} % of  DS{}'.format(100 * (sum(class_pred == classi) / len(class_pred)), classi + 1))


def plotClassPred(lfp_ds_GR, class_pred, ax=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,3))
    if ax is None:
        ax = plt.gca()
    for classi in range(nclass):
        idx = class_pred == classi

        mu = lfp_ds_GR[idx].mean(0)
        sem = stats.sem(lfp_ds_GR[idx], axis=0)
        utils.plt_Mean_ErrArea(taxis, mu, sem, color=colors_DSs[classi], label='DS' + str(classi + 1), ax=ax)
    ax.set_xlabel('Time from peak (ms)', fontdict=utils.fontdict)
    ax.set_ylabel('LFP (mV)', fontdict=utils.fontdict)
    utils.AdjustBoxPlot(ax=ax)


def matchCh2ref(lfp_ds, trodes):
    '''
    Finds channel which best matches the template used to train the model;
    lfp_ds is the triggered avg lfp from dentate spikes times'''

    lfp_ds = norm_traces(lfp_ds)

    scores = np.zeros(len(lfp_ds))
    for chi, lfp in enumerate(lfp_ds):
        aux = trodes['desel'].iloc[chi]
        if (aux == 'dg') or ('dg' in aux):  # only loop for dg channels
            scores[chi] = stats.pearsonr(refLFP_gr, lfp)[0]  # compute correlation between

    return np.argmax(scores)


def predictDS_class(lfps, dspikes=None, GRchi=None, trodes=None, plot=False, report=False):
    '''
    Estimates DS class from model (accuracy 85% after k=20 cross validation)
    return DSclass_pred, lfp_ds_GR, dspikes

    INPUTS:
    lfps: lfp of session - it can be:
                            - (all channels, all session)
                            - triggered LFP (events,all channels,400 ms) (check)
    dspikes: timestamps of dspikes if None it will consider lfps already
             triggered by the DSpikes; else it will trigger lfps
    GRchi: channel of granule cell layer - if None it will get the channel
           within the dg which best corresponds to model template

    trodes : desel info - required if GRchi is 'None'
    plot: if True will plot the LFP of the two DSpikes estimation
    report: prints report on prediciton of ds1 and ds2

    OUTPUTS:
    DSclass_pred: DSspike class prediction  0s and 1s -> ds1,ds2
    lfp_ds_GR: is the lfp in the GRC triggered for all DSpikes
    dspikes: DSpikes timestamps corresponding to the class (in case triggering  discards some events)
    '''

    # Check if lfps are already triggered
    if dspikes is None:
        lfp_ds = lfps

    else:
        # Trigger LFPS with DSpikes
        dspikes = dspikes[utils.getEventsInWin(dspikes, talenh, len(lfps[0]))]
        lfp_ds = triggerLFP(lfps, dspikes)

    # Check if specific channel in DG is to be used
    if GRchi is None:
        GRchi = matchCh2ref(lfp_ds.mean(axis=0), trodes)

    # Get LFP from GR layer
    lfp_ds_GR = lfp_ds[:, GRchi]

    lfp_ds_GR_norm = preProcessing(lfp_ds_GR)
    DSclass_pred = model.predict(lfp_ds_GR_norm)

    # Plot estimates
    if plot:
        plotClassPred(lfp_ds_GR, DSclass_pred)
    if report:
        predictionReport(DSclass_pred)

    return DSclass_pred, lfp_ds_GR, dspikes

def detectDSpikes(lfps,ch,refch=0,mininumDelay = int(50*1.25)):
    '''
    Detect DSpikes and their profile
    Returns dSpikes, lfp_ds,profile_ds

    INPUTS:
    lfp: lfp signal (sampling rate = 1250 Hz)
    ch: channel idx for dspikes detection (DG)
    refch: channel idx. to be set as reference from ch for detection
    mininumDelay: minimum time lag between consecutive dspikes in samples

    OUTPUTS:
    DSpikes: times correponding to DSspikes occurence in samples
    '''
    lfp_c = np.array([])
    # Subtract Referenc LFP from DG Channel
    lfp0 = lfps[ch,:]-lfps[refch,:]
    # Low Pass LFP with cut-off 200 Hz
    lfp_ = utils.bandpass(lfp0,lowf=1,hif=200)
    lfp_c = np.concatenate((lfp_c,lfp_))
    # Set threshold for detection
    thrs = np.median(np.abs(lfp_c))*7

    dspikes = utils.detect_peaks(lfp_,mph=thrs,mpd=mininumDelay)
    return dspikes

def runDSpikes_detection(lfps,dg_chi=-1):
    '''
    Detect DSpikes and their profile
    Returns dSpikes, lfp_ds,profile_ds
    
    INPUTS:
    lfp: lfp signal from gr. cell layer in DG (sampling rate = 1250 Hz)
    dg_chi: channel in DG to be used to dentate spikes detection

    OUTPUTS:
    DSpikes: eeg times corresponding to DSspikes
    '''

    ## DETECT DSpikes
    return detectDSpikes(lfps,dg_chi,refch=0)