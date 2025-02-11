"""
Utils Functions used for dentate spikes classification model.
Created on Fri Oct 20 2023

Author: Manfredi Castelli - manfredi.castelli@ndcn.ox.ac.uk
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt


sr = 1250 # sampling rate
DTT = 1000/sr # sampling time (in ms) of eeg signal

# Plot style fonts
fontdict = {
        'style':'normal', #‘normal’, ‘italic’, ‘oblique’
        'color':  'k',
        'weight': 'normal',
        'size': 18,
        }


def readfoldernm(folder):
    '''Gets the folder name and reports the basefile and recording day '''
    if folder[-1] == '/':
        folder = folder[:-1]
    aux = np.max([i for (i, char_) in enumerate(folder) if char_ == '/'])
    recday = folder[1 + aux:]
    b = folder + '/' + recday

    return b, recday

def b2mouse(b):
    '''Extracts mouse id from basefile of session'''
    recday = b.split('/')[-1]
    return recday.split('-')[0]

def describe_data(dataset):
    '''Describes dataset by counting how many recording days and unique mice it includes'''
    mice_list = []
    n_days = len(dataset)

    for folder in dataset:
        b,recday = readfoldernm(folder)
        mID = b2mouse(b)

        if mID not in mice_list:
            mice_list.append(mID)

    print('#################################')
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

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False):
    __author__ = "Marcos Duarte, https://github.com/demotu/BMC"
    __version__ = "1.0.4"
    __license__ = "MIT"

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])
    return ind

def MatrixGaussianSmooth(Matrix,GaussianStd,GaussianNPoints=0,NormOperator=np.sum):

	# Matrix: matrix to smooth (rows will be smoothed)
	# GaussiaStd: standard deviation of Gaussian kernell (unit has to be number of samples)
	# GaussianNPoints: number of points of kernell
	# NormOperator: # defines how to normalise kernell

	if GaussianNPoints<GaussianStd:
		GaussianNPoints = int(4*GaussianStd)

	GaussianKernel = sig.get_window(('gaussian',GaussianStd),GaussianNPoints)
	GaussianKernel = GaussianKernel/NormOperator(GaussianKernel)

	if len(np.shape(Matrix))<2:
		SmoothedMatrix = np.convolve(Matrix,GaussianKernel,'same')
	else:
	    	SmoothedMatrix = np.ones(np.shape(Matrix))*np.nan
	    	for row_i in range(len(Matrix)):
	    		SmoothedMatrix[row_i,:] = \
		    		np.convolve(Matrix[row_i,:],GaussianKernel,'same')

	return SmoothedMatrix,GaussianKernel


def runCSD_(LFPs):
        nChs = np.size(LFPs,0)
        nSamples = np.size(LFPs,1)

        CSD = np.zeros((nChs,nSamples))
        for chi in range(1,nChs-1):
                CSD[chi,:] = -(LFPs[chi-1,:]-2*LFPs[chi,:]+LFPs[chi+1,:])
                
        return CSD

    
def groupConsec(data,minsize=1):
    '''Group consecutive values'''
    from itertools import groupby
    from operator import itemgetter

    groups =[]

    for k,g in groupby(enumerate(data),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))

        if np.size(group)>=minsize:
                groups.append(np.array(group))

    return groups 

def runCSD(LFPs,smooth=True,spacing=20,sm=50):
    '''
    Returns CSD for signals in LFPs.

    CSD = runCSD(LFPs)

    input: LFPs - array (channels, time samples)
                IMPORTANT!: remember to order channels in LFPs according to their spatial locations.
                Typically, LFPs are actually event-triggered averages, but you can also input raw traces.
                Event-triggers might be troughs/peaks of an oscillation.
                If you are interested in the CSD of a particular oscillation you might input filtered traces.

    output: CSD - array (channels, time samples)
    '''
    validchs = np.where(~np.isnan(np.mean(LFPs,axis=1)))[0]
    chgroups = groupConsec(validchs,minsize=3) # to make sure only groups of at least 3 consecutive chs are used

    nChs = np.size(LFPs,0)
    nSamples = np.size(LFPs,1)

    CSD = np.zeros((nChs,nSamples))
    for chgroupi,chgroup in enumerate(chgroups):
            CSD[chgroup] = runCSD_(LFPs[chgroup])

    if smooth: CSD = smoothCSD(CSD,spacing=spacing,sm=sm)
    return CSD


def smoothCSD(csd_,spacing,sm=20):
    '''Smooths CSD.
    csd_:csd to smooth.
    spacing:spacing between channels in um
    sm:smoothing parameter
    '''

    from scipy.interpolate import interp1d

    nch = len(csd_)
    #csd_ = runCSD(trigLFP_)

    validchs = ~np.isnan(csd_[:,0])
    x = (np.arange(nch)*spacing)[validchs]

    interpolator = interp1d(x,csd_[validchs,:],kind='quadratic',axis=0)
    x_ = np.arange(x.min(),x.max()+.1)
    csd_interpl = interpolator(x_)

    csd_interpl_s = MatrixGaussianSmooth(csd_interpl.T,sm)[0].T

    interpolator = interp1d(x_,csd_interpl_s,kind='quadratic',axis=0)
    csd_s = interpolator(np.arange(nch)*spacing)

    return csd_s
# ------------------------------------------------------------------
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


def AdjustBoxPlot(ax=None,alpha=0.75,color=(.6,.6,.6)):
    '''Despines axis and adds a grid'''
    import seaborn as sns
    if ax is None:
        ax = plt.gca()
    ax.grid(True,which='major',color=color,alpha=alpha,linestyle='--')
    sns.despine(ax=ax)
    
    
def plotCSD(csd,lfp,time,spacing,zerochi=0,levels_=50,xlim=None,cmap='seismic',ax_lbl=None,ax=None):
    '''
    Plots the csd and overlays the LFP.
    time: time axis to be plotted in the x-axis
    spacing: spacing between each lfp line 
    
    levels_:colormaps used for contourf. Can be int or array.
    '''
    nchs = len(csd)

    if ax is None:
        ax = plt.gca()

    if xlim is None:
            xlim = [time.min(),time.max()]

    xlbl = 'Time from event (ms)'
    ylbl = ''
    # Check if specfic x and y lbls were inserted 
    if ax_lbl is not None:
        xlbl = ax_lbl['xlabel']
        ylbl = ax_lbl['ylabel']

    tmask = (time>=np.min(xlim))&(time<=np.max(xlim))

    if type(levels_) == int:
        climaux = np.nanmax(np.abs(csd[:,tmask]))
        levels = np.linspace(-climaux,climaux,levels_)
    else:
        levels = levels_.copy()

    csd_ = csd[:,tmask]    
    taxis_ = time[tmask]

    yaxis_ = (np.arange(nchs)-zerochi)*spacing


    cnt = ax.contourf(taxis_,yaxis_,csd_,levels,cmap=cmap,extend='both')

    aux = lfp[~np.isnan(lfp)]
    scaling = 2.75*spacing/(np.max(aux)-np.min(aux))

    for chi in range(len(lfp)):
            ax.plot(taxis_,-lfp[chi,tmask]*scaling+yaxis_[chi],'k',linewidth=1.5)

    ax.set_ylim((nchs-zerochi-1)*spacing,-zerochi*spacing)
    ax.set_ylabel(ylbl,fontsize=16)
    ax.set_xlabel(xlbl,fontsize=16)
    ax.set_yticks(yaxis_)
    ax.set_yticklabels(yaxis_,fontsize=15)
    ax.tick_params(direction='out',labelsize=15)