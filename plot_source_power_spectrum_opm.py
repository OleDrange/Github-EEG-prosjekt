"""
.. _ex-opm-resting-state:

======================================================================
Compute source power spectral density (PSD) of VectorView and OPM data
======================================================================

Here we compute the resting state from raw for data recorded using
a Neuromag VectorView system and a custom OPM system.
The pipeline is meant to mostly follow the Brainstorm [1]_
`OMEGA resting tutorial pipeline <bst_omega_>`_.
The steps we use are:

1. Filtering: downsample heavily.
2. Artifact detection: use SSP for EOG and ECG.
3. Source localization: dSPM, depth weighting, cortically constrained.
4. Frequency: power spectral density (Welch), 4 sec window, 50% overlap.
5. Standardize: normalize by relative power for each source.

.. contents::
   :local:
   :depth: 1

.. _bst_omega: https://neuroimage.usc.edu/brainstorm/Tutorials/RestingOmega

Preprocessing
-------------
"""
# sphinx_gallery_thumbnail_number = 13

# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Luke Bloy <luke.bloy@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

from mne.filter import next_fast_len
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import mne
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from matplotlib.colors import LinearSegmentedColormap


new_sfreq = 90.  # Nyquist frequency (45 Hz) < line noise freq (50 Hz)


import os
filename = dict()

directory = r'C:/Users/oledr/OneDrive - NTNU/Code/EEG prosjekt/Sleepdata_from_anot'
for entry in os.scandir(directory):
    if entry.path.endswith(".fif") and entry.is_file():
        key = entry.path[len(directory)+1:len(directory)+3]
        filename.setdefault(key,[]).append(entry.path)
raws = dict()
#filename="C:/Users/oledr/OneDrive - NTNU/Code/EEG prosjekt/Sleepdata_from_anot/N3_2703_to_3180.fif"
for key, values in filename.items():
    for value in values:
        raws.setdefault(key, []).append(mne.io.read_raw_fif(value,preload=False))
#raws['vv'] = mne.io.read_raw_fif(filename,preload=False)

# clear data for annotations
#for i in range(raws['vv'].annotations.__len__()):
#    raws['vv'].annotations.delete(0)

reduced_channels = ['F3','F4','C3','C4', 'P3', 'P4', 'O1', 'O2']
reduced_channels_16 = ['F3','F4','C3','C4', 'P3', 'P4', 'O1', 'O2','AF3','AF4', 'FC1', 'FC2', 'CP1', 'CP2', 'PO3', 'PO4']
reduced_channels_x = ['F7','F8','FC1','FC2','CP1','CP2','P7','P8']
reduced_channels_x_16 = ['F7','F8','FC1','FC2','CP1','CP2','P7','P8','AF3','AF4','Fz','T7','T8','Pz','O2','O1']
raws_reduced_channels = dict()
for key, values in raws.items():
    for value in values:
        raws_reduced_channels.setdefault(key, []).append(value.pick(reduced_channels_x))



# Load forward model

fwd = mne.read_forward_solution("C:/Users/oledr/OneDrive - NTNU/Code/EEG prosjekt/sleep-data-fwd.fif", verbose=None)


##############################################################################
# Explore data
kinds = []
#kinds.append('vv') #, 'opm')
kinds.extend(('AW','N1','N2','N3','RE'))
n_fft = next_fast_len(int(round(4 * new_sfreq)))



freq_bands = dict(
    delta=(1, 4), theta=(4, 8), alpha=(8, 12), beta=(14, 30), #gamma=(30, 60), #high_gamma=(60, 100),
    slow_wave=(0.2, 1.2),sigma=(12, 15))
#topos = dict(vv=dict())#, opm=dict())
#stcs = dict(vv=dict())#, opm=dict())
#titles = dict(vv='VV')
titles = dict(AW='AWAKE',N1='N1',N2='N2',N3='N3',RE='REM')
topos = dict(AW=dict(),N1=dict(),N2=dict(),N3=dict(),RE=dict())
stcs = dict(AW=dict(),N1=dict(),N2=dict(),N3=dict(),RE=dict())
snr = 3.
lambda2 = 1. / snr ** 2

# Make stcs (PSD)
for kind in kinds:
    i = 0
    # if kind != 'AW':
    #     continue
    for dataset in raws_reduced_channels[kind]: # raws or raws_reduced_channels
        noise_cov = mne.compute_raw_covariance(dataset)
        inverse_operator = mne.minimum_norm.make_inverse_operator(
            dataset.info, forward=fwd, noise_cov=noise_cov, verbose=True)
        stc_psd, sensor_psd = mne.minimum_norm.compute_source_psd(
            dataset, inverse_operator, lambda2=lambda2,
            n_fft=n_fft, dB=False, return_sensor=True, verbose=True, method='MNE') # DENNE TAR LANG TID
        topo_norm = sensor_psd.data.sum(axis=1, keepdims=True)
        stc_norm = stc_psd.sum()  # same operation on MNE object, sum across freqs
        # Normalize each source point by the total power across freqs
        if not i:
            stc_data = stc_psd.data.copy()
            stc_norm_hold = stc_norm.copy()
            sensor_data = sensor_psd.data.copy()
            topo_norm_hold = topo_norm.copy()
        else:
            stc_data = np.add(stc_data,stc_psd.data)
            stc_norm_hold = np.add(stc_norm_hold,stc_norm)
            sensor_data = np.add(sensor_data,sensor_psd.data)
            topo_norm_hold = np.add(topo_norm_hold,topo_norm)
        i += 1
    stc_psd.data = stc_data/ len(raws[kind])
    sensor_psd.data = sensor_data/ len(raws[kind])
    stc_norm = stc_norm_hold/ len(raws[kind])
    topo_norm = topo_norm_hold/ len(raws[kind])

    for band, limits in freq_bands.items():
        data = sensor_psd.copy().crop(*limits).data.sum(axis=1, keepdims=True)
        topos[kind][band] = mne.EvokedArray(100 * data / topo_norm, sensor_psd.info)
        stcs[kind][band] = 100 * stc_psd.copy().crop(*limits).sum() / stc_norm.data

# OLD PLOT FUNCTIONS
# def plot_band(kind, band):
#     """Plot activity within a frequency band on the subject's brain."""
#     title = "%s %s\n(%d-%d Hz)" % ((titles[kind],band) + freq_bands[band])
#     topos[kind][band].plot_topomap(
#         times=0., scalings=1., cbar_fmt='%0.1f', vmin=0, cmap='inferno',
#         time_format=title)
#     #brain[kind][band] = stcs[kind][band]
#     brain_plot = stcs[kind][band].plot(
#         #subject=subject, subjects_dir=subjects_dir,
#         views='cau', hemi='both',
#         time_label=title, title=title, colormap='inferno',
#         time_viewer=False, show_traces=False,
#         clim=dict(kind='percent', lims=(70, 85, 99)), smoothing_steps=10)
#     brain_plot.show_view(dict(azimuth=0, elevation=0), roll=0)
#
#
# def plot_sleepstate_relative_to_awakeState(kind,band):
#     """Plot activity within a frequency band on the subject's brain."""
#     title = "%s %s Relative to Awake\n(%d-%d Hz)" % ((titles[kind],band) + freq_bands[band])
#     new_topo = minus(topos[kind][band],topos['AW'][band])
#     new_stc = minus(stcs[kind][band], stcs['AW'][band])
#     new_topo.plot_topomap(
#         times=0., scalings=1., cbar_fmt='%0.1f', vmin=0,#, cmap='inferno',
#         time_format=title)
#     new_stc.plot(
#         views='cau', hemi='both',
#         time_label=title, title=title#, colormap='inferno',
#         #time_viewer=False#,# show_traces=False,
#         #clim=dict(kind='percent', lims=(70, 85, 99)), smoothing_steps=10
#                 )



withreference = ''
def plot_report(kind,stcs_in):
    screenshots = dict()
    # cmap = cm.get_cmap('plasma')
    # alphas = np.abs(np.linspace(-1.0, 1.0, cmap.N))
    # cmap._init()
    # cmap._lut[:-3, -1] = alphas
    colormap = 'bwr'
    clim = dict()

    for band in freq_bands:
        #if stcs_in[kind][band].data.min() < 0:
        hold = round(np.mean(stcs_in[kind][band].data))
        if withreference != '':
            value_mean = 0
            maximum = 2#2*hold+2
            minimum = -2#-2*hold-2
        else:
            value_mean = hold
            maximum = value_mean + value_mean / 2
            minimum = value_mean - value_mean / 2

        clim[band] = dict(kind='value', lims=[minimum, value_mean, maximum])
        brainImage = stcs_in[kind][band].plot(views = 'lat', hemi = 'split', size = (800, 400),
        background = 'w',  clim=clim[band], colormap=colormap,
        colorbar = False, time_viewer = False, show_traces = False)
        screenshots[band] = brainImage.screenshot()
        brainImage.close()
        nonwhite_pix = (screenshots[band] != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        screenshots[band] = screenshots[band][nonwhite_row][:, nonwhite_col]

    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(6, 13))
    #fig.tight_layout(pad=.3)
    fig.suptitle(titles[kind]+withreference,size=15)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    for i,((key, image), (key2, title)) in enumerate(zip(screenshots.items(), freq_bands.items())):
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(key2 +' {} Hz'.format(title),size=12)
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes('right', size='5%', pad=0.2)
        cax.tick_params(labelsize=10)
        cbar = mne.viz.plot_brain_colorbar(cax, clim[key2], colormap=colormap,label='')



def clear_unimportant_values(stc_in,kind,freq):
    for i in range(len(stc_in[kind][freq].data)):
        hold = stc_in['N2']['alpha'].data[i]
        if abs(hold) < abs(np.mean(hold) * 2):
            stc_in['N2']['alpha'].data[i] = 0

def minus(data1,data2):
    itemOut = data1 - data2
    return itemOut

def get_stcs_relative_to_AWAKE():
    new_stcs = stcs.copy()
    for kind in kinds:
        if kind == 'AW':
            continue
        for freq_band in freq_bands:
            new_stcs[kind][freq_band].data = minus(stcs[kind][freq_band].data, stcs['AW'][band].data)

    return new_stcs

for kind in kinds:
    plot_report(kind,stcs)

new_stcs = get_stcs_relative_to_AWAKE()
# withreference = ' with reference to AWAKE'
# for kind in kinds:
#     plot_report(kind,new_stcs)


# END OF SCRIPT







# plot_report('N1',new_stcs)
# plot_report('N2',new_stcs)
# plot_report('N3',new_stcs)
# plot_report('RE',new_stcs)
# brain_plot = new_stcs['N2']['alpha'].plot(
#         views='cau', hemi='both',
#         colormap='bwr',
#         time_viewer=True, show_traces=True,
#         clim=dict(kind='value', lims=(-20, 0, 20)), smoothing_steps=10)
#
# mean_val = np.mean(new_stcs['N2']['alpha'].data)
# antal = 0
# antal0 = 0


#
#
# colormap = 'bwr'
#
# # plot the evoked in the desired subplot, and add a line at peak activation
# screenshots = dict()
# value_mean = np.mean(stcs['AW']['alpha'].data)
# maximum = value_mean + value_mean/2
# minimum = value_mean - value_mean/2
#
# #clim = dict(kind='percent', pos_lims=(70, 85, 99))
# clim = dict(kind='value', lims=[minimum, value_mean, maximum])
# brainImage = stcs['AW']['alpha'].plot(views = 'lat', hemi = 'split', size = (800, 400),
#         background = 'w',  clim=clim, colormap=colormap,
#         colorbar = False, time_viewer = False, show_traces = False)
# screenshots[test] = brainImage.screenshot()
# brainImage.close()
# nonwhite_pix = (screenshots[test] != 255).any(-1)
# nonwhite_row = nonwhite_pix.any(1)
# nonwhite_col = nonwhite_pix.any(0)
# screenshots[test] = screenshots[test][nonwhite_row][:, nonwhite_col]
# # now add the brain to the lower axes
# fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(7, 15))
#
# for i in range(7):
#     axes[i].imshow(screenshots[test])
#     axes[i].axis('off')
#     divider = make_axes_locatable(axes[i])
#     cax = divider.append_axes('right', size='5%', pad=0.2)
#     cbar = mne.viz.plot_brain_colorbar(cax, clim,colormap=colormap, label='(F)')
#
#

# delta = (1, 4), theta = (4, 8), alpha = (8, 12), beta = (14, 30), gamma = (30, 60),  # high_gamma=(60, 100),
# slow_wave = (0.2, 1.2), sigma = (12, 15))
#fig_theta, brain_theta = plot_band('vv', 'theta')

###############################################################################
# Alpha
# -----

#fig_alpha, brain_alpha = plot_band('vv', 'alpha')

###############################################################################
# Beta
# ----
# Here we also show OPM data, which shows a profile similar to the VectorView
# data beneath the sensors. VectorView first:

#fig_beta, brain_beta = plot_band('vv', 'beta')

###############################################################################
# Then OPM:
#fig_beta_opm, brain_beta_opm = plot_band('opm', 'beta')

###############################################################################
# Gamma
# -----

#fig_gamma, brain_gamma = plot_band('vv', 'gamma')

###############################################################################
# References
# ----------
# .. [1] Tadel F, Baillet S, Mosher JC, Pantazis D, Leahy RM.
#        Brainstorm: A User-Friendly Application for MEG/EEG Analysis.
#        Computational Intelligence and Neuroscience, vol. 2011, Article ID
#        879716, 13 pages, 2011. doi:10.1155/2011/879716
