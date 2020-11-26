# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

import os.path as op
import os
import mne
from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator,  compute_source_psd_epochs, apply_inverse

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
# subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

#

raw_fname, = eegbci.load_data(subject=1, runs=[6])
# raw = mne.io.read_raw_edf(raw_fname, preload=True)
data_path = "C:/Users/oledr/OneDrive - NTNU/Code/EEG prosjekt/HeadModelData_0_to_5000.fif"
raw = mne.io.read_raw_fif(data_path,preload=False)
# events = mne.find_events(raw, stim_channel='STI 014')
for i in range(raw.annotations.__len__()):
    raw.annotations.delete(0)
# Clean channel names to be able to use a standard 1005 montage
new_names = dict(
    (ch_name,
     ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp'))
    for ch_name in raw.ch_names)
raw.rename_channels(new_names)

# Read and set the EEG electrode locations
# montage = mne.channels.make_standard_montage('standard_1005')

raw = raw.set_montage('biosemi32')
raw.set_eeg_reference(projection=True)  # needed for inverse modeling

# Check that the locations of EEG electrodes is correct with respect to MRI
mne.viz.plot_alignment(
    raw.info, src=src, eeg=['original', 'projected'], trans=trans,
    show_axes=True, mri_fiducials=True, dig='fiducials')


#

fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=1)
print(fwd)
mne.write_forward_solution("C:/Users/oledr/OneDrive - NTNU/Code/EEG prosjekt/sleep-data-fwd.fif", fwd, overwrite=False, verbose=None)
# for illustration purposes use fwd to compute the sensitivity map
eeg_map = mne.sensitivity_map(fwd, ch_type='eeg', mode='fixed')
eeg_map.plot(time_label='EEG sensitivity', subjects_dir=subjects_dir,
             clim=dict(lims=[5, 50, 100]))


#

tmin = -0.2  # start of each epoch (200ms before the trigger)
tmax = 0.5  # end of each epoch (500ms after the trigger)
#raw.info['bads'] = ['MEG 2443', 'EEG 053']
baseline = (None, 0)  # means from the first instant to t = 0
# reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

events = mne.make_fixed_length_events(raw=raw,id=1,start=0,stop=None,duration=1.5,first_samp=True,overlap=0)
event_id = {'test':1}

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=('eeg'), baseline=baseline)  # , reject=reject)
evoked = epochs.average().pick('eeg')


noise_cov = mne.compute_covariance(
    epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=True)


inverse_operator = make_inverse_operator(
    evoked.info, fwd, noise_cov, loose=0.2, depth=0.8)



method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2
stc, residual = apply_inverse(evoked, inverse_operator, lambda2,
                              method=method, pick_ori=None,
                              return_residual=True, verbose=True)

# fig, ax = plt.subplots()
# ax.plot(1e3 * stc.times, stc.data[::100, :].T)
# ax.set(xlabel='time (ms)', ylabel='%s value' % method)

# MEG has 2 axes and EEG has 1? needed to change to 1 to make eeg work
# fig, axes = plt.subplots(1, 1)
# evoked.plot(axes=axes)
# # for ax in axes:
# axes.texts = []
# for line in ax.lines:
#     line.set_color('#98df81')
# residual.plot(axes=axes)

vertno_max, time_max = stc.get_peak(hemi='rh')

# subjects_dir = data_path + '/subjects'
surfer_kwargs = dict(
    hemi='rh', #subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
               font_size=14)
