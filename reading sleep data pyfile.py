import mne
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#
data_path = os.path.expanduser("C:/Users/oledr/desktop/")
raw_fname = os.path.join(data_path,'Testdata.bdf')
raw = mne.io.read_raw_bdf(raw_fname, preload=False)
#raw = mne.io.read_raw_fif("C:/Users/oledr/OneDrive - NTNU/Code/EEG prosjekt/save_part_of_raw_0_100.fif",preload=False)
#
tmin, tmax = 0, 5000

# SE https://github.com/mauriceaj/CRNNeeg-sleep
# and https://academic.oup.com/sleep/advance-article/doi/10.1093/sleep/zsaa112/5849506#205167769
# Bruk dette
# filename = 'C:/Users/oledr/OneDrive - NTNU/Code/EEG prosjekt/annotations_for_sleep_data/' + str(tmin) + '_to_' + str(tmax) +'.csv'
# raw._annotations.save(filename)

#Save data fra anotations
# filename = 'C:/Users/oledr/OneDrive - NTNU/Code/EEG prosjekt/annotations_for_sleep_data/' + str(tmin) + '_to_' + str(tmax) +'.csv'
# anot = mne.read_annotations(filename)
# for a in anot.__iter__():
#     type = a['description']
#     onset = a['onset']
#     duration = a['duration']
#     tstart = round(tmin + onset)
#     tstopp = round(tstart + duration)
#     saveFileName = 'C:/Users/oledr/OneDrive - NTNU/Code/EEG prosjekt/Sleepdata_from_anot/' + type + '_' + str(tstart) + '_to_' + str(tstopp) +'.fif'
#     data.save(saveFileName, tmin=onset, tmax= onset + duration, overwrite=True)



print(raw.ch_names[:])
raw.set_channel_types({'EMG-c':'emg',
                       'EMG-r':'emg',
                       'EMG-l':'emg',
                       'ECG-r-0':'ecg',
                       'ECG-l-0': 'ecg',
                       'EOG-u':'eog',
                       'ECG-l-1':'ecg',
                       'ECG-r-1':'ecg',
                       'Ana1':'misc',
                       'Ana2':'misc',
                       'Ana3':'misc',
                       'Ana4':'misc',
                       'Ana5':'misc',
                       'Ana6':'misc',
                       'Ana7':'misc',
                       'Ana8':'misc',})
#


raw = raw.set_montage('biosemi32')

#


raw.plot_sensors(kind='topomap',ch_type='eeg', show_names=True) #2D Plot

#


data = raw.crop(tmin=tmin,tmax=tmax)
data.load_data()

#

to_drop = ['Ana1', 'Ana2', 'Ana3', 'Ana4', 'Ana5', 'Ana6', 'Ana7', 'Ana8','ECG-l-1', 'ECG-r-1', 'ECG-r-0', 'Status']
data.drop_channels(to_drop)

#


data = data.resample(100,npad='auto')

#

picks = mne.pick_types(raw.info, eeg=True, eog=True, ecg=True, emg=True)
data.filter(0.3, 40, picks=picks)
data.notch_filter(50, fir_design='firwin',picks=picks)

#


color = dict(eeg='k', eog='b', ecg='r',emg='k',)
scalings = dict(eeg=40e-6, eog=150e-6/4, ecg=5e-4, emg=1e-3/10, stim=1)
#data.plot(scalings=scalings, color=color)


# Make channels corect for AI program
# ch1 = F3-C3, ch2 = C3-O1, ch3 = F4-C4, ch4 = C4-O2
# hold = data.copy()
# hold.pick_channels(['F3','C3','O1','F4','C4','O2'])
# ch1 = hold.get_data(picks=['F3']) -hold.get_data(picks=['C3'])
# ch2 = hold.get_data(picks=['C3']) -hold.get_data(picks=['O1'])
# ch3 = hold.get_data(picks=['F4']) -hold.get_data(picks=['C4'])
# ch4 = hold.get_data(picks=['C4']) -hold.get_data(picks=['O2'])
#
# # Save to file
# ch1_file = 'C:/Users/oledr/OneDrive - NTNU/Code/EEG prosjekt/Data_for_clasify/ch1_' + str(tmin) + '_to_' + str(tmax) +'.dat'
# ch2_file = 'C:/Users/oledr/OneDrive - NTNU/Code/EEG prosjekt/Data_for_clasify/ch2_' + str(tmin) + '_to_' + str(tmax) +'.dat'
# ch3_file = 'C:/Users/oledr/OneDrive - NTNU/Code/EEG prosjekt/Data_for_clasify/ch3_' + str(tmin) + '_to_' + str(tmax) +'.dat'
# ch4_file = 'C:/Users/oledr/OneDrive - NTNU/Code/EEG prosjekt/Data_for_clasify/ch4_' + str(tmin) + '_to_' + str(tmax) +'.dat'
# # raw._annotations.save(filename)
#
# ch1.tofile(ch1_file)
# ch2.tofile(ch2_file)
# ch3.tofile(ch3_file)
# ch4.tofile(ch4_file)
#
# #
#
# start = 0
# stop = None
# duration = 30
#
# events = mne.make_fixed_length_events(data, start=start, stop=stop, duration=duration)
# epochs = mne.Epochs(data, events=events, tmin=0, tmax=0, baseline=None,verbose=True)
# times=[0.0, 0.5, 1]
# evoked = epochs.average()
#evoked.plot_topomap(ch_type='eeg',times=times, proj=True)

# epochs.plot(scalings='auto', block=True)

# raw._annotations.save('annotation_test_time.csv')
# annot_from_file = mne.read_annotations('annotation_test_time2.csv')
# annot_from_file_time = mne.read_annotations('annotation_test_time.csv')
# annot_from_file_time = mne.read_annotations('C:/Users/oledr/OneDrive - NTNU/Code/EEG prosjekt/annotation_test_time.csv')
# print(annot_from_file.orig_time)
# print(annot_from_file.onset)
# data.save('name_of_file.fif') for å save data til fif med annotations




# put in annotations
# annot_from_file_time = mne.read_annotations('C:/Users/oledr/OneDrive - NTNU/Code/EEGprosjekt/annotation_test_time.csv')
# data.set_annotations(annot_from_file_time)
# data.plot(scalings=scalings, color=color)

# Bruk dette
# filename = 'C:/Users/oledr/OneDrive - NTNU/Code/EEG prosjekt/annotations_for_sleep_data/' + str(tmin) + '_to_' + str(tmax) +'.csv'
# raw._annotations.save(filename)