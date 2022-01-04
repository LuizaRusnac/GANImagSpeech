import os
import mne
import numpy as np
import pandas as pd

#Load cnt data from file
sel_subjects = ['MM05', 'MM10', 'MM11', 'MM16', 'MM18', 'MM19', 'MM21', 'P02']
data_evidence = pd.read_excel(r'KO evidence v2.xlsx')

for sub in range(len(sel_subjects)):
	subject = sel_subjects[sub]

	base_path = r'D:\DataBase'
	path = base_path + r'\DB Imagined Speech KO' + '\\' + subject

	text_file = [f for f in os.listdir(path) if f.endswith('.cnt')]
	filename_record = path + '\\' + text_file[0]
	data = mne.io.read_raw_cnt(filename_record, eog = ['VEO','HEO'], ecg = ['EKG'], emg = ['EMG'], preload = True)
	data.notch_filter(np.array((60,120,180,240)))

	subject_df = data_evidence[[col for col in data_evidence if col.startswith(subject)]]
	subject_df = subject_df.loc[subject_df[subject_df.columns[-1]] == 1]
	subject_df = subject_df.reset_index(drop=True)

	for i in range(0,len(subject_df)):
	    if i<10:
	        filename = 'imagined_speech_' + subject + '_0' + str(i) + '_tag' + str(subject_df[subject + '_tag'][i]) + '.raw.fif'
	    else:
	        filename = 'imagined_speech_' + subject + '_' + str(i)  + '_tag' + str(subject_df[subject + '_tag'][i]) + '.raw.fif'
	    
	    os.chdir(r'C:\D\Doctorat\GANImagSpeech\DataBase')
	    imagined_speech = data.save(filename, tmin = subject_df[subject + '_start'][i]/1000, tmax = subject_df[subject + '_stop'][i]/1000)
    