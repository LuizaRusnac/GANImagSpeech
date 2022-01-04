import os

#Load cnt data from file
sel_subjects = ['MM05', 'MM10', 'MM11', 'MM16', 'MM18', 'MM19', 'MM21', 'P02']

subject = sel_subjects[0]

base_path = r'D:\DataBase'
path = base_path + r'\DB Imagined Speech KO' + '\\' + subject

text_file = [f for f in os.listdir(path) if f.endswith('.cnt')]
print(text_file)
# filename_record = os.path.join(path, str(text_file[0]))
# data = mne.io.read_raw_cnt(filename_record, eog = ['VEO','HEO'], ecg = ['EKG'], emg = ['EMG'], preload = True)