import numpy as np
import pandas as pd
from segmentation import data_segmentation
from FileUtils import create_input_file

base_path = r'E:\DataBase\DB Imagined Speech KO'
sel_subjects = ['MM05', 'MM10', 'MM11', 'MM16', 'MM18', 'MM19', 'MM21', 'P02']
data_evidence = pd.read_excel(r'KO evidence v2.xlsx')
save_path = r'C:\Doctorat\GANImagSpeech\DataBase'
drop_ch = (['M1', 'M2', 'EKG', 'EMG', 'Trigger'])

# data_segmentation(base_path, sel_subjects, data_evidence)
create_input_file(save_path, drop_ch)

