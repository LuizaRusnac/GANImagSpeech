import numpy as np
from FileUtils import load_data
from FileUtils import split_leaveOneOut
import preprocessing
import featureExtr

x, y = load_data("KaraOne_EEGSpeech_X_noLPF.npy","KaraOne_EEGSpeech_y_noLPF.npy")

# sel_subjects = ['MM05', 'MM10', 'MM11', 'MM16', 'MM18', 'MM19', 'MM21', 'P02']
# subj_idx_start = np.array([0, 120, 236, 362, 493, 603, 734, 864])
# subj_idx_stop = np.array([119, 235, 361, 492, 602, 733, 863, 993])

idxtrain = np.array([0, 361, (361 + 131), 993])
idxtest = np.array([361, (361 + 131)])

xtrain, ytrain, xtest, ytest = split_leaveOneOut(x, y, idxtrain, idxtest)

window = 1000

xtrain, ytrain = preprocessing.spWin(xtrain, window, ytrain)
xtest, ytest = preprocessing.spWin(xtest, window, ytest)

print(xtrain.shape)
print(xtest.shape)

xtrain = featureExtr.chConv(xtrain)
xtest = featureExtr.chConv(xtest)

xtrain, mean, std = preprocessing.featureStd(xtrain, flag = 1)
xtest = preprocessing.featureStd(xtest, mean = mean, std = std)

np.save('xtrain', xtrain)
np.save('ytrain', ytrain)
np.save('xtest', xtest)
np.save('ytest', ytest)