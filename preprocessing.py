import numpy as np
import numpy.matlib

""" 
This preprocessing module contains:

sgnNorm - transform the EEG signal space into range [0, 1] over the channels

sgnStd - transform the EEG signal space having the mean 0 and std 1, over the channels

featureNorm - transform the EEG signal space into range [0, 1] over the features

featureStd - transform the EEG signal space into a space uit mean 0 and std 1 over the features

mat3d2mat2d - reshape the 3D matrix x into a 2D matrix

spWin - This function split a matrix x of dimension [nr. observations x nr. channels x nr. samples] into a matrix with dimension 
	[nr. observations * (nr. samples/window) x nr. channels x window]

"""

def sgnNorm(X):
	"""
	This function transform the EEG signal space into range [0, 1] over the channels.

	Input data:
		X - EEG signals with dimension [nr. channels x nr. samples]

	Output data:
		xnorm - Normalized X signal. Dimension: [nr. channels x nr. samples]

	The function will compute the minim and maxim over the channels (dimension: [nr. channels x 1]) and will transform the
	space using the minin and maxim values using equation:
					X = (X - minim)/(maxim-minim)
	"""

	dim = X.shape

	minim = np.reshape(X.min(axis=1),(dim[0],1))
	maxim = np.reshape(X.max(axis=1),(dim[0],1))

	xnorm = (X - numpy.matlib.repmat(minim,1,dim[1]))/numpy.matlib.repmat((maxim-minim),1,dim[1])

	xnorm = np.asarray(xnorm)

	return xnorm


def sgnStd(X):
	"""
	This function transform the EEG signal space having the mean 0 and std 1, over the channels.

	Input data:
		X - EEG signals with dimension [nr. channels x nr. samples]

	Output data:
		xstandard - Standardized X signal. Dimension: [nr. channels x nr. samples]

	The function will compute the mean and standard deviation over the channels (dimension: [nr. channels x 1]) and will transform 
	the space using the minim and maxim values using equation:
					X = (X - mean)/std
	"""
	dim = X.shape
	
	mean = np.reshape(X.mean(axis=1),(dim[0],1))
	std = np.reshape(X.std(axis=1),(dim[0],1))

	xstandard = (X - numpy.matlib.repmat(mean,1,dim[1]))/numpy.matlib.repmat(std,1,dim[1])

	return xstandard

def featureNorm(X, minim = None, maxim = None, flag = 0):
	"""
	This function transform the EEG signal space into range [0, 1] over the features.

	Input data:
		X - EEG signals with dimension [nr. observations x nr. features] or [nr. observations x nr. channels x nr. features]
		minim - if exists, the values from X will be normalized using this values of minim. Dimension: [1 x nr. features] or
		[nr. channels x nr. features]
		maxim - if exists, the values from X will be normalized using this values of maxim. Dimension: [1 x nr. features] or
		[nr. channels x nr. features]
		flag - flag takes values 0 and 1, 0 if the user don't want to return the values of minim and maxim and 1 if the user
		wich to reurn the minim and maxim values

	IMPORTANT: The function MUST receive minim AND maxim. If one is given, the other must be given too!!!

	Output data:
		xnorm - Normalized X signal. Dimension: [nr. observations x nr. features] or [nr. observations x nr. channels x nr. features]

	The function will compute the minim and maxim over the features (dimension: [1 x nr. features] or [nr. channels x nr. features])
	and will transform the space using the minim and maxim values using equation:
					X = (X - minim)/(maxim-minim)
	"""

	dim = X.shape

	if ((minim is None) and not(maxim is None)) or (not(minim is None) and (maxim is None)):
		raise AttributeError("The function MUST receive minim AND maxim. If one is given, the other must be given too!!!")

	if not(minim is None) and not(maxim is None):
		if minim.shape != maxim.shape:
			raise AttributeError("Minim and maxim must be the same length")

		if len(dim)==2:
			if dim[1]!=minim.shape[1]:
				raise AttributeError("X features and minim must be the same length")

			if dim[1]!=maxim.shape[1]:
				raise AttributeError("X features and maxim must be the same length")

			xnorm = (X - numpy.matlib.repmat(minim,dim[0],1))/numpy.matlib.repmat((maxim-minim),dim[0],1)
			return xnorm

		if len(dim)==3:
			if dim[1]!=minim.shape[0] or dim[2]!=minim.shape[1]:
				raise AttributeError("X features and minim/maxim must be the same length")

			xnorm = (X - minim)/(maxim-minim)
			return xnorm
	else:
		if len(dim)==2:
			minim = np.reshape(X.min(axis=0),(1,dim[1]))
			maxim = np.reshape(X.max(axis=0),(1,dim[1]))

			xnorm = (X - numpy.matlib.repmat(minim,dim[0],1))/numpy.matlib.repmat((maxim-minim),dim[0],1)

		elif len(dim)==3:
			minim = X.min(axis=0)
			maxim = X.max(axis=0)

			xnorm = (X - minim)/(maxim-minim)
		else:
			raise ValueError("Too many dimensions for X!")

		if flag==0:
				return xnorm
		elif flag==1:
			return xnorm, minim, maxim
		else:
			raise ValueError("It's not a valid flag value!")

def featureStd(X, mean = None, std = None, flag = 0):
	"""
	This function transform the EEG signal space into a space uit mean 0 and std 1 over the features.

	Input data:
		X - EEG signals with dimension [nr. observations x nr. features] or [nr. observations x nr. channels x nr. features]
		mean - if exists, the values from X will be standardized using this values of mean. Dimension: [1 x nr. features] or
		[nr. channels x nr. features]
		std - if exists, the values from X will be standardized using this values of std. Dimension: [1 x nr. features] or
		[nr. channels x nr. features]
		flag - flag takes values 0 and 1, 0 if the user don't want to return the values of mean and std and 1 if the user
		wish to return the mean and std values

	IMPORTANT: The function MUST receive mean AND std. If one is given, the other must be given too!!!

	Output data:
		xstd - Standardized X signal. Dimension: [nr. observations x nr. features] or [nr. observations x nr. channels x nr. features]

	The function will compute the mmean and std over the features (dimension: [1 x nr. features] or [nr. channels x nr. features])
	and will transform the space using the mean and std values using equation:
					X = (X - mean)/std
	"""

	dim = X.shape

	if ((mean is None) and not(std is None)) or (not(mean is None) and (std is None)):
		raise AttributeError("The function MUST receive mean AND std. If one is given, the other must be given too!!!")

	if not(mean is None) and not(std is None):
		if mean.shape != std.shape:
			raise AttributeError("Minim and maxim must be the same length")

		if len(dim)==2:
			if dim[1]!=mean.shape[1]:
				raise AttributeError("X features and mean must be the same length")

			if dim[1]!=std.shape[1]:
				raise AttributeError("X features and std must be the same length")

			xstd = (X - numpy.matlib.repmat(mean,dim[0],1))/numpy.matlib.repmat(std,dim[0],1)
			return xstd

		if len(dim)==3:
			if dim[1]!=mean.shape[0] or dim[2]!=std.shape[1]:
				raise AttributeError("X features and mean/std must be the same length")

			xstd = (X - mean)/std
			return xstd
	else:
		if len(dim)==2:
			mean = np.reshape(X.mean(axis=0),(1,dim[1]))
			std = np.reshape(X.std(axis=0),(1,dim[1]))

			xstd = (X - numpy.matlib.repmat(mean,dim[0],1))/numpy.matlib.repmat(std,dim[0],1)

		elif len(dim)==3:
			mean = X.mean(axis=0)
			std = X.std(axis=0)

			xstd = (X - mean)/std
		else:
			raise ValueError("Too many dimensions for X!")

		if flag==0:
				return xstd
		elif flag==1:
			return xstd, mean, std
		else:
			raise ValueError("It's not a valid flag value!")

def mat3d2mat2d(x):
	"""
	This function reshape the 3D matrix x into a 2D matrix.

	Input data:
		x - A 3D matrix

	Output data:
		xm - A 2D matrix

	"""
	dim = x.shape
	xm = np.zeros((dim[0],dim[1]*dim[2]))
	for i in range(dim[0]):
	    xm[i,:] = np.reshape(x[i,:,:],(1,dim[1]*dim[2]))

	return xm

def mat2d2mat3d(x,n,m):
	"""
	This function reshape the 3D matrix x into a 2D matrix.

	Input data:
		x - A 2D matrix

	Output data:
		xm - A 3D matrix

	"""
	dim = x.shape
	xm = np.zeros((dim[0],n,m))
	for i in range(dim[0]):
	    xm[i,:,:] = np.reshape(x[i,:],(1,n,m))

	return xm

def spWin(x, window, y=None):
	"""
	This function split a matrix x of dimension [nr. observations x nr. channels x nr. samples] into a matrix with dimension 
	[nr. observations * (nr. samples/window) x nr. channels x window]

	Input data:
		x - A 3D matrix of dimension [nr. observations x nr. channels x nr. samples]
		window - The numbers of samples of one window
		y - the data target, if needed
		
	Output data:
		xsplit - the splited matrix x over the desired window Dimension: [nr. observations * (nr. samples/window) x nr. channels x window]
		ysplit - only if y is provided, which contains the new target for the splitted matrix

	"""
	dim = x.shape

	nr_recf = int(dim[0]*(dim[2]/window))

	xsplit = np.zeros((nr_recf,dim[1],window))

	if not(y is None):
		if dim[0]!=len(y):
			raise AttributeError("The length of y must match the first dimension of x!")
		ysplit = np.zeros((nr_recf,1))

	i = 0
	for rec,j in zip(x,range(len(x))):
		for win in range(0,len(rec[0]),window):
			print(win)
			xsplit[i,:,:] = rec[:,win:win+window]

			if not(y is None):
				ysplit[i,0] = y[j]
			i+=1

	if not(y is None):
		return xsplit,ysplit
	else:
		return xsplit

def featureNormRange(X, minim = None, maxim = None, flag = 0, rng = [-1, 1]):
	"""
	This function transform the EEG signal space into range [0, 1] over the features.

	Input data:
		X - EEG signals with dimension [nr. observations x nr. features] or [nr. observations x nr. channels x nr. features]
		minim - if exists, the values from X will be normalized using this values of minim. Dimension: [1 x nr. features] or
		[nr. channels x nr. features]
		maxim - if exists, the values from X will be normalized using this values of maxim. Dimension: [1 x nr. features] or
		[nr. channels x nr. features]
		flag - flag takes values 0 and 1, 0 if the user don't want to return the values of minim and maxim and 1 if the user
		wich to reurn the minim and maxim values

	IMPORTANT: The function MUST receive minim AND maxim. If one is given, the other must be given too!!!

	Output data:
		xnorm - Normalized X signal. Dimension: [nr. observations x nr. features] or [nr. observations x nr. channels x nr. features]

	The function will compute the minim and maxim over the features (dimension: [1 x nr. features] or [nr. channels x nr. features])
	and will transform the space using the minim and maxim values using equation:
					X = (X - minim)/(maxim-minim)
	"""

	dim = X.shape

	if ((minim is None) and not(maxim is None)) or (not(minim is None) and (maxim is None)):
		raise AttributeError("The function MUST receive minim AND maxim. If one is given, the other must be given too!!!")

	if not(minim is None) and not(maxim is None):
		if minim.shape != maxim.shape:
			raise AttributeError("Minim and maxim must be the same length")

		if len(dim)==2:
			if dim[1]!=minim.shape[1]:
				raise AttributeError("X features and minim must be the same length")

			if dim[1]!=maxim.shape[1]:
				raise AttributeError("X features and maxim must be the same length")

			xnorm = (X - numpy.matlib.repmat(minim,dim[0],1))/numpy.matlib.repmat((maxim-minim),dim[0],1)*(rng[1]-rng[0]) + rng[0]
			return xnorm

		if len(dim)==3:
			if dim[1]!=minim.shape[0] or dim[2]!=minim.shape[1]:
				raise AttributeError("X features and minim/maxim must be the same length")

			xnorm = (X - minim)/(maxim-minim)*(rng[1]-rng[0]) + rng[0]
			return xnorm
	else:
		if len(dim)==2:
			minim = np.reshape(X.min(axis=0),(1,dim[1]))
			maxim = np.reshape(X.max(axis=0),(1,dim[1]))

			xnorm = (X - numpy.matlib.repmat(minim,dim[0],1))/numpy.matlib.repmat((maxim-minim),dim[0],1)*(rng[1]-rng[0]) + rng[0]

		elif len(dim)==3:
			minim = X.min(axis=0)
			maxim = X.max(axis=0)

			xnorm = (X - minim)/(maxim-minim)*(rng[1]-rng[0]) + rng[0]
		else:
			raise ValueError("Too many dimensions for X!")

		if flag==0:
			return xnorm
		elif flag==1:
			return xnorm, minim, maxim
		else:
			raise ValueError("It's not a valid flag value!")