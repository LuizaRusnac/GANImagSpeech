from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
import numpy as np
import preprocessing
import featureExtr
# import sklearn
# import sklearn.metrics
from sklearn.metrics import recall_score, accuracy_score, precision_score
import matplotlib.pyplot as plt
import itertools
import time

# define the standalone discriminator model
def define_discriminator(in_shape=(62,62,1), n_classes=11, conv_layers = [128, 128], dropout = 0.4, fact_fnc = 'relu', loss = 'mse', metrics = 'accuracy'):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 62)(in_label)
	# scale up to image dimensions with linear activation
	n_nodes = in_shape[0] * in_shape[1]
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((in_shape[0], in_shape[1], 1))(li)
	# image input
	in_image = Input(shape=in_shape)
	# concat label as a channel
	merge = Concatenate()([in_image, li])
	# downsample
	fe = Conv2D(conv_layers[0], (3,3), strides=(2,2), padding='same')(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(conv_layers[1], (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(dropout)(fe)
	# output
	out_layer = Dense(n_classes, activation=fact_fnc)(fe)
	# define model
	model = Model([in_image, in_label], out_layer)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=loss, optimizer=opt, metrics=[metrics])
	return model

# define the standalone generator model
def define_generator(latent_dim, n_classes=11):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 62)(in_label)
	# linear multiplication
	n_nodes = 31 * 31
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((31,31, 1))(li)
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 128 * 31 * 31
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((31, 31, 128))(gen)
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	# upsample to 14x14
	gen = Conv2DTranspose(128, (4,4), strides=(1,1), padding='same')(merge)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Conv2D(1, (31,31), activation='tanh', padding='same')(gen)
	# define model
	model = Model([in_lat, in_label], out_layer)
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, loss = 'mse', metrics = 'accuracy'):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# get noise and label inputs from generator model
	gen_noise, gen_label = g_model.input
	# get image output from the generator model
	gen_output = g_model.output
	# connect image output and label input from generator as inputs to discriminator
	gan_output = d_model([gen_output, gen_label])
	# define gan model as taking noise and label and outputting a classification
	model = Model([gen_noise, gen_label], gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=loss, optimizer=opt, metrics = [metrics])
	return model

def generate_real_samples(dataset, n_samples):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	y = label2mat(labels)
	return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=11):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = label2mat(labels_input)
	return [images, labels_input], y

def label2mat(label, nr_cls=11):
	matl = np.zeros((len(label),nr_cls))
	for i in range(len(label)):
		matl[i,int(label[i])] = 1
	return matl
 
 # train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, tdataset, n_epochs=5, n_batch=128):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	history = np.zeros((2, n_epochs))
	history_batch = np.zeros((2, n_batch, n_epochs))
	thistory = np.zeros((2, n_epochs))
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			start = time.time()
			# get randomly selected 'real' samples
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
			# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
			# generate 'fake' examples
			
			[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
			# prepare points in latent space as input for the generator
			[z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
			# print("************")
			# print(z_input.shape)
			# create inverted labels for the fake samples
			y_gan = label2mat(labels_input)
			#y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss, acc = gan_model.train_on_batch([z_input, labels_input], y_gan)
			history_batch[0,j,i] = g_loss
			history_batch[1,j,i] = acc
			# summarize loss on this batch
			print('>%d/%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f ----- acc: %.3f' %
				(i+1,n_epochs, j+1,bat_per_epo, d_loss1, d_loss2, g_loss, acc))
			end = time.time()
			print("Timpul:")
			print(end - start)

		images, labels = dataset
		y = label2mat(labels)
		history[0][i], history[1][i] = d_model.evaluate(dataset,y)
		# print("##### Train dataset id: \n")
		# print(dataset)
		print('>>%d/%d, Tain loss: %.3f, Train acc: %.3f'%(n_epochs, i+1, history[0][i], history[1][i]))

		timages, tlabels = tdataset
		y = label2mat(tlabels)
		thistory[0][i],thistory[1][i] = d_model.evaluate(tdataset,y)
		predict = d_model.predict(tdataset)
		print("##### Test dataset predict: \n")
		print(predict)
		print('>>%d/%d, Test loss: %.3f, Test acc: %.3f'%(n_epochs, i+1, thistory[0][i], thistory[1][i]))

	return history,thistory,history_batch

def save_plot(examples, n, name):
	# plot images
	for i in range(n * n):
		# define subplot
		plt.subplot(n, n, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(examples[i, :, :, 0])
	plt.savefig(name)
	plt.show()

def plot_confusion_matrix(cm, class_names):
	"""
	Returns a matplotlib figure containing the plotted confusion matrix.

	Args:
	cm (array, shape = [n, n]): a confusion matrix of integer classes
	class_names (array, shape = [n]): String names of the integer classes
	"""
	figure = plt.figure(figsize=(8, 8))
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title("Confusion matrix")
	plt.colorbar()
	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names, rotation=45)
	plt.yticks(tick_marks, class_names)

	# Normalize the confusion matrix.
	#cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
	cm = np.around(cm.astype('int'), decimals=2)

	# Use white text if squares are dark; otherwise black.
	threshold = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		color = "white" if cm[i, j] > threshold else "black"
		plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()
	return figure

def plot_history_loss(history):
	plt.figure()
	plt.plot(history[0], label='loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.show()

def plot_history_acc(history):
	plt.figure()
	plt.plot(history[1], label='loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.show()
