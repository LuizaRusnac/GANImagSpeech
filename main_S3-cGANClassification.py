from FileUtils import load_data
import CGAN

xtrain, ytrain = load_data('xtrain.npy', 'ytrain.npy')
xtest, ytest = load_data('xtrain.npy', 'ytrain.npy')

dataset = [xtrain, ytrain]
tdataset = [xtest, ytest]

dim = xtrain.shape
tdim = xtest.shape

# size of the latent space
latent_dim = 1000
# create the discriminator
d_model = CGAN.define_discriminator(in_shape=(dim[1],dim[2],1))
# create the generator
g_model = CGAN.define_generator(latent_dim)
# create the gan
gan_model = CGAN.define_gan(g_model, d_model)
gan_model.summary()

# train model
history, thistory,history_batch = CGAN.train(g_model, d_model, gan_model, dataset, latent_dim, tdataset, n_epochs = 50)

gan_model.save('cgan_generator_leaveOneOut_MM16.h5')
g_model.save('generator_model_leaveOneOut_MM16.h5')
d_model.save('discriminator_model_leaveOneOut_MM16.h5')
np.save('history_leaveOneOut_MM16.npy', history)
np.save('thistory_leaveOneOut_MM16.npy', thistory)
np.save('history_batch_leaveOneOut_MM16.npy', history_batch)