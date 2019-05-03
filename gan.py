import os
import time
from glob import glob

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Embedding, Flatten, Input, Reshape, ZeroPadding2D,
                          multiply)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (AveragePooling2D, Conv2D,
                                        Conv2DTranspose, MaxPooling2D,
                                        UpSampling2D, ZeroPadding2D)
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from tqdm import tqdm



class DCGAN():

    def __init__(self,experiment_name='experiment'):

        self.experiment = experiment_name
        self.img_rows = 512
        self.img_cols = 512
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 150
        self.no_samples = 10

        self.ensure_dirs()

        optimizer = Adam(0.0001, 0.5)

        if os.path.isfile(self.lpath("models\\dis.h5")) and os.path.isfile(self.lpath("models\\gen.h5")):
            # uncomment to load discriminator, generator
            print("Loading existing models...")
            self.discriminator = load_model(self.lpath('models\\dis.h5'))
            self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            self.generator = load_model(self.lpath('models\\gen.h5'))
        else:
            # build discriminator, generator
            self.discriminator = self.create_discriminator()
            self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
            self.generator = self.create_generator()


        # the combined model take an image as input and output validity from 0 to 1
        # note that in the combined model, the discriminator is not trainable
        self.discriminator.trainable = False
        
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        valid = self.discriminator(img)

        self.combined = Model(z, valid) 
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def lpath(self,s):
        return os.path.join(self.experiment,s)

    def ensure_dirs(self):
        print("Creating experiment %s" % self.experiment)
        for d in ['models','output','temp']:
            os.makedirs(os.path.join(self.experiment,d),exist_ok=True)

    def create_generator(self):

        model = Sequential()

        model.add(Dense(8 * 8 * 512, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 512)))

        for x in [512,256,128,64,32,16,8]:
            model.add(Conv2D(x, kernel_size=(3,3), padding="same"))
            model.add(Activation("relu"))
            model.add(BatchNormalization(momentum=0.8))
            if x!=8:
                model.add(UpSampling2D())

        model.add(Conv2D(self.channels, kernel_size=(1,1), padding="same"))
        model.add(Activation("tanh"))

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        model.summary()

        return Model(noise, img)


    def create_discriminator(self):
        
        model = Sequential()

        for i,x in enumerate([32,64,128,256]):
            if i==0:
                model.add(Conv2D(x, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
            else:
                model.add(Conv2D(x, kernel_size=3, strides=2, padding="same"))

            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
            model.add(AveragePooling2D())
            model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        model.summary()

        return Model(img, validity)

    def train(self, batch_size=128, save_interval=50, save_img_interval=50, print_interval = 10, train_path='.\\data'):
        
        #get dataset
        X_train = self.load_dataset(train_path+'\\*')

        print("Training on {} images".format(len(X_train)))

        # ones = label for real images
        # zeros = label for fake images
        ones = np.ones((batch_size, 1)) 
        zeros = np.zeros((batch_size, 1))

        # create some noise to track AI's progression
        noise_pred = []
        for _ in range(self.no_samples):
            noise_pred.append(np.random.normal(0, 1, (1, self.latent_dim)))
        self.noise_pred = noise_pred

        epoch = 0
        while(1):
            epoch+=1

            # Select a random batch of images in dataset
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            
            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)   

            # Train the discriminator with generated images and real images
            d_loss_r = self.discriminator.train_on_batch(imgs, ones)
            d_loss_f = self.discriminator.train_on_batch(gen_imgs, zeros)
            d_loss = np.add(d_loss_r , d_loss_f)*0.5

            # Trains the generator to fool the discriminator
            g_loss = self.combined.train_on_batch(noise, ones)

            #print loss and accuracy of both trains
            if epoch % print_interval == 0:
                print ("%d D loss: %f, acc.: %.2f%% G loss: %f" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % save_img_interval == 0:
                print("Saving images...", end='')
                self.save_imgs(epoch)
                print("done")
            
            if epoch % save_interval == 0:
                print("Saving models...",end='')
                # We save models twice to simplify loading - both using epoch-specific name, and using short name
                # so that dis/gen.h5 models always contain the latest trained epoch
                self.discriminator.save(self.lpath('models\\dis_'+str(epoch)+'.h5'))
                self.generator.save(self.lpath('models\\gen_'+str(epoch)+'.h5'))
                self.discriminator.save(self.lpath('models\\dis.h5'))
                self.generator.save(self.lpath('models\\gen.h5'))
                print("done")

    def save_imgs(self, epoch):
        for i,s in enumerate(self.noise_pred):
            gen_img = self.generator.predict(s)
            confidence = self.discriminator.predict(gen_img)
            # Rescale image to 0 - 255
            gen_img = (0.5 * gen_img[0] + 0.5)*255
            cv2.imwrite(self.lpath('output\\img_%d_%d_%f.png'%(i,epoch, confidence)), gen_img)

    def load_dataset(self,path):

        try:
            # try to load existing X_train
            print('Loading pre-processed dataset...',end='')
            X_train = np.load(self.lpath('temp\\data.npy'))
            print("done")

        except:
            # else, build X_train and save it
            print("Loading original dataset...",end='')
            X_train = []
            dos = glob(path)

            for i in tqdm(dos):
                try:
                    img = cv2.imread(i)
                    if img.shape[0]<self.img_rows or img.shape[1]<self.img_cols:
                        print("Image {} too small - skipping")
                    elif not(0.8<img.shape[0]/img.shape[1]<1.25):
                        print("Image {} too skew - skipping")
                    else:
                        img = cv2.resize(img,(self.img_cols, self.img_rows))
                        X_train.append(img)
                except:
                    print("Error loading {} - skipping".format(i))

            #cv2.destroyAllWindows()
            X_train = np.array(X_train)

            # Rescale dataset to -1 - 1
            X_train = X_train / 127.5 - 1
            print('...saving...',end='')
            np.save(self.lpath('temp\\data.npy'),X_train)
            print('done')
            
        return X_train


if __name__ == '__main__':
    cgan = DCGAN(experiment_name='e512')
    cgan.train(batch_size=1, save_interval=5000, save_img_interval=25, train_path='e:\\art\\data\\flower-painting-sel')
