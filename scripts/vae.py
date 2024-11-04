import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.python.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.python.keras.layers import MaxPool2D, UpSampling2D
from tensorflow.python.keras.layers import Reshape, GlobalAveragePooling2D
from tensorflow.python.keras.layers import Layer, Dense
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from PIL import Image
import numpy as np


class SampleLayer(Layer):
    '''
    Keras Layer to grab a random sample from a distribution (by multiplication)
    Computes "(normal)*stddev + mean" for the vae sampling operation
    (written for tf backend)
    Additionally,
        Applies regularization to the latent space representation.
        Can perform standard regularization or B-VAE regularization.
    call:
        pass in mean then stddev layers to sample from the distribution
        ex.
            sample = SampleLayer('bvae', 16)([mean, stddev])
    '''
    def __init__(self, latent_regularizer='bvae', beta=5., 
                 capacity=128., randomSample=True, **kwargs):
        '''
        args:
        ------
        latent_regularizer : str
            Either 'bvae', 'vae', or None
            Determines whether regularization is applied
                to the latent space representation.
        beta : float
            beta > 1, used for 'bvae' latent_regularizer,
            (Unused if 'bvae' not selected)
        capacity : float
            used for 'bvae' to try to break input down to a set number
                of basis. (e.g. at 25, the network will try to use 
                25 dimensions of the latent space)
            (unused if 'bvae' not selected)
        randomSample : bool
            whether or not to use random sampling when selecting from 
                distribution.
            if false, the latent vector equals the mean, essentially turning 
                this into a standard autoencoder.
        ------
        ex.
            sample = SampleLayer('bvae', 16)([mean, stddev])
        '''
        self.reg = latent_regularizer
        self.beta = beta
        self.capacity = capacity
        self.random = randomSample
        super(SampleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # save the shape for distribution sampling
        self.shape = input_shape[0]

        super(SampleLayer, self).build(input_shape) # needed for layers

    def call(self, x):
        if len(x) != 2:
            raise Exception('input layers must be a list: mean and stddev')
        if len(x[0].shape) != 2 or len(x[1].shape) != 2:
            raise Exception('input shape is not a vector [batchSize, latentSize]')

        mean = x[0]
        stddev = x[1]

        if self.reg:
            # kl divergence:
            latent_loss = -0.5 * K.mean(1 + stddev
                                        - K.square(mean)
                                        - K.exp(stddev), axis=-1)        
    
            if self.reg == 'bvae':
                # use beta to force less usage of vector space:
                # also try to use <capacity> dimensions of the space:
                latent_loss = self.beta * K.abs(latent_loss - self.capacity/self.shape.as_list()[1])

            self.add_loss(latent_loss, x)

        epsilon = K.random_normal(shape=self.shape,
                              mean=0., stddev=1.)
        if self.random:
            # 'reparameterization trick':
            return mean + K.exp(stddev / 2) * epsilon
        else: # do not perform random sampling, simply grab the impulse value
            return mean + 0*stddev # Keras needs the *0 so the gradinent is not None

    def compute_output_shape(self, input_shape):
        return input_shape[0]



class Encoder():
    
    def __init__(self, input_shape, latent_size, 
                 batch_size=4, width=1, base_size=16):
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.batch_size = batch_size        
        self.width = width
        self.base_size = base_size
        
    def build(self,):
        model_input = Input(shape=self.input_shape, batch_size=self.batch_size)
        
        # first block
        model = Conv2D(filters=1*self.base_size*self.width, 
                       kernel_size=(3, 3), strides=(1, 1),
                       padding='same', 
                       activation='elu')(model_input)
        model = MaxPool2D(pool_size=(2, 2))(model)
        
        # second block
        model = Conv2D(filters=2*self.base_size*self.width, 
                           kernel_size=(3, 3), strides=(1, 1),
                           padding='same', 
                           activation='elu')(model_input)
        model = MaxPool2D(pool_size=(2, 2))(model)        
        
        # third block
        model = Conv2D(filters=3*self.base_size*self.width, 
                            kernel_size=(3, 3), strides=(1, 1),
                            padding='same', activation='elu')(model)
        model = MaxPool2D(pool_size=(2, 2))(model)
        
        # variational encoder output (distributions)
        mean = Conv2D(filters=self.latent_size, kernel_size=(1, 1),
                      padding='same', activation='sigmoid')(model)
        mean = GlobalAveragePooling2D()(mean)
        
        stddev = Conv2D(filters=self.latent_size, kernel_size=(1, 1),
                        padding='same', activation='relu')(model)
        stddev = GlobalAveragePooling2D()(stddev)
    
        model_output = SampleLayer(latent_regularizer='vae')([mean, stddev])
        
        return Model(inputs=[model_input], outputs=[model_output])


class Decoder():
    
    def __init__(self, input_shape, latent_size, 
                 batch_size=4, width=1, base_size=16):
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.width = width
        self.base_size = base_size
        
    def build(self,):
        model_input = Input(shape=[self.latent_size]) #, batch_size=self.batch_size)
        
        model = Dense(8*8*4)(model_input)
        
        # reshape for convolving
        model = Reshape(target_shape=(8, 8, 4))(model)
        model = Conv2D(filters=3*self.base_size*self.width, 
                       kernel_size=(3, 3), strides=(1, 1),
                       padding='same', activation='elu')(model)
        
        # second block
        model = UpSampling2D(size=(2, 2))(model)
        model = Conv2D(filters=2*self.base_size*self.width, 
                            kernel_size=(3, 3), strides=(1, 1),
                            padding='same', activation='elu')(model)
        
        # third block
        model = UpSampling2D(size=(2, 2))(model)
        model = Conv2D(filters=1*self.base_size*self.width, 
                            kernel_size=(3, 3), strides=(1, 1),
                            padding='same', activation='elu')(model)
        
        model_output = Conv2D(filters=self.input_shape[-1], 
                              kernel_size=(1, 1), padding='same',
                              activation='sigmoid')(model)
        
        return Model(inputs=[model_input], outputs=[model_output])


if __name__ == '__main__':
    input_shape = (32, 32, 3)
    latent_size = 256
    batch_size = 1
    
    encoder = Encoder(input_shape, latent_size, batch_size)
    encoder_model = encoder.build()
    encoder_model.summary()
    
    decoder = Decoder(input_shape, latent_size, batch_size)
    decoder_model = decoder.build()
    decoder_model.summary()
    
    vae = Model(inputs=encoder_model.inputs, 
                outputs=decoder_model(encoder_model.outputs))
    
    from tensorflow.keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    Xt = x_train / 255
    Xv = x_test / 255
    
    #Xt = Xt[0:1024]
    
    print(x_train.shape)
    print(y_train.shape)
    
    vae.compile(optimizer='adam', loss='mae')
    
    img = Xv[0]

    test_img = Image.fromarray(np.uint8(img * 255))
    test_img.save('output/orig.png')
    
    for i in range(100):
        print('on epoch {}'.format(i+1))
        vae.fit(x=Xt, y=Xt, batch_size=batch_size, epochs=1)
        
        test_batch = np.array([img for _ in range(batch_size)])
        
        pred = vae.predict(test_batch)
        #pred[pred > 0.5] = 0.5
        #pred[pred < -0.5] = -0.5
        #pred = np.uint8((pred + 0.5)* 255)
        pred = np.uint8(pred * 255)
        
        pred = Image.fromarray(pred[0])
        pred.save('output/{}.png'.format(i))
        