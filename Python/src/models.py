"""
    Ce fichier comprend les méthodes de constructions des différents réseaux de neurones utilisés lors de ce stage.  Les réseaux disponibles sont tous listés dans le dictionnaire situé à la  toute fin de ce fichier. S'y référer lors de la construction d'un modèle, et ne pas oublier de le mettre à jour lors de l'ajout d'une nouvelle méthode constructrice !
"""

from __future__ import absolute_import,print_function,division

from keras.initializers import random_uniform, zeros
from keras.layers.core import Activation
from keras.activations import relu
from keras.layers import Input, Conv2D, Conv2DTranspose, Lambda, Concatenate, MaxPool2D, Dropout
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential, Model
import keras.backend as K
import tensorflow as tf

from src.Layers import Inception, InceptionPooling, InceptionConcat, UpscalingLayer, UpscalingBicubic




# <editor-fold desc="Définition des fonctions de construction de modèles">
def simple_model(input_shape):
    """
    Dummy function, just to test. Builds a reallys simple model, very fast but useless.
    :param 
        input_shape: a tuple of 3 ints, the shape of the input. 
    :return
        mod : a keras model of the network
    """
    ins = Input(shape=input_shape)
    a = Conv2D(filters=32,
               kernel_size=(3, 3),
               padding='same',
               use_bias=True,
               kernel_initializer=random_uniform(minval=-0.1, maxval=0.1),
               bias_initialzer=zeros(),
               __name__='Test'
               )(ins)
    a = Conv2D(filters=32,
               kernel_size=(3, 3),
               padding='same',
               use_bias=True,
               kernel_initializer=random_uniform(minval=-0.1, maxval=0.1),
               bias_initialzer=zeros()
               )(a)
    a = Conv2D(filters=32,
               kernel_size=(3, 3),
               padding='same',
               use_bias=True,
               kernel_initializer=random_uniform(minval=-0.1, maxval=0.1),
               bias_initialzer=zeros()
               )(a)
    mod = Model(inputs=ins,
                outputs=a,
                name='Network')
    return mod


def upscaled(input_shape, num_classes):
    """
    Builds a simple network using upscaled 2D convolutions.
    :param 
        input_shape: tuple of 3 integers, the shape of the input of the network
        num_classes: integer, number of classes in the output.
    :return
        mod: a keras model of the network.
            
    """
    mod = Sequential()

    mod.add(Conv2D(filters=16,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=1,
                   activation='linear',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters=32,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=1,
                   activation='linear',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters=64,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=2,
                   activation='linear',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters=64,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=2,
                   activation='linear',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters=128,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=4,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )
    mod.add(Conv2D(filters=128,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=4,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )
    mod.add(Conv2D(filters=128,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=8,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )
    mod.add(Conv2D(filters=128,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=8,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )
    mod.add(Conv2D(filters=256,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=16,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )
    mod.add(Conv2D(filters=num_classes,
                   kernel_size=(1, 1),
                   padding='same',
                   dilation_rate=4,
                   activation='softmax',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )

    return mod


def upscaled_truncated(input_shape, num_classes):
    """
    Builds a smaller network of upscaled convolutions, to fit on a gtx980ti, for testing purpose only.
    :param 
        input_shape: same as above. 
        num_classes: same as above
    :return
        mod: a keras model of the network.
    """
    mod = Sequential()

    mod.add(Conv2D(filters=16,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=1,
                   activation='linear',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape,
                   name='test'
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters=32,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=1,
                   activation='linear',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform()
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters=64,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=2,
                   activation='linear',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform()
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters=64,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=2,
                   activation='linear',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform()
                   )
            )
    mod.add(PReLU())

    mod.add(Conv2D(filters=num_classes,
                   kernel_size=(1, 1),
                   padding='same',
                   dilation_rate=4,
                   activation='softmax',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform()
                   )
            )

    return mod

def upscaled_without_aggreg(input_shape, num_classes):

    ins = Input(shape=input_shape,
                name='net_inputs')
    a = Conv2D(
            filters=16,
            kernel_size=(3,3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv0'
            )(ins)
    a = Dropout(rate=0.1)(a)
    a = Conv2D(
            filters=32,
            kernel_size=(3,3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name = 'net_conv1'
            )(a)
    a = Conv2D(
            filters=64,
            kernel_size=(3,3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name = 'net_conv2'
            )(a)
    a = Conv2D(
            filters=64,
            kernel_size=(3,3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name = 'net_conv3'
            )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv4'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv5'
    )(a)
    a = Dropout(rate=0.1)(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv6'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv7'
    )(a)
    a = Dropout(rate=0.1)(a)
    a = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=16,
            activation='relu',
            name='net_conv8'
    )(a)
    a = Conv2D(
            filters=num_classes,
            kernel_size=1,
            kernel_initializer=random_uniform(),
            use_bias=False,
            activation='softmax',
            padding='same',
            name = 'net_out'
    )(a)

    mod = Model(
            inputs = ins,
            outputs = a,
        )

    return mod

def upscaled_with_aggreg(input_shape, num_classes):
    ins = Input(shape=input_shape,
                name='net_inputs')
    a = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv0'
    )(ins)
    a = Dropout(rate=0.1)(a)
    a = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv1'
    )(a)
    a = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv2'
    )(a)
    a = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv3'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv4'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv5'
    )(a)
    a = Dropout(rate=0.1)(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv6'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv7'
    )(a)
    a = Dropout(rate=0.1)(a)
    a = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=16,
            activation='relu',
            name='net_conv8'
    )(a)
    b = Conv2D(
            filters=num_classes,
            kernel_size=1,
            kernel_initializer=random_uniform(),
            use_bias=False,
            activation='softmax',
            padding='same',
            name='net_out'
    )(a)


    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_1'
    )(b)
    b= Dropout(rate=0.1)(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=2,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_2'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=4,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_3'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=8,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_4'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=16,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_5'
    )(b)
    b = Dropout(rate=0.1)(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_6'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=1,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='softmax',
            padding='same',
            name='aggreg_7'
    )(b)

    mod = Model(
        inputs=ins,
        outputs=b
    )
    return mod

def upscaled_with_deeper_aggreg(input_shape, num_classes):
    ins = Input(shape=input_shape,
                name='net_inputs')
    a = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv0'
    )(ins)
    a = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv1'
    )(a)
    a = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv2'
    )(a)
    a = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv3'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv4'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv5'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv6'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv7'
    )(a)
    a = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=16,
            activation='relu',
            name='net_conv8'
    )(a)
    a = Conv2D(
            filters=num_classes,
            kernel_size=1,
            kernel_initializer=random_uniform(),
            use_bias=False,
            activation='softmax',
            padding='same',
            name='net_out'
    )(a)

    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_0'
    )(a)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_1'
    )(b)
    b = Conv2D(
            filters=2*num_classes,
            kernel_size=3,
            dilation_rate=2,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_2'
    )(b)
    b = Conv2D(
            filters=4*num_classes,
            kernel_size=3,
            dilation_rate=4,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_3'
    )(b)
    b = Conv2D(
            filters=8*num_classes,
            kernel_size=3,
            dilation_rate=8,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_4'
    )(b)
    b = Conv2D(
            filters=8*num_classes,
            kernel_size=3,
            dilation_rate=16,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_5'
    )(b)
    b = Conv2D(
            filters=8*num_classes,
            kernel_size=3,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_6'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=1,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='softmax',
            padding='same',
            name='aggreg_7'
    )(b)

    mod = Model(
        inputs=ins,
        outputs=b
    )
    return mod

def test_inception(input_shape,num_classes):
    mod = Sequential()

    mod.add(Inception(input_shape=input_shape,
                      output_depth=num_classes)
            )
    return mod

def inception_with_deeper_aggreg(input_shape, num_classes):
    inputs = Input(input_shape,
                   name='net_ins')
    a = Inception(input_shape=input_shape,
                  output_depth=16,
                  dilation_rate=(1, 1),
                  name='net_incept_1'
                  )(inputs)
    current_shape = input_shape[:-1] + (16,)
    a = Inception(input_shape=current_shape,
                  output_depth=32,
                  dilation_rate=(2,2),
                  name='net_incept_2'
                  )(a)
    current_shape=current_shape[:-1]+(32,)
    a = Inception(input_shape=current_shape,
                  output_depth=64,
                  dilation_rate=(4, 4),
                  name='net_incept_3'
                  )(a)
    current_shape=current_shape[:-1]+(64,)
    a = Inception(input_shape=current_shape,
                  output_depth=128,
                  dilation_rate=(8,8),
                  name='net_incept_4'
                  )(a)
    current_shape=current_shape[:-1]+(64,)
    a = Inception(input_shape=current_shape,
                  output_depth=128,
                  dilation_rate=(16,16),
                  name='net_incept_5'
                  )(a)
    current_shape=current_shape[:-1]+(128,)
    a = Inception(input_shape=current_shape,
                  output_depth=256,
                  dilation_rate=(32, 32),
                  name='net_incept_6'
                  )(a)
    current_shape=current_shape[:-1]+(256,)
    a = Inception(input_shape=current_shape,
                  output_depth=num_classes,
                  dilation_rate=(32, 32),
                  name='net_incept_7'
                  )(a)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_0'
    )(a)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_1'
    )(b)
    b = Conv2D(
            filters=2 * num_classes,
            kernel_size=3,
            dilation_rate=2,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_2'
    )(b)
    b = Conv2D(
            filters=4 * num_classes,
            kernel_size=3,
            dilation_rate=4,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_3'
    )(b)
    b = Conv2D(
            filters=8 * num_classes,
            kernel_size=3,
            dilation_rate=8,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_4'
    )(b)
    b = Conv2D(
            filters=8 * num_classes,
            kernel_size=3,
            dilation_rate=16,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_5'
    )(b)
    b = Conv2D(
            filters=8 * num_classes,
            kernel_size=3,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_6'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=1,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='softmax',
            padding='same',
            name='aggreg_7'
    )(b)
    mod = Model(
            inputs=inputs,
            outputs=b
    )
    return mod

def inception_with_aggreg(input_shape, num_classes):
    input_shape=tuple(input_shape)
    inputs = Input(input_shape,
                   name='net_ins')
    a = Inception(input_shape=input_shape,
                  output_depth=16,
                  dilation_rate=(1, 1),
                  name='net_incept_1'
                  )(inputs)
    current_shape = input_shape[:-1] + (16,)
    a = Inception(input_shape=current_shape,
                  output_depth=32,
                  dilation_rate=(2,2),
                  name='net_incept_2'
                  )(a)
    current_shape=current_shape[:-1]+(32,)
    a = Inception(input_shape=current_shape,
                  output_depth=64,
                  dilation_rate=(4, 4),
                  name='net_incept_3'
                  )(a)
    current_shape=current_shape[:-1]+(64,)
    a = Inception(input_shape=current_shape,
                  output_depth=128,
                  dilation_rate=(8,8),
                  name='net_incept_4'
                  )(a)
    current_shape=current_shape[:-1]+(64,)
    a = Inception(input_shape=current_shape,
                  output_depth=128,
                  dilation_rate=(8,8),
                  name='net_incept_5'
                  )(a)
    current_shape=current_shape[:-1]+(128,)
    a = Inception(input_shape=current_shape,
                  output_depth=128,
                  dilation_rate=(8, 8),
                  name='net_incept_6'
                  )(a)
    current_shape=current_shape[:-1]+(128,)
    a = Inception(input_shape=current_shape,
                  output_depth=num_classes,
                  dilation_rate=(32, 8),
                  name='net_incept_7',
                  softmax=True,
                  )(a)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_0'
    )(a)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_1'
    )(b)
    b = Conv2D(
            filters= num_classes,
            kernel_size=3,
            dilation_rate=2,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_2'
    )(b)
    b = Conv2D(
            filters= num_classes,
            kernel_size=3,
            dilation_rate=4,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_3'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=1,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='softmax',
            padding='same',
            name='aggreg_7'
    )(b)
    mod = Model(
            inputs=inputs,
            outputs=b
    )
    return mod

def inception_pure(input_shape, num_classes):
    inputs = Input(input_shape,
                   name='net_ins')
    a = Inception(input_shape=input_shape,
                  output_depth=16,
                  dilation_rate=(1, 1),
                  name='net_incept_1'
                  )(inputs)
    current_shape = tuple(input_shape[:-1]) + (16,)
    a = Inception(input_shape=current_shape,
                  output_depth=32,
                  dilation_rate=(2,2),
                  name='net_incept_2'
                  )(a)
    current_shape=current_shape[:-1]+(32,)
    a = Inception(input_shape=current_shape,
                  output_depth=64,
                  dilation_rate=(4, 4),
                  name='net_incept_3'
                  )(a)
    current_shape=current_shape[:-1]+(64,)
    a = Inception(input_shape=current_shape,
                  output_depth=128,
                  dilation_rate=(8,8),
                  name='net_incept_4'
                  )(a)
    current_shape=current_shape[:-1]+(64,)
    a = InceptionConcat(input_shape=current_shape,
                  output_depth=192,
                        mid_depth=96,
                  dilation_rate=(16,16),
                  name='net_incept_5'
                  )(a)
    current_shape=current_shape[:-1]+(192,)
    a = InceptionConcat(input_shape=current_shape,
                  output_depth=128,
                    mid_depth=64,
                  dilation_rate=(16, 16),
                  name='net_incept_6'
                  )(a)
    current_shape=current_shape[:-1]+(128,)
    a = Inception(input_shape=current_shape,
                  output_depth=num_classes,
                  dilation_rate=(16, 16),
                  softmax=True,
                  name='net_incept_7'
                  )(a)

    mod = Model(
            inputs=inputs,
            outputs=a
    )
    return mod

def unpooling_4times(input_shape, num_classes):
    """
    Just a dummy network to learn how to convolutionnaly upscale an input.
    Args:
        input_shape ():
        num_classes ():

    Returns:

    """
    input_shape = tuple(input_shape)
    inputs = Input(input_shape,
                   name = 'input_layer')
    a = Conv2DTranspose(filters=num_classes,
                        kernel_size=(3,3),
                        strides=(2,2),
                        padding='same',
                        use_bias=False,
                        name='Upscale_1'
                        )(inputs)
    a = Conv2DTranspose(filters=num_classes,
                        kernel_size=(3,3),
                        strides=(2,2),
                        padding='same',
                        use_bias=False,
                        name='Upscale_2'
                        )(a)
    mod = Model(inputs=inputs,
                outputs=a)

    return mod

def inception_with_pooling(input_shape, num_classes):
    input_shape = tuple(input_shape)
    current_shape = input_shape
    inputs = Input(shape=input_shape)

    a = InceptionPooling(input_shape=current_shape,
                         output_depth=32,
                         mid_depth=8,
                         name='Inception_pool_1'
                         )(inputs)
    current_shape = current_shape[:-3] + (current_shape[-3]//2,) + (current_shape[-2]//2,) + (32,)
    a = InceptionPooling(input_shape=current_shape,
                         output_depth=128,
                         mid_depth=32,
                         name='Inception_pool_2'
                         )(a)
    current_shape = current_shape[:-3] + (current_shape[-3]//2,) + (current_shape[-2]//2,) + (128,)
    a = InceptionConcat(input_shape=current_shape,
                        output_depth=256,
                        mid_depth=64,
                        name='Inception_upscale_1'
                        )(a)
    current_shape = current_shape[:-1] + (256, )
    a = InceptionConcat(input_shape=current_shape,
                        output_depth=256,
                        mid_depth=92,
                        name='Inception_upscale_2'
                        )(a)
    a = InceptionConcat(input_shape=current_shape,
                        output_depth=256,
                        mid_depth=92,
                        name='Inception_upscale_3'
                        )(a)
    a = Inception(input_shape=current_shape,
                  output_depth=num_classes,
                  name='Inception_Last'
                  )(a)
    a = UpscalingBicubic(name='Upscale_1')(a)
    a = UpscalingBicubic(name='Upscale_2')(a)
    # a = UpscalingLayer(name='Unpooling_1')(a)
    a = Activation('softmax')(a)
    mod = Model(inputs=inputs,
                outputs=a
                )

    return mod

def test_inception_with_pooling(input_shape, num_classes):
    input_shape = tuple(input_shape)
    current_shape = input_shape
    inputs = Input(shape=input_shape)

    a = InceptionPooling(input_shape=current_shape,
                         output_depth=32,
                         mid_depth=8,
                         name='Inception_pool_1'
                         )(inputs)
    current_shape = current_shape[:-3] + (current_shape[-3]//2,) + (current_shape[-2]//2,) + (32,)
    a = InceptionPooling(input_shape=current_shape,
                         output_depth=num_classes,
                         mid_depth=64,
                         name='Inception_pool_2'
                         )(a)
    a = Activation('softmax')(a)
    current_shape = current_shape[:-3] + (current_shape[-3]//2,) + (current_shape[-2]//2,) +  (128,)

    a = UpscalingBicubic(name='Unpooling_0')(a)
    a = UpscalingBicubic(name='Unpooling_1')(a)
    mod = Model(inputs=inputs,
                outputs=a
                )

    return mod

def upscaled_with_skips(input_shape, num_classes):

    ins = Input(shape=input_shape,
                name='net_inputs')
    a = Conv2D(
            filters=16,
            kernel_size=(3,3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv0'
            )(ins)
    a = Conv2D(
            filters=32,
            kernel_size=(3,3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name = 'net_conv1'
            )(a)
    a = Dropout(rate=0.1)(a)
    b = Conv2D(
            filters=64,
            kernel_size=(3,3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name = 'net_conv2'
            )(a)
    c = Conv2D(
            filters=64,
            kernel_size=(3,3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name = 'net_conv3'
            )(b)
    d = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv4'
    )(c)
    d = Dropout(rate=0.1)(d)
    e = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv5'
    )(d)
    f = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv6'
    )(e)
    g = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv7'
    )(f)
    g = Dropout(rate=0.1)(g)
    h = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=16,
            activation='relu',
            name='net_conv8'
    )(g)
    i = Concatenate(name='Fusion')([b,d,f,h])
    i = Conv2D(
            filters=num_classes,
            kernel_size=1,
            kernel_initializer=random_uniform(),
            use_bias=False,
            activation='softmax',
            padding='same',
            name = 'net_out'
    )(i)

    mod = Model(
            inputs = ins,
            outputs = i,
        )

    return mod

def upscaled_with_skips_aggreg(input_shape, num_classes):
    ins = Input(shape=input_shape,
                name='net_inputs')
    a = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv0'
    )(ins)
    a = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv1'
    )(a)
    a = Dropout(rate=0.1)(a)
    b = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv2'
    )(a)
    c = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv3'
    )(b)
    d = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv4'
    )(c)
    d = Dropout(rate=0.1)(d)
    e = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv5'
    )(d)
    f = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv6'
    )(e)
    g = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv7'
    )(f)
    g = Dropout(rate=0.1)(g)
    h = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=16,
            activation='relu',
            name='net_conv8'
    )(g)
    i = Concatenate(name='Fusion')([b, d, f, h])
    i = Conv2D(
            filters=num_classes,
            kernel_size=1,
            kernel_initializer=random_uniform(),
            use_bias=False,
            activation='softmax',
            padding='same',
            name='net_out'
    )(i)

    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_1'
    )(i)
    b = Dropout(rate=0.1)(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=2,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_2'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=4,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_3'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=8,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_4'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=16,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_5'
    )(b)
    b = Dropout(rate=0.1)(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_6'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=1,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='softmax',
            padding='same',
            name='aggreg_7'
    )(b)

    mod = Model(
            inputs = ins,
            outputs = b,
        )

    return mod


def upscaled_with_skips_and_meta(input_shape, num_classes):
    # <editor-fold desc="Gestion des inputs">
    ins = Input(shape=input_shape,
                name='net_inputs')
    ins_rgb = Lambda(lambda x: x[:,:,:,0:3], name='rgb_select')(ins)
    ins_meta = Lambda(lambda x: x[:,:,:,3:], name='meta_select')(ins)
    # </editor-fold>
    # <editor-fold desc="Traitement dde la couche rgb">
    a = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv0'
    )(ins_rgb)
    a = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv1'
    )(a)
    b = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv2'
    )(a)
    c = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv3'
    )(b)
    d = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv4'
    )(c)
    e = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv5'
    )(d)
    f = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv6'
    )(e)
    g = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv7'
    )(f)
    h = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=16,
            activation='relu',
            name='net_conv8'
    )(g)
    # </editor-fold>

    # <editor-fold desc="Traitement des meta-données">
    meta = Conv2D(
            filters=16,
            kernel_size=3,
            padding='same',
            dilation_rate=(1,1),
            activation='relu',
            use_bias=True,
            name='net_conv_meta_0'
    )(ins_meta)
    meta2 = Conv2D(
            filters=16,
            kernel_size=5,
            padding='same',
            dilation_rate=(2,2),
            activation='relu',
            use_bias=True,
            name='net_conv_meta_1'
    )(meta)
    meta3 = Conv2D(
            filters=16,
            kernel_size=5,
            padding='same',
            dilation_rate=(4,4),
            activation='relu',
            use_bias=True,
            name='net_conv_meta_2'
    )(meta2)
    # </editor-fold>

    i = Concatenate(name='net_Fusion')([b, d, f, h, meta2])
    i = Conv2D(
            filters=num_classes,
            kernel_size=1,
            kernel_initializer=random_uniform(),
            use_bias=False,
            activation='softmax',
            padding='same',
            name='net_out'
    )(i)

    mod = Model(
            inputs=ins,
            outputs=i,
    )

    return mod

def upscaled_with_skips_and_meta_aggreg(input_shape, num_classes):

    # <editor-fold desc="Gestion des inputs">
    ins = Input(shape=input_shape,
                name='net_inputs')
    ins_rgb = Lambda(lambda x: x[:,:,:,0:3], name='rgb_select')(ins)
    ins_meta = Lambda(lambda x: x[:,:,:,3:], name='meta_select')(ins)
    # </editor-fold>
    # <editor-fold desc="Traitement dde la couche rgb">
    a = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv0'
    )(ins_rgb)
    a = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv1'
    )(a)
    b = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv2'
    )(a)
    c = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv3'
    )(b)
    d = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv4'
    )(c)
    e = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv5'
    )(d)
    f = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv6'
    )(e)
    g = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv7'
    )(f)
    h = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=16,
            activation='relu',
            name='net_conv8'
    )(g)
    # </editor-fold>

    # <editor-fold desc="Traitement des meta-données">
    meta = Conv2D(
            filters=16,
            kernel_size=3,
            padding='same',
            dilation_rate=(1,1),
            activation='relu',
            use_bias=True,
            name='net_conv_meta_0'
    )(ins_meta)
    meta2 = Conv2D(
            filters=16,
            kernel_size=5,
            padding='same',
            dilation_rate=(2,2),
            activation='relu',
            use_bias=True,
            name='net_conv_meta_1'
    )(meta)
    meta3 = Conv2D(
            filters=16,
            kernel_size=5,
            padding='same',
            dilation_rate=(4,4),
            activation='relu',
            use_bias=True,
            name='net_conv_meta_2'
    )(meta2)
    # </editor-fold>

    # <editor-fold desc="Fusion">
    i = Concatenate(name='net_Fusion')([b, d, f, h, meta2])
    i = Conv2D(
            filters=num_classes,
            kernel_size=1,
            kernel_initializer=random_uniform(),
            use_bias=False,
            activation='softmax',
            padding='same',
            name='net_out'
    )(i)
    # </editor-fold>

    # <editor-fold desc="Aggregation">
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_1'
    )(i)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=2,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_2'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=4,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_3'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=8,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_4'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=16,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_5'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_6'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=1,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='softmax',
            padding='same',
            name='aggreg_7'
    )(b)
    # </editor-fold>

    mod = Model(
            inputs=ins,
            outputs=b,
    )

    return mod

def upscaled_with_skips_and_meta_pool(input_shape, num_classes):
    # <editor-fold desc="Gestion des inputs">
    ins = Input(shape=input_shape,
                name='net_inputs')
    ins_rgb = Lambda(lambda x: x[:,:,:,0:3], name='rgb_select')(ins)
    ins_meta = Lambda(lambda x: x[:,:,:,3:], name='meta_select')(ins)
    # </editor-fold>
    # <editor-fold desc="Traitement dde la couche rgb">
    a = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv0'
    )(ins_rgb)
    a = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv1'
    )(a)
    b = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv2'
    )(a)
    c = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv3'
    )(b)
    d = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv4'
    )(c)
    e = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv5'
    )(d)
    f = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv6'
    )(e)
    g = MaxPool2D(
            padding='same',
            name='net_pool'
    )(f)
    g = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv7'
    )(g)
    h = Conv2D(
            filters=512,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv8'
    )(g)
    h = Conv2D(
            filters=128,
            kernel_size=(1,1),
            padding='same'
    )(h)
    h = Lambda(
            lambda x:tf.image.resize_bilinear(x,size=[input_shape[0],input_shape[1]]),
            name='net_upscale'
    )(h)
    # </editor-fold>

    # <editor-fold desc="Traitement des meta-données">
    meta = Conv2D(
            filters=16,
            kernel_size=3,
            padding='same',
            dilation_rate=(1,1),
            activation='relu',
            use_bias=True,
            name='net_conv_meta_0'
    )(ins_meta)
    meta2 = Conv2D(
            filters=16,
            kernel_size=5,
            padding='same',
            dilation_rate=(2,2),
            activation='relu',
            use_bias=True,
            name='net_conv_meta_1'
    )(meta)
    meta3 = Conv2D(
            filters=16,
            kernel_size=5,
            padding='same',
            dilation_rate=(4,4),
            activation='relu',
            use_bias=True,
            name='net_conv_meta_2'
    )(meta2)
    # </editor-fold>

    i = Concatenate(name='net_Fusion')([b, d, f, h, meta2])
    i = Conv2D(
            filters=num_classes,
            kernel_size=1,
            kernel_initializer=random_uniform(),
            use_bias=False,
            activation='softmax',
            padding='same',
            name='net_out'
    )(i)

    mod = Model(
            inputs=ins,
            outputs=i,
    )

    return mod

def upscaled_with_skips_and_meta__pool_aggreg(input_shape, num_classes):

    # <editor-fold desc="Gestion des inputs">
    ins = Input(shape=input_shape,
                name='net_inputs')
    ins_rgb = Lambda(lambda x: x[:,:,:,0:3], name='rgb_select')(ins)
    ins_meta = Lambda(lambda x: x[:,:,:,3:], name='meta_select')(ins)
    # </editor-fold>
    # <editor-fold desc="Traitement dde la couche rgb">
    a = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv0'
    )(ins_rgb)
    a = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv1'
    )(a)
    b = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv2'
    )(a)
    c = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv3'
    )(b)
    d = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv4'
    )(c)
    e = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv5'
    )(d)
    f = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv6'
    )(e)
    g = MaxPool2D(
            padding='same',
            name='net_pool'
    )(f)
    g = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv7'
    )(g)
    h = Conv2D(
            filters=512,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv8'
    )(g)
    h = Conv2D(
            filters=128,
            kernel_size=(1, 1),
            padding='same'
    )(h)
    h = Lambda(
            lambda x: tf.image.resize_bilinear(x, size=[input_shape[0], input_shape[1]]),
            name='net_upscale'
    )(h)
    # </editor-fold>

    # <editor-fold desc="Traitement des meta-données">
    meta = Conv2D(
            filters=16,
            kernel_size=3,
            padding='same',
            dilation_rate=(1,1),
            activation='relu',
            use_bias=True,
            name='net_conv_meta_0'
    )(ins_meta)
    meta2 = Conv2D(
            filters=16,
            kernel_size=5,
            padding='same',
            dilation_rate=(2,2),
            activation='relu',
            use_bias=True,
            name='net_conv_meta_1'
    )(meta)
    meta3 = Conv2D(
            filters=16,
            kernel_size=5,
            padding='same',
            dilation_rate=(4,4),
            activation='relu',
            use_bias=True,
            name='net_conv_meta_2'
    )(meta2)
    # </editor-fold>

    # <editor-fold desc="Fusion">
    i = Concatenate(name='net_Fusion')([b, d, f, h, meta2])
    i = Conv2D(
            filters=num_classes,
            kernel_size=1,
            kernel_initializer=random_uniform(),
            use_bias=False,
            activation='softmax',
            padding='same',
            name='net_out'
    )(i)
    # </editor-fold>

    # <editor-fold desc="Aggregation">
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_1'
    )(i)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=2,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_2'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=4,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_3'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=8,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_4'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=16,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_5'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_6'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=1,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='softmax',
            padding='same',
            name='aggreg_7'
    )(b)
    # </editor-fold>

    mod = Model(
            inputs=ins,
            outputs=b,
    )

    return mod

def upscaled_with_skips_and_meta_pool_dropout(input_shape, num_classes):
    # <editor-fold desc="Gestion des inputs">
    ins = Input(shape=input_shape,
                name='net_inputs')
    ins_rgb = Lambda(lambda x: x[:,:,:,0:3], name='rgb_select')(ins)
    ins_meta = Lambda(lambda x: x[:,:,:,3:], name='meta_select')(ins)
    # </editor-fold>
    # <editor-fold desc="Traitement dde la couche rgb">
    a = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv0'
    )(ins_rgb)
    a = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv1'
    )(a)
    b = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv2'
    )(a)
    b = Dropout(rate=0.2)(b)
    c = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv3'
    )(b)
    d = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv4'
    )(c)
    d = Dropout(rate=0.2)(d)
    e = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv5'
    )(d)
    f = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv6'
    )(e)
    f = Dropout(rate = 0.2)(f)
    g = MaxPool2D(
            padding='same',
            name='net_pool'
    )(f)
    g = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv7'
    )(g)
    h = Conv2D(
            filters=512,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv8'
    )(g)
    h = Conv2D(
            filters=128,
            kernel_size=(1,1),
            padding='same'
    )(h)
    h = Lambda(
            lambda x:tf.image.resize_bilinear(x,size=[input_shape[0],input_shape[1]]),
            name='net_upscale'
    )(h)
    # </editor-fold>

    # <editor-fold desc="Traitement des meta-données">
    meta = Conv2D(
            filters=16,
            kernel_size=3,
            padding='same',
            dilation_rate=(1,1),
            activation='relu',
            use_bias=True,
            name='net_conv_meta_0'
    )(ins_meta)
    meta2 = Conv2D(
            filters=16,
            kernel_size=5,
            padding='same',
            dilation_rate=(2,2),
            activation='relu',
            use_bias=True,
            name='net_conv_meta_1'
    )(meta)
    meta3 = Conv2D(
            filters=32,
            kernel_size=5,
            padding='same',
            dilation_rate=(4,4),
            activation='relu',
            use_bias=True,
            name='net_conv_meta_2'
    )(meta2)
    # </editor-fold>

    i = Concatenate(name='net_Fusion')([b, d, f, h, meta3])
    i = Conv2D(
            filters=num_classes,
            kernel_size=1,
            kernel_initializer=random_uniform(),
            use_bias=False,
            activation='softmax',
            padding='same',
            name='net_out'
    )(i)

    mod = Model(
            inputs=ins,
            outputs=i,
    )

    return mod

def upscaled_with_skips_and_meta__pool_dropout_aggreg(input_shape, num_classes):

    # <editor-fold desc="Gestion des inputs">
    ins = Input(shape=input_shape,
                name='net_inputs')
    ins_rgb = Lambda(lambda x: x[:,:,:,0:3], name='rgb_select')(ins)
    ins_meta = Lambda(lambda x: x[:,:,:,3:], name='meta_select')(ins)
    # </editor-fold>
    # <editor-fold desc="Traitement dde la couche rgb">
    a = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv0'
    )(ins_rgb)
    a = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv1'
    )(a)
    b = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv2'
    )(a)
    b = Dropout(rate=0.2)(b)
    c = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv3'
    )(b)
    d = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv4'
    )(c)
    d = Dropout(rate=0.2)(d)
    e = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv5'
    )(d)
    f = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv6'
    )(e)
    f = Dropout(rate = 0.2)(f)
    g = MaxPool2D(
            padding='same',
            name='net_pool'
    )(f)
    g = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv7'
    )(g)
    h = Conv2D(
            filters=512,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv8'
    )(g)
    h = Conv2D(
            filters=128,
            kernel_size=(1, 1),
            padding='same'
    )(h)
    h = Lambda(
            lambda x: tf.image.resize_bilinear(x, size=[input_shape[0], input_shape[1]]),
            name='net_upscale'
    )(h)
    # </editor-fold>

    # <editor-fold desc="Traitement des meta-données">
    meta = Conv2D(
            filters=16,
            kernel_size=3,
            padding='same',
            dilation_rate=(1,1),
            activation='relu',
            use_bias=True,
            name='net_conv_meta_0'
    )(ins_meta)
    meta2 = Conv2D(
            filters=16,
            kernel_size=5,
            padding='same',
            dilation_rate=(2,2),
            activation='relu',
            use_bias=True,
            name='net_conv_meta_1'
    )(meta)
    meta3 = Conv2D(
            filters=32,
            kernel_size=5,
            padding='same',
            dilation_rate=(4,4),
            activation='relu',
            use_bias=True,
            name='net_conv_meta_2'
    )(meta2)
    # </editor-fold>

    # <editor-fold desc="Fusion">
    i = Concatenate(name='net_Fusion')([b, d, f, h, meta3])
    i = Conv2D(
            filters=num_classes,
            kernel_size=1,
            kernel_initializer=random_uniform(),
            use_bias=False,
            activation='softmax',
            padding='same',
            name='net_out'
    )(i)
    # </editor-fold>

    # <editor-fold desc="Aggregation">
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_1'
    )(i)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=2,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_2'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=4,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_3'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=8,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_4'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=16,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_5'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_6'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=1,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='softmax',
            padding='same',
            name='aggreg_7'
    )(b)
    # </editor-fold>

    mod = Model(
            inputs=ins,
            outputs=b,
    )

    return mod
# </editor-fold>

# A dictionnary linking model builder names to the actual functions.

models_dict = {
    'simple_model': simple_model,
    'up': upscaled,
    'up_mini': upscaled_truncated,
    # A tester : upscale simple, sans skips ni meta.
    'up_without': upscaled_without_aggreg,
    'up_with': upscaled_with_aggreg,

    # <editor-fold desc="Obsolete : tests de modeles ratés">
    'up_with_deeper_aggreg': upscaled_with_deeper_aggreg,
    'test_inception': test_inception,
    'inception_with': inception_with_deeper_aggreg,
    'inception_with_lighter': inception_with_aggreg,
    'inception_pure': inception_pure,
    'unpooling': unpooling_4times,
    'test_inception_pooling':test_inception_with_pooling,
    'inception_pooling': inception_with_pooling,
    'upscale_skips_meta': upscaled_with_skips_and_meta,
    # </editor-fold>
    # Utiliser ces deux la : avec skips et sans meta
    'upscale_skips': upscaled_with_skips,
    'up_skips_aggreg':upscaled_with_skips_aggreg,
    # <editor-fold desc="Obsolètes : pas de dropout">
    # Les deux suivantes sont obsoloetes : pas de dropout
    'up_skips_meta_pool': upscaled_with_skips_and_meta__pool_aggreg,
    'up_skips_meta_pool_aggreg': upscaled_with_skips_and_meta__pool_aggreg,
    # </editor-fold>
    # Utiliser les deux suivantes
    'up_skips_meta_pool_drop': upscaled_with_skips_and_meta_pool_dropout,
    'up_skips_meta_pool_drop_aggreg': upscaled_with_skips_and_meta__pool_dropout_aggreg
}
