from __future__ import absolute_import, print_function, division

import keras.backend as K
from keras.layers import Layer, UpSampling2D, MaxPool2D
import tensorflow as tf

# TODO : voir implémentation du bias pour le layer...
class Inception(Layer):
    """
    Defines a inception layer. Weights are not optimized due to the hight memory cost of the architecture.
    :properties : 
    """
    def __init__(self, output_depth, **kwargs):
        self.output_depth = output_depth
        self.use_bias = False
        self.dilation_rate = (1,1)
        self.use_softmax = False
        if 'use_bias' in kwargs.keys():
            self.use_bias=kwargs.pop('use_bias')
        if 'dilation_rate' in kwargs.keys():
            self.dilation_rate = kwargs.pop('dilation_rate')
        if 'softmax' in kwargs.keys():
            self.use_softmax = kwargs.pop('softmax')
        super(Inception,self).__init__(**kwargs)

    def build(self, input_shape):
        self.K1 = self.add_weight(
                shape=[3, 3, input_shape[-1], self.output_depth],
                initializer='glorot_normal',
                trainable=True,
                name = self.name + '_K1'
        )
        self.K2 = self.add_weight(
                shape=[3, 3, self.output_depth, self.output_depth],
                initializer='glorot_normal',
                trainable=True,
                name=self.name+'_K2'
        )
        self.K3=self.add_weight(
                shape=[3,3,input_shape[-1], self.output_depth],
                initializer='glorot_normal',
                trainable=True,
                name=self.name+'_K3'
        )
        self.K4=self.add_weight(
                shape=[1, 1, input_shape[-1], self.output_depth],
                trainable=True,
                initializer='glorot_normal',
                name=self.name+'_K4'
        )
        super(Inception,self).build(input_shape)

    def call(self, inputs, **kwargs):
        tower1 = K.conv2d(x=inputs,
                          kernel=self.K1,
                          strides=(1, 1),
                          padding='same',
                          dilation_rate=self.dilation_rate
                          )
        tower1 = K.conv2d(x=tower1,
                          kernel=self.K2,
                          strides=(1, 1),
                          padding='same',
                          dilation_rate=self.dilation_rate
                          )
        tower2 = K.conv2d(x=inputs,
                          kernel=self.K3,
                          strides=(1, 1),
                          padding='same',
                          dilation_rate=self.dilation_rate
                          )
        tower3 = K.conv2d(x=inputs,
                          kernel=self.K4,
                          strides=(1,1),
                          padding='same',
                          dilation_rate=(1, 1)
                          )
        output = tower3+tower2+tower1
        if (self.use_softmax):
            return K.softmax(output)
        else:
            return K.relu(output)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_depth,)

class InceptionConcat(Layer):
    """
    Defines a inception layer. Weights are not optimized due to the hight memory cost of the architecture.
    :properties : 
    """
    def __init__(self, output_depth, **kwargs):
        self.output_depth = output_depth
        self.mid_depth = self.output_depth // 4
        self.use_bias = False
        self.use_softmax = False
        self.dilation_rate = (1,1)
        if 'use_bias' in kwargs.keys():
            self.use_bias=kwargs.pop('use_bias')
        if 'dilation_rate' in kwargs.keys():
            self.dilation_rate = kwargs.pop('dilation_rate')
        if 'mid_depth' in kwargs.keys():
            self.mid_depth = kwargs.pop('mid_depth')
        if 'softmax' in kwargs.keys():
            self.use_softmax = kwargs.pop('softmax')
        super(InceptionConcat,self).__init__(**kwargs)

    def build(self, input_shape):
        self.K1 = self.add_weight(
                shape=[3, 3, input_shape[-1], self.mid_depth],
                initializer='glorot_normal',
                trainable=True,
                name = self.name + '_K1'
        )
        self.K2 = self.add_weight(
                shape=[3, 3, self.mid_depth, self.mid_depth],
                initializer='glorot_normal',
                trainable=True,
                name=self.name+'_K2'
        )
        self.K3=self.add_weight(
                shape=[3,3,input_shape[-1], self.mid_depth],
                initializer='glorot_normal',
                trainable=True,
                name=self.name+'_K3'
        )
        self.K4=self.add_weight(
                shape=[1, 1, input_shape[-1], self.mid_depth],
                trainable=True,
                initializer='glorot_normal',
                name=self.name+'_K4'
        )
        self.K5 = self.add_weight(
                shape=[1,1,self.mid_depth*3,self.output_depth],
                trainable=True,
                initializer='glorot_normal',
                name=self.name+'_K5'
        )
        super(InceptionConcat,self).build(input_shape)

    def call(self, inputs, **kwargs):
        tower1 = K.conv2d(x=inputs,
                          kernel=self.K1,
                          strides=(1, 1),
                          padding='same',
                          dilation_rate=self.dilation_rate
                          )
        tower1 = K.conv2d(x=tower1,
                          kernel=self.K2,
                          strides=(1, 1),
                          padding='same',
                          dilation_rate=self.dilation_rate
                          )
        tower2 = K.conv2d(x=inputs,
                          kernel=self.K3,
                          strides=(1, 1),
                          padding='same',
                          dilation_rate=self.dilation_rate
                          )
        tower3 = MaxPool2D(
                pool_size=(2*self.dilation_rate[0],2*self.dilation_rate[1]),
                padding='same',
                           strides=(1,1),
                           )(inputs)
        tower3 = K.conv2d(x=tower3,
                          kernel=self.K4,
                          strides=(1,1),
                          padding='same',
                          dilation_rate=(1, 1)
                          )
        output = K.concatenate((tower1,tower2,tower3))
        output = K.conv2d(x=output,
                          kernel=self.K5,
                          strides=(1,1),
                          padding='same',
                          dilation_rate=(1,1)
                          )
        if self.use_softmax:
            return K.softmax(output)
        else:
            return K.relu(output)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_depth,)

class InceptionPooling(Layer):
    """
    Inception layer with pooling and concatenation at the end.
    """
    def __init__(self, output_depth, **kwargs):
        self.output_depth = output_depth
        self.mid_depth = self.output_depth // 4
        self.use_bias = False
        self.use_softmax = False
        self.dilation_rate = (1, 1)
        if 'use_bias' in kwargs.keys():
            self.use_bias = kwargs.pop('use_bias')
        if 'dilation_rate' in kwargs.keys():
            self.dilation_rate = kwargs.pop('dilation_rate')
        if 'mid_depth' in kwargs.keys():
            self.mid_depth = kwargs.pop('mid_depth')
        if 'softmax' in kwargs.keys():
            self.use_softmax = kwargs.pop('softmax')
        super(InceptionPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.K1 = self.add_weight(
                shape=[3, 3, input_shape[-1], self.mid_depth],
                initializer='glorot_normal',
                trainable=True,
                name=self.name + '_K1'
        )
        self.K2 = self.add_weight(
                shape=[3, 3, self.mid_depth, self.mid_depth],
                initializer='glorot_normal',
                trainable=True,
                name=self.name + '_K2'
        )
        self.K3 = self.add_weight(
                shape=[3, 3, input_shape[-1], self.mid_depth],
                initializer='glorot_normal',
                trainable=True,
                name=self.name + '_K3'
        )
        self.K4 = self.add_weight(
                shape=[1, 1, input_shape[-1], self.mid_depth],
                trainable=True,
                initializer='glorot_normal',
                name=self.name + '_K4'
        )
        self.K5 = self.add_weight(
                shape=[1, 1, self.mid_depth * 3, self.output_depth],
                trainable=True,
                initializer='glorot_normal',
                name=self.name + '_K5'
        )
        super(InceptionPooling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        tower1 = K.conv2d(x=inputs,
                          kernel=self.K1,
                          strides=(1, 1),
                          padding='same',
                          dilation_rate=self.dilation_rate
                          )
        tower1 = K.conv2d(x=tower1,
                          kernel=self.K2,
                          strides=(2, 2),
                          padding='same',
                          dilation_rate=(1,1)
                          )
        tower2 = K.conv2d(x=inputs,
                          kernel=self.K3,
                          strides=(1, 1),
                          padding='same',
                          dilation_rate=self.dilation_rate
                          )
        tower2 = K.pool2d(x=tower2,
                          strides=(2,2),
                          pool_size=(3,3),
                          pool_mode='max',
                          padding='same'
                          )
        tower3 = K.pool2d(x=inputs,
                          pool_size=(3, 3),
                          strides=(2, 2),
                          padding='same',
                          pool_mode='max'
                          )
        tower3 = K.conv2d(x=tower3,
                          kernel=self.K4,
                          strides=(1, 1),
                          padding='same',
                          dilation_rate=(1, 1)
                          )
        output = K.concatenate((tower1, tower2, tower3))
        output = K.conv2d(x=output,
                          kernel=self.K5,
                          strides=(1, 1),
                          padding='same',
                          dilation_rate=(1, 1)
                          )
        if self.use_softmax:
            return K.softmax(output)
        else:
            return K.relu(output)

    def compute_output_shape(self, input_shape):
        return input_shape[:-3] + (input_shape[-3]//2,) + (input_shape[-2]//2,) + (self.output_depth,)

class UpscalingLayer(Layer):
    def __init__(self, **kwargs):
        super(UpscalingLayer,self).__init__(**kwargs)

    def build(self, input_shape):
        self.K1 = self.add_weight(
                shape=[9,9,input_shape[-1],input_shape[-1]],
                initializer='glorot_normal',
                trainable=True,
                name=self.name + '_K1'
        )
        super(UpscalingLayer,self).build(input_shape)

    def call(self, inputs, **kwargs):
        a = UpSampling2D()(inputs)
        a = K.conv2d(x=a,
                     kernel=self.K1,
                     padding='same'
                     )
        return a

    def compute_output_shape(self, input_shape):
        return input_shape[:-3] + (input_shape[-3] * 2,) + (input_shape[-2] * 2,) + (input_shape[-1 ],)

class UpscalingBicubic(Layer):
    def __init__(self, **kwargs):
        self.x = 0
        self.y = 0
        super(UpscalingBicubic,self).__init__(**kwargs)
        self.trainable = False

    def build(self, input_shape):
        self.x = input_shape[1]*2
        self.y = input_shape[2]*2

    def call(self, inputs, **kwargs):
        return tf.image.resize_bilinear(images=inputs,
                                       size=(self.x,self.y),
                                       )


    def compute_output_shape(self, input_shape):
        return input_shape[:-3] + (input_shape[-3] * 2,) + (input_shape[-2] * 2,) + (input_shape[-1],)
