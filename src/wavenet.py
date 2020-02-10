import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape, Activation, Dropout, Add, TimeDistributed, Multiply, Conv1D, Conv2D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import History, ModelCheckpoint


class WaveNet:
    def __init__(self, input_shape, output_shape, kernel_size=2, filters=40, dilation_depth=9):
        self.out_act = 'softmax'

        if len(input_shape) != 2:
            print("are you sure this is a time series..")
            return
        if len(output_shape) !=1:
            print("wrong output shape! Should be 1D")

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_size =  kernel_size
        self.dilation = dilation_depth
        self.filters = filters
        self.model = self.build_model()


    def _make_tanh(self, dilation_rate):
        tanh = Conv1D(self.filters,
                      self.kernel_size,
                      dilation_rate = dilation_rate,
                      padding='causal',
                      name = 'dilated_conv_{}_tanh'.format(dilation_rate),
                      activation='tanh')
        return tanh

    def _make_signmoid(self, dilation_rate):
        sigmoid = Conv1D(self.filters,
                      self.kernel_size,
                      dilation_rate = dilation_rate,
                      padding='causal',
                      name = 'dilated_conv_{}_sigmoid'.format(dilation_rate),
                      activation='sigmoid')
        return sigmoid

    def residual_block(self, x, i):
        dr = self.kernel_size**i
        tanh = self._make_tanh(dr)
        sigm = self._make_signmoid(dr)
        z = Multiply(name='gated_activation_{}'.format(i))([fn(x) for fn in [tanh, sigm]])
        skip = Conv1D(self.filters, 1, name = 'skip_{}'.format(i))(z)
        res = Add(name = 'residual_{}'.format(i))([skip, x])
        return res, skip

    def get_model(self):
        return self.model
    def build_model(self):
        inp = Input(shape = self.input_shape)
        skips = []
        x = Conv1D(self.filters, 2, dilation_rate=1, padding='causal', name = 'dilate_1')(inp)
        for i in range(1, self.dilation + 1):
            x, skip = self.residual_block(x, i)
            skips.append(skip)
        x = Add(name='skips')(skips)
        x = Activation('relu')(x)
        x = Conv1D(self.filters, 3, strides=1, padding='same', name = 'conv_5ms', activation='relu')(x)
        x = AveragePooling1D(3, padding='same', name='downsample')(x)
        x=Conv1D(self.filters, 3, padding='same', activation='relu', name='upsample')(x)
        x = Conv1D(self.output_shape[0], 3, padding='same', activation='relu', name='target')(x)
        x = AveragePooling1D(3, padding='same', name = 'downsample_2')(x)
        x = Conv1D(self.output_shape[0], self.input_shape[0] //10, padding='same', name = 'final')(x)
        x = AveragePooling1D(self.input_shape[0] // 10, name = 'final_pool')(x)
        x = Reshape(self.output_shape)(x)
        out = Activation(self.out_act)(x)
        mod = Model(inp, out)
        return mod

