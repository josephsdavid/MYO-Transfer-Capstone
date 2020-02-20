## stacked att-gru imports from tf
from tensorflow.keras.layers import Dense, Input, GRU, Add
# wavenet layers from tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Activation
from tensorflow.keras.layers import Dropout,  Multiply, Conv1D, AveragePooling1D, BatchNormalization
# FCNN-LSTM layers from tf:
from tensorflow.keras.layers import GlobalAveragePooling1D, Masking, PReLU
# conv-rnn imports:
from tensorflow.keras.layers import TimeDistributed, MaxPooling1D
# convlstm imports
from tensorflow.keras.layers import ConvLSTM2D
# other imports
from tensorflow.keras.models import Model
from activations import Mish
from optimizers import Ranger
from layers import Attention

'''
'''
'''
Stacked GRU with attention
--------
main export: build_att_gru

inputs: n_time(timesteps), n_out(classes), nodes, **arguments for ranger

outputs: compiled model

'''
def gru(inn, nodes=40, **kwargs):
    # this is the only way it works!
    return GRU(nodes, activation=Mish(),  return_state=True, return_sequences=True)(inn, **kwargs)
def block(inn, nodes=40, **kwargs):
    val, state = gru(inn, nodes,**kwargs)
    val2 = Attention()(val)
    return val, state, val2
def build_att_gru(n_time, n_out, nodes=40,**optim_args):
    '''
    build_att_gru
    --------------
    args:
        n_time
        n_out
        nodes=40
        arguments for ranger
    '''
    inputs = Input((n_time, 8))
    x = Dense(128)(inputs)
    x1, h, a1 = block(x, nodes)
    x2, h, a2 = block(x1, nodes, initial_state=h)
    x3, h, a3 = block(x2, nodes, initial_state=h)
    out = Add()([a1,a2,a3])
    #out = Dense(60, activation=PReLU())(out)
    outputs = Dense(n_out, activation="softmax")(out)
    model = Model(inputs, outputs)
#    print(model.summary())
    model.compile(Ranger(**optim_args), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


'''
Wavenet:
---------------------
builds a wavenet model
Use the build_model() method after init to
get a wavenet
'''
class WaveNet:
    def __init__(self, input_shape, output_shape, kernel_size=2, filters=40, dilation_depth=9):
        '''
        wavenet: build a wavenet model
        args: input_shape, output_shape
        kernel_size=2,
        filters=40,
        dilation_depth=9
        outputs:
            use the build_model method to build the model. Note the model isnt compiled
        '''
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
        x = BatchNormalization()(x)
        for i in range(1, self.dilation + 1):
            x, skip = self.residual_block(x, i)
            skips.append(skip)
        x = Add(name='skips')(skips)
        x = Activation('relu')(x)
        x = Conv1D(self.filters, 3, strides=1, padding='same', name = 'conv_5ms', activation='relu')(x)
        x = BatchNormalization()(x)
        x = AveragePooling1D(3, padding='same', name='downsample')(x)
        x=Dropout(0.5)(x)
        x=Conv1D(self.filters, 3, padding='same', activation='relu', name='upsample')(x)
        x = Conv1D(self.output_shape[0], 3, padding='same', activation='relu', name='target')(x)
        x = BatchNormalization()(x)
        x = AveragePooling1D(3, padding='same', name = 'downsample_2')(x)
        x=Dropout(0.5)(x)
        x = Conv1D(self.output_shape[0], self.input_shape[0] //10, padding='same', name = 'final')(x)
        x = BatchNormalization()(x)
        x = AveragePooling1D(self.input_shape[0] // 10, name = 'final_pool')(x)
        x=Dropout(0.5)(x)
        x = Reshape(self.output_shape)(x)
        out = Activation(self.out_act)(x)
        mod = Model(inp, out)
        return mod


'''
Fully connected conv lstm:
    use the build_fcnn_lstm function to build a fully connected convolutional lstm
    probably there is something wrong with this as i found it online somewhere,
    but it does indeed run
'''
def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    filters = input.shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se

def build_fcnn_lstm(n_time, n_classes, n_cells=8):
    '''
    build_fcnn_lstm:
    -----------------
    builds a fully connected convolutional lstm, following some paper and
    medium article. Probably not correct. But runs
    args:
        n_time: number of timesteps
        n_classes: number of classes
        n_cells: number of lstm cells
    outputs:
        non compiled model
    '''
    ip = Input(shape=(n_time,8))

    x = Masking()(ip)
    x = LSTM(n_cells, recurrent_dropout=0.8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2,1))(ip)

    y = Conv1D(128,8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = PReLU()(y)
    y = squeeze_excite_block(y)


    y = Conv1D(256,5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = PReLU()(y)
    y = squeeze_excite_block(y)


    y = Conv1D(128,3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = PReLU()(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x,y])

    out = Dense(n_classes, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    return model


'''
conv-rnn:
    use the build_conv_rnn() function!
outputs:
    uncompiled model
'''
def build_conv_rnn(n_time, n_classes, filters=[20, 64, 64, 64],
                   kernels=[7,5,5,5], lstm=[52,52]):
    '''
    build_conv_rnn
    ----------------
    builds a conv_rnn
    args:
        n_time: timesteps,
        n_classesL num classes
        filters: list of filters to use
        kernels=list of kernels to use
        lstm: list of lstm sizes
    outputs:
        not compiled model
    '''
    inputs = Input((n_time, 8))
    x = inputs
    for f, k in zip(filters, kernels):
        x = TimeDistributed(Conv1D(filters=f, kernel_size=k, activation='relu'))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(Flatten())(x)
    for l in range(len(lstm)):
        seq = True if (l != len(lstm)-1) else False
        x = LSTM(lstm[l], dropout=0.2, return_sequences=seq)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    outputs = Dense(n_classes, activation='relu')(x)
    return Model(inputs, outputs)


'''
concolutonal nn
use build_cnn()
'''

def build_cnn(n_time, n_classes, filters=[20, 64, 64, 64],
              kernels=[7,5,5,5]):
    '''
    build_cnn
    ----------------
    builds a cnn
    args:
        n_time: timesteps,
        n_classesL num classes
        filters: list of filters to use
        kernels=list of kernels to use
    outputs:
        not compiled model
    '''
    inputs = Input((n_time, 8))
    x = inputs
    for f, k in zip(filters, kernels):
        x = Conv1D(filters=f, kernel_size=k, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    return Model(inputs, outputs)


'''
convlstm
build_convlstm()
'''

def build_convlstm(n_time, n_classes,
                   filters=[12,24,24],
                   kernels=[(1,5), (1,3), (1,3)],
                   dense=512):
    '''
    build_convlstm
    --------------
    args:
        n_time: timesteps
        n_classes: classes
        filters: list of filters
        kernels: list of tuple kernels
        dense: size of dense layer
    '''
    inputs = Input((None, 1, n_time, 8))
    x = inputs
    i = 1
    for f,k in zip(filters, kernels):
        seq = True if (i!=len(filters)) else False
        x = ConvLSTM2D(filters=f, kernel_size=k,
                       data_format='channels_last',
                       padding='same', return_sequences=seq)(x)
        if seq:
            x = BatchNormalization()(x)
        i += 1
    x = Flatten()(x)
    x = Dense(dense, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    return Model(inputs, outputs)



'''
simple_attention

a simple attention model
'''

def build_simple_att(n_time, n_class, dense = [500,500,2000], drop=[0.1, 0.1, 0.1],activation='relu'):
    inputs = Input((n_time, 8))
    x = inputs
    x = Dense(128, activation=activation)(x)
    x = Attention()(x)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x)
        x = Dense(d, activation=activation)(x)
    outputs = Dense(n_class, activation='softmax')(x)
    return Model(inputs, outputs)