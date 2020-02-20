from tensorflow.keras.layers import Dense, Input, GRU, Add, Dropout
from tensorflow.keras.models import Model
from activations import Mish
from optimizers import Ranger
from layers import Attention

def gru(inn, nodes=40, **kwargs):
    # this is the only way it works!
    return GRU(nodes, activation=Mish(),  return_state=True, return_sequences=True)(inn, **kwargs)


def block(inn, nodes=40,**kwargs):
    val, state = gru(inn, nodes,**kwargs)
    val2 = Attention()(val)
    return val, state, val2


def build_att_gru(n_time, n_classes, nodes=40, blocks=3,
                  loss='categorical_crossentropy', optimizer=Ranger, **optim_args):
    '''
    build_att_gru
    --------------
    args:
        n_time
        n_out
        nodes=40
        blocks=3, represents number of att_gru blocks in the model
        loss='categorical_crossentropy'
        optimizer=Ranger (no parentheses)
        args for optimizer
    Notes:
        returns compiled model.
        requires by default one hot encoded Y data
    '''
    inputs = Input((n_time, 8))
    x = Dense(128)(inputs)
    x, h, a = block(x, nodes)
    attention=[a]
    for _ in range(blocks-1):
        x, h, a = block(x, nodes, initial_state=h)
        attention.append(a)
    out = Add()(attention)
    outputs = Dense(n_classes, activation="softmax")(out)
    model = Model(inputs, outputs)
    model.compile(optimizer(**optim_args), loss=loss,  metrics=['accuracy'])
    print(model.summary())
    return model



def build_simple_att(n_time, n_class, dense = [50,50,50], drop=[0.1, 0.1, 0.1]):
    '''
    build_simple_att
    ------------------
    args:
        n_time
        n_class
        dense = [50,50,50], list of dense node sizes
        drop = [0.1, 0.1, 0.1] list of dropout
    '''
    inputs = Input((n_time, 8))
    x = inputs
    x = Dense(128, activation=Mish())(x)
    x = Attention()(x)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x)
        x = Dense(d, activation=activation)(x)
    outputs = Dense(n_class, activation='softmax')(x)
    return Model(inputs, outputs)
