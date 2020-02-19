from tensorflow.keras.layers import Dense, Input, GRU, Add
from activations import Mish, sparsemax, SparsemaxLoss
from optimizers import Ranger, Yogi, Lookahead
from layers import Attention
from tensorflow.keras.models import Model


def gru(inn, nodes=40, **kwargs):
    # this is the only way it works!
    return GRU(nodes, activation=Mish(),  return_state=True, return_sequences=True)(inn, **kwargs)
def block(inn, nodes=40, **kwargs):
    val, state = gru(inn, nodes,**kwargs)
    val2 = Attention()(val)
    return val, state, val2
def build_att_gru(n_time, n_out, nodes=40,**optim_args):
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
