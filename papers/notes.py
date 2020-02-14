import numpy as np

"""
define the state signal vector, in d dimensions, s(t)

s'(t) = f(t) + 𝝓,
where 𝝓 is a constant d dimensional vector, f(t) is a d dimensional vector function of time
f(t) is typically written as:
    f(t) = h(s(t), x(t)) + 𝝓
that is f(t) is a function is a function of the state vector and the input signal vector
This gives us:

    s'(t) = h(s(t),x(t)) + 𝝓

Which is a common equation throughout physics, biolofy, and chemistry

One special case of f(t) is the so called "additive model" in brain dynamics:
"""

def additive_model(t):
    return a(t) + b(t) + c(t)

"""
We can saturate or shunt the additive model by defining a, b, and c,
bounding the system
Lets consider a case of the attitive model:
"""

def a(a_k,state,t,delay):
    out = []
    for i in range(len(state)):
        out.append(a_k[i](s[i]*(t-delay(i))))
    return sum(out)


def G(x):
    return np.tanh(x)

def readout_signal(state,t,delay):
    return G(state*(t-delay))

def b(b_k,state,t,delay):
    out = []
    for i in range(len(signal)):
        out.append(
            b_k(readout_signal(state[i], t[i], delay[i]))
        )
    return sum(out)


def c(c_k, signal, t, delay):
    out = []
    for i in range(len(signal)):
        out.append(
            c_k[i](signal[i]*(t-delay[i]))
        )
    return sum(out)

def dsdt(a_k, b_k, c_k, signal, state, t, delay, bias):
    a = a(a_k, state, t, delay)
    b = b(b_k, state, t, delay)
    c = c(c_k, signal, t, delay)
    return sum([a,b,c,bias])

'''
equation 9

so A is the combination of the some set of functions of the statevector,
B is the combination of  of some set of functions of the warped singal vector, which
is basically the signal multiplied by a nonlinearity, or activation function
C is a combination of some set of functions of the signal vector, and 𝝓 is
just a bias constant.

We use something like tanh because it is monotonic (dy/dx stays the same time), has
a linear region, and  it is bounded at the ends, which squashes the system state and stops explosions.
the linear region helps it behave in a normal manner, its basically a squashed linear regression. Information is
preserved, but still bounded, which is really nice

The time delay makes up the "memory" of the network.
'''

'''
equations 11 and 12:
    lets assume that our three functions a, b, and c are combinations of linear functions.
    This allows us to say that a_k, b_k, and c_k are are actually matrices of linear coefficients

    Next, lets also assume that a_k, b_k, and c_k are circulant matrices
    (reference: https://en.wikipedia.org/wiki/Circulant_matrix, just look at the picture)
'''

