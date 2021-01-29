import numpy as np


########### P CONTROLLER ###########

class ProportionalController(object):

    def __init__(self, x_init, x_d_init):
        state_shape = (2, x_init.shape[0])
        self.state = np.zeros(state_shape)
        self.state[0,:] = x_init
        self.state[1,:] = x_d_init
        self.action = np.zeros(x_init.shape[0])
        self.gain = 0



    def policy(self, state):
        return self.gain * state[0, :]
