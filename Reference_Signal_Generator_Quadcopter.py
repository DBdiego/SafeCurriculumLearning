import numpy as np


class ReferenceSignal(object)
    '''
    
    '''
    def __init__(self, configuration):

        self.F = np.array([[configuration.dt, 0,0], [0,0,0], [0,0,0]])

        self.x_init =