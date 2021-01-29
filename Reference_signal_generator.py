import numpy  as np
import scipy
import scipy.linalg

class ReferenceSignal(object):

    def __init__(self, configuration):
        self.coniguration = configuration
        self.reference_map = configuration.reference_map

        # Define initial values for x_init and F
        self.x_init = np.zeros((0,1))
        self.F = None

        # Define sinusoidal signal dynamics
        if configuration.signal_type == 'sinusoidal':
            # Derive amplitudes, frequencies and phases from configuration
            amplitudes = configuration.amplitudes
            frequencies = configuration.frequencies
            phases = configuration.phases

            for i in range(amplitudes.shape[0]):
                amplitude = amplitudes[i]
                frequency = frequencies[i]
                phase = phases[i]

                # Compute the initial reference state and reference (linear) dynamics
                x_init_i, F_i = self.sinusoidal(amplitude, frequency, phase)

                # Combine with other reference signals
                self.x_init =  np.vstack((self.x_init, x_init_i))

                if self.F is None:
                    self.F = np.copy(F_i)
                else:
                    self.F = scipy.linalg.block_diag(self.F, F_i)

        else:
            raise ValueError('Reference signal type: "{}" not implemented'.format(configuration.signal_type))

        return

    def sinusoidal(self, amplitude, frequency, phase):
        radial_freq = 2 * np.pi * frequency

        # Create discrete sine wave by creating SS of mass spring system
        virtual_k = radial_freq ** 2
        F_matrix = np.array([[0, 1],
                             [-virtual_k, 0]])

        # Generate initial position and velocity
        x_r_0 = amplitude * np.sin(phase)
        x_r_0_d = amplitude * radial_freq * np.cos(phase)

        x_r_init = np.asmatrix(np.array([[x_r_0], [x_r_0_d]]))

        return x_r_init, F_matrix
