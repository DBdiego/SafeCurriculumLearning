import numpy as np


class MeasurementNoise(object):
    '''

    '''

    def __init__(self, configuration):
        self.mu = configuration.mean
        self.sigma = configuration.standard_deviation
        self.noisy_reference = configuration.noisy_reference
        # self.noise_bounds = np.array([-0.001, 0.001])
        # self.noise_bounds = np.array([0, 0])

        self.variance = self.sigma * self.sigma

        return


    def add_measurement_noise(self, state, n, k):

        if self.noisy_reference:
            noise = np.random.normal(self.mu, self.sigma, (state.shape[0], 1))
            # noise = self.noise_bounds[0] + np.random.random((state.shape[0], 1))*np.diff(self.noise_bounds)
        else:
            noise = np.zeros(state.shape)
            # noise_value = self.random_noise(k)
            # noise[:n, :] = np.ones((n, 1)) * noise_value
            noise[:n,:] = np.random.normal(self.mu, self.sigma, (n, 1))

            # noise[:n, :] = self.noise_bounds[0] + np.random.random((n, 1))*np.diff(self.noise_bounds)

        noisy_state = state + noise

        return noisy_state

    def random_noise(self, time_step):
        '''
        Define noise signal

        :param time_step: array of time steps
        :return:
        '''

        a2 = self.sigma
        time_step = time_step + 1
        noise = a2 * (0.5 * np.sin(2.0 * time_step) ** 2 * np.cos(10.1 * time_step)
                      + 0.9 * np.sin(1.102 * time_step) ** 2 * np.cos(4.001 * time_step)
                      + 0.3 * np.sin(1.99 * time_step) ** 2 * np.cos(7 * time_step)
                      + 0.3 * np.sin(10.0 * time_step) ** 3
                      + 0.7 * np.sin(3.0 * time_step) ** 2 * np.cos(4.0 * time_step)
                      + 0.3 * np.sin(3.00 * time_step) * 1 * np.cos(1.2 * time_step) ** 2
                      + 0.4 * np.sin(1.12 * time_step) ** 2
                      + 0.5 * np.cos(2.4 * time_step) * np.sin(8 * time_step) ** 2
                      + 0.3 * np.sin(1.000 * time_step) ** 1 * np.cos(0.799999 * time_step) ** 2
                      + 0.3 * np.sin(4 * time_step) ** 3
                      + 0.4 * np.cos(2 * time_step) * 1 * np.sin(5 * time_step) ** 4
                      + 0.3 * np.sin(10.00 * time_step) ** 3)

        return noise
