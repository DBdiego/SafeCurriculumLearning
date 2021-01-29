import numpy as np


class ExcitationSignal(object):
    '''
    Class creating an excitation signal
    '''

    def __init__(self, configuration):

        self.signal_type = configuration.signal_type

        if self.signal_type == 'normal':
            self.mu = configuration.means
            self.sigma = configuration.standard_deviations

        elif self.signal_type == 'frequency_limited':
            self.mu = configuration.means
            self.amplitudes = configuration.amplitudes
            self.frequency_ranges = configuration.frequency_ranges

        else:
            raise ValueError('Excitation signal type "{}" not implemented'.format(self.signal_type))

        return

    def generate_excitation_time_series(self, m, times):
        '''

        :param m:
        :param times:
        :return:
        '''

        excitation_signals = np.zeros((m, times.shape[0]))
        for i in range(self.mu.shape[0]):
            if self.signal_type == 'normal':
                excitation_signals[i,:] = np.random.normal(self.mu[i], self.sigma[i], (times.shape[0],))

            elif self.signal_type == 'frequency_limited':
                excitation_signals[i,:] = self.random_noise(times,
                                                            self.mu[i],
                                                            self.amplitudes[i],
                                                            self.frequency_ranges[i,:])

        return excitation_signals

    def random_noise(self, times, mu, amplitude, frequency_range):
        '''
        Define noise signal, with frequency limitations

        :param times: array of time steps
        :return: noisy time series
        '''

        # Randomly generate a set of frequencies within the defined band
        freqs = frequency_range[0] + np.random.random(20) * np.diff(frequency_range)[0]

        a2 = amplitude / 2.5
        times = np.copy(times) + 1

        noise = a2 * (0.5 * np.sin(freqs[0] * times) ** 2 * np.cos(freqs[1] * times)
                      + 0.9 * np.sin(freqs[2] * times) ** 2 * np.cos(freqs[3] * times)
                      + 0.3 * np.sin(freqs[4] * times) ** 2 * np.cos(freqs[5] * times)
                      + 0.3 * np.sin(freqs[6] * times) ** 3
                      + 0.7 * np.sin(freqs[7] * times) ** 2 * np.cos(freqs[8] * times)
                      + 0.3 * np.sin(freqs[9] * times) * 1 * np.cos(freqs[10] * times) ** 2
                      + 0.4 * np.sin(freqs[11] * times) ** 2
                      + 0.5 * np.cos(freqs[12] * times) * np.sin(freqs[13] * times) ** 2
                      + 0.3 * np.sin(freqs[14] * times) ** 1 * np.cos(freqs[15] * times) ** 2
                      + 0.3 * np.sin(freqs[16] * times) ** 3
                      + 0.4 * np.cos(freqs[17] * times) * 1 * np.sin(freqs[18] * times) ** 4
                      + 0.3 * np.sin(freqs[19] * times) ** 3)

        # # Convert to white noise:
        if noise.shape[0] >= 4000:
            noise[:4000] = noise[:4000] - np.mean(noise[:4000])
            noise[4000:] = noise[4000:] - np.mean(noise[4000:])
        else:
            noise = noise - np.mean(noise)

        # Adjust noise to provided mean, mu
        noise += mu

        return noise
