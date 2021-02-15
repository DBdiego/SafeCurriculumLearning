import numpy as np

# MSD configurations
from .MSD_CurL_No_Noise import Configuration as msd_curl_no_noise_config
from .MSD_CurL_Noise import Configuration as msd_curl_noise_config
from .MSD_FL_Noise import Configuration as msd_fl_noise_config

# Quadrotor configurations
from .Quadrotor_CurL_No_Noise_SHERPA import Configuration as quadrotor_curl_no_noise_safety_filter_config
from .Quadrotor_CurL_No_Noise import Configuration as quadrotor_curl_no_noise_config
from .Quadrotor_FL_No_Noise import Configuration as quadrotor_fl_no_noise_config


def select_configuration_class(system, impose_FL, add_measurement_noise, add_safety_filter):
    '''
    Selects a predefined configuration module based on the inputs provided to the function.

    :param system: which system is to be simulated ('msd' and 'quadrotor' are available)
    :param impose_FL: Boolean denoting the learning strategy (False = curriculum learning, True = flat learning)
    :param add_measurement_noise: Boolean denoting the presence of measurement noise
                                  (False = no noise, True = with noise)
    :return: configuration class, an uninitiated class of contained in the desired coniguration module
    '''

    # Check if the requested system is implemented
    avail_systems = ['msd', 'quadrotor']
    if system.lower() not in avail_systems:
        error_str = 'No configuration is implemented for a {} system.\n'.format(system)
        error_str += 'Either the __init__.py file in the Configurations package needs to be adapted or the system was misspelled.\n'
        error_str += '  Available systems: ' + ', '.join(avail_systems)
        raise ValueError(error_str)

    # Converting inputs to bools
    impose_FL = bool(impose_FL)
    add_measurement_noise = bool(add_measurement_noise)
    add_safety_filter = bool(add_safety_filter)

    # Define simulation string
    noise_str = 'no_'*np.invert(add_measurement_noise) + 'measurement_noise'
    learning_str = 'FL'*impose_FL + 'CurL'*np.invert(impose_FL)
    safety_filter_str = 'no_'*np.invert(add_safety_filter)+'safety_filter'
    sim_str = '_'.join([system, learning_str, safety_filter_str, noise_str])

    ## Mass-Spring-Damper System
    if system.lower() == avail_systems[0]:
        # Flat learning without safety filter (without measurement noise)
        if impose_FL and not add_safety_filter and not add_measurement_noise:
            configuration_class = msd_fl_noise_config

        # Flat learning without safety filter (with measurement noise)
        elif impose_FL and not add_safety_filter and add_measurement_noise:
            configuration_class = msd_fl_noise_config

        # Flat learning with safety filter (with/without measurement noise)
        elif impose_FL and add_safety_filter:
            raise ValueError('No configuration for {} simulation'.format(sim_str))

        # Curriculum Learning without safety filter (without measurement noise)
        elif not impose_FL and not add_safety_filter and not add_measurement_noise:
            configuration_class = msd_curl_no_noise_config

        # Curriculum Learning without safety filter (without measurement noise)
        elif not impose_FL and add_safety_filter and not add_measurement_noise:
            configuration_class = msd_curl_no_noise_config

        # Curriculum Learning without safety filter (with measurement noise)
        elif not impose_FL and not add_safety_filter and add_measurement_noise:
            configuration_class = msd_curl_noise_config

        # Curriculum Learning with safety filter (without measurement noise)
        elif not impose_FL and add_safety_filter and not add_measurement_noise:
            raise ValueError('No configuration for {} simulation'.format(sim_str))

        # Curriculum Learning with safety filter (with measurement noise)
        elif not impose_FL and add_safety_filter and add_measurement_noise:
            configuration_class = msd_curl_noise_config

        else:
            raise ValueError('Logical error in the implementation')

    ## Quadrotor System
    elif system.lower() == 'quadrotor':
        if add_measurement_noise:
            raise ValueError('No configuration for {} simulation'.format(sim_str))
        else:
            # Flat Learning without safety filter (no measurement noise)
            if impose_FL and not add_safety_filter:
                configuration_class = quadrotor_fl_no_noise_config

            # Flat learning with safety filter (no measurement noise)
            elif impose_FL and add_safety_filter:
                raise ValueError('No configuration for {} simulation'.format(sim_str))

            # Curriculum Learning without safety filter (no measurement noise)
            elif not impose_FL and not add_safety_filter:
                configuration_class = quadrotor_curl_no_noise_config

            # Curriculum Learning with safety filter (no measurement noise)
            elif not impose_FL and add_safety_filter:
                configuration_class = quadrotor_curl_no_noise_safety_filter_config

            else:
                raise ValueError('Logical error in the implementation')

    else:
        raise ValueError('The given system "{}" is not implemented'.format(system))

    print('Simulation Configuration: {}'.format(sim_str.replace('_',' ')))
    return configuration_class
