import numpy as np

class Configuration(object):
    '''
    Class containing all the options used for the agent-environment interaction.
    These options can be overwritten by redefining the attribute value prior to
    agent-environment simulation
    '''

    def __init__(self, curr_step_ind):

        # Output generation booleans
        self.show_plots = True
        self.show_animation = False
        self.save_H_values = True

        # Other booleans
        self.apply_curriculum = True
        self.train_agent = True

        # Number of Offline-learning resets
        self.num_offline_resets = 1

        # Time configuration parameters
        self.duration = 50
        self.dt = 0.01

        # Set up the configuration of the agent
        self.agent_config = AgentConfig(curr_step_ind)

        # Set up the configuration of the environment
        self.env_config = EnvironmentConfig(curr_step_ind)

        # Set up the configuration of the excitation signal
        self.excitation_signal_config = ExcitationSignalConfig(curr_step_ind)

        # Set up the configuration of the measurement noise
        self.measurement_noise_config = MeasurementNoiseConfig()

        # Set up the configuration of the reference signal
        self.reference_signal_config = ReferenceSignalConfig(curr_step_ind)

        # Set up the configuration of the safety filter
        self.safety_filter_config = SafetyFilterConfig(curr_step_ind, self.dt, self.agent_config.action_bounds)

        # Set up the configuration of the system
        self.system_config = SystemConfig(curr_step_ind)

        # Set up the configuration of the system identifier
        self.system_id_config = SystemIDConfig(curr_step_ind)

        # Set up the configuration of transfer learning
        self.transfer_learning_config = TransferLearningConfig(curr_step_ind)

        return


####################################### INDIVIDUAL CONFIGS #######################################

# ---> Agent
class AgentConfig(object):
    '''
    Configuration class that is given to the agent class during the set up of the
    curricular step.
    '''

    def __init__(self, curr_step_ind):
        self.save_H_values = True

        self.learning_rate = 0.8
        self.sample_factor = 20
        self.method = 'VI'  # 'PI' also available

        if curr_step_ind == 0:
            self.action_bounds = np.array([[-5, 5]])

        elif curr_step_ind == 1:
            self.action_bounds = np.array([[-5, 5],
                                           [-5, 5]])

        elif curr_step_ind == 2:
            self.action_bounds = np.array([[-5, 5],
                                           [-5, 5],
                                           [-5, 5]])
        else:
            raise ValueError('Curricular step {} not implemented'.format(curr_step_ind+1))

        return

# ---> Environment
class EnvironmentConfig(object):
    '''
    Configuration class that is given to the environment class during the set up of the
    curricular step.
    '''

    def __init__(self, curr_step_ind):
        self.run_system_checks = True

        if curr_step_ind == 0:
            self.Q_diag_values = np.array([1e5])
            self.R_diag_value = 1e1

        elif curr_step_ind == 1:
            self.Q_diag_values = np.array([1e5, 1e5])
            self.R_diag_value = 1e1

        elif curr_step_ind == 2:
            self.Q_diag_values = np.array([1e5, 1e5, 1e5])
            self.R_diag_value = 1e1

        else:
            raise ValueError('curricular step {} not implemented'.format(curr_step_ind+1))

        self.RMSE_limit = 0.05

        return

# ---> Excitation Signal
class ExcitationSignalConfig(object):
    '''
    Configuration class that is given to the excitation signal class during the set up of the
    curricular step.
    '''

    def __init__(self, curr_step_ind):
        self.signal_type = 'normal'

        if curr_step_ind == 0:
            self.means = np.array([0])
            self.standard_deviations = np.array([1.5])

        elif curr_step_ind == 1:
            self.means = np.array([0, 0])
            self.standard_deviations = np.array([1.5, 1.5])

        elif curr_step_ind == 2:
            self.means = np.array([0, 0, 0])
            self.standard_deviations = np.array([1.5, 1.5, 1.5])

        else:
            raise ValueError('Curricular step {} not implemented'.format(curr_step_ind+1))

        return

# ---> Measurement Noise
class MeasurementNoiseConfig(object):
    '''
    Configuration class that is given to the measurement noise class during the set up of the
    curricular step.
    '''
    def __init__(self):
        self.mean = 0
        self.standard_deviation = 1e-3*1.2
        self.noisy_reference = False

# ---> Reference Signal
class ReferenceSignalConfig(object):
    '''
    Configuration class that is given to the reference signal class during the set up of the
    curricular step.
    '''

    def __init__(self, curr_step_ind):
        self.signal_type = 'sinusoidal'

        self.amplitudes = np.array([0.1])
        self.phases = np.random.random(1) * np.pi
        self.frequencies =np.array([0.5]) * 1/(2*np.pi)

        if curr_step_ind == 0:
            self.reference_map = np.array([[-1,0]])
        elif curr_step_ind == 1:
            self.reference_map = np.array([[ 0, 0],
                                           [-1, 0]])
        elif curr_step_ind == 2:
            self.reference_map = np.array([[ 0, 0],
                                           [ 0, 0],
                                           [-1, 0]])
        else:
            raise ValueError('Curricular step {} not implemented'.format(curr_step_ind+1))

        return

# ---> Safety Filter
class SafetyFilterConfig(object):
    '''
    Configuration class that is given to the safety filter class during the set up of the
    curricular step.
    '''

    def __init__(self, curr_step_ind, dt, action_bounds):
        # Denotes activation of the safety filter (Allows for activating safety filter for certain curricular steps)
        self.activated = True

        if self.activated:
            # Setting dt
            self.dt = dt

            # Setting action bounds (taken from agent configuration)
            self.action_bounds = np.copy(action_bounds)

            # Plot policy simulations
            self.plot_backup_policy_projection = True

            # Define what SHERPA does when no backup policy is found an no safe projection can be found.
            self.last_resort_action = 'random'  # also 'pass-through'

            # Define whether the policy search is processed sequentially or in parallel
            self.policy_search_processing = 'sequential'  # also 'parallel'

            # Iteration parameters
            self.num_iterations_input = 10  # Number of tries to find a backup policy
            self.num_iterations_backup = 10  # Number of policies to be checked
            self.backup_size = 20  # Number of time steps are simulated per policy (in policy search)
            self.backup_projection_size = np.array([15, 20])

            # Range in which the parameters of a randomly generated policy (in policy generation)
            self.policy_parameter_range = np.array([-25, 25])

            ## Curricular Step 1
            if curr_step_ind == 0:
                self.RSS_to_ref_mapping = {0:1}
                self.states_closeness_condition = np.array([0, 1, 2, 3])
                self.RSS = np.array([0])
                self.FSS = np.array([[-0.5, 0.5],
                                     [ 0.0, 0.0]])

                self.sensor_reach = np.array([[-0.1, 0.1],
                                              [ 0.0, 0.0]])

                self.v_epsilon = np.array([[-0.2, 0.2],  # x_1
                                           [-0.2, 0.2],  # x_dot_1
                                           [-0.2, 0.2],  # x_ref
                                           [-0.2, 0.2]]) # x_ref_dot

            ## Curricular Step 2
            elif curr_step_ind == 1:
                self.RSS_to_ref_mapping = {1:1}
                self.states_closeness_condition = np.array([0, 1, 2, 3, 4, 5])
                self.RSS = np.array([1])
                self.FSS = np.array([[ 0.0, 0.0],
                                     [-0.5, 0.5],
                                     [ 0.0, 0.0],
                                     [ 0.0, 0.0]])

                self.sensor_reach = np.array([[-0.1, 0.1],
                                              [-0.1, 0.1],
                                              [ 0.0, 0.0],
                                              [ 0.0, 0.0]])

                self.v_epsilon = np.array([[-0.2, 0.2],  # x_1
                                           [-0.2, 0.2],  # x_2
                                           [-0.2, 0.2],  # x_dot_1
                                           [-0.2, 0.2],  # x_dot_2
                                           [-0.2, 0.2],  # x_ref
                                           [-0.2, 0.2]]) # x_ref_dot

            ## Curricular Step 3
            elif curr_step_ind == 2:
                self.RSS_to_ref_mapping = {2:1}
                self.states_closeness_condition = np.array([0, 1, 2, 3, 4, 5, 6, 7])
                self.RSS = np.array([2])
                self.FSS = np.array([[ 0.0, 0.0],
                                     [ 0.0, 0.0],
                                     [-0.5, 0.5],
                                     [ 0.0, 0.0],
                                     [ 0.0, 0.0],
                                     [ 0.0, 0.0]])

                self.sensor_reach = np.array([[-0.1, 0.1],
                                              [-0.1, 0.1],
                                              [-0.1, 0.1],
                                              [ 0.0, 0.0],
                                              [ 0.0, 0.0],
                                              [ 0.0, 0.0]])

                self.v_epsilon = np.array([[-0.2, 0.2],  # x_1
                                           [-0.2, 0.2],  # x_2
                                           [-0.2, 0.2],  # x_3
                                           [-0.2, 0.2],  # x_dot_1
                                           [-0.2, 0.2],  # x_dot_2
                                           [-0.2, 0.2],  # x_dot_3
                                           [-0.2, 0.2],  # x_ref
                                           [-0.2, 0.2]]) # x_ref_do

            else:
                raise ValueError('Curricular step {} not implemented'.format(curr_step_ind+1))

        return

# ---> System
class SystemConfig(object):
    '''
    Configuration class that is given to the system class during the set up of the
    curricular step.
    '''

    def __init__(self, curr_step_ind):
        # Plotting Booleans
        self.plot_static = True
        self.plot_animation = False

        # Position plot bounds
        self.position_plot_bounds = np.array([-0.5, 0.5])

        # Bounds for the initial conditions (position and velocity)
        self.init_bounds_x = np.array([-0.2, 0.2])
        self.init_bounds_x_dot = np.array([-0.25, 0.25])

        # Curricular step 1
        if curr_step_ind == 0:
            # Define number of masses in the MSD system
            self.n_masses = 1

            # Define system characteristics
            self.masses = np.array([0.4])
            self.all_k = np.array([4])
            self.all_c = np.array([3])

            # Create C matrix
            self.C = np.hstack((np.identity(self.n_masses),
                                np.zeros((self.n_masses, self.n_masses))))

        # Curricular step 2
        elif curr_step_ind == 1:
            # Define number of masses in the MSD system
            self.n_masses = 2

            # Define system characteristics
            self.masses = np.array([0.8, 0.4])
            self.all_k = np.array([ 3, 4])
            self.all_c = np.array([-1, 3])

            # Create C matrix
            self.C = np.hstack((np.identity(self.n_masses),
                                np.zeros((self.n_masses, self.n_masses))))


        # Curricular step 3
        elif curr_step_ind == 2:
            # Define number of masses in the MSD system
            self.n_masses = 3

            # Define system characteristics
            self.masses = np.array([0.3, 0.8, 0.4])
            self.all_k = np.array([-1,  3, 4])
            self.all_c = np.array([ 6, -1, 3])

            # Create C matrix
            self.C = np.hstack((np.identity(self.n_masses),
                                np.zeros((self.n_masses, self.n_masses))))

        else:
            raise ValueError('No configuration provided for curricular step {}'.format(curr_step_ind))

        return

# --> System Identification
class SystemIDConfig(object):
    '''
    Configuration
    '''

    def __init__(self, curr_step_ind):

        # Boolean whether to use sliding window principle or not
        self.use_sliding_window = True

        # Define sample factor for each curricular step
        if curr_step_ind == 0:
            self.sample_factor = 8
        elif curr_step_ind == 1:
            self.sample_factor = 9
        elif curr_step_ind == 2:
            self.sample_factor = 10
        else:
            raise ValueError('curricular step {} not implemented'.format(curr_step_ind))

        return

# --> Transfer Learning
class TransferLearningConfig(object):
    '''

    '''

    def __init__(self, curr_step_ind):
        self.curr_step_ind = curr_step_ind

        # Boolean denoting the use of the supervisor policy as initial value for agent policy
        self.use_supervisor_policy_as_basis = False

        return

##################################################################################################

    # def import_options(self, options_dict):
    #     '''
    #     Setting options from an external dictionary
    #
    #     :param options_dict: external dictionary options
    #     :return:
    #     '''
    #
    #     for key in options_dict.keys():
    #         setattr(self, key, options_dict[key])