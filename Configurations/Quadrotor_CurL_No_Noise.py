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

        # Number of Offline-learning resets
        self.num_offline_resets = 1

        # Time configuration parameters
        if curr_step_ind == 0:
            self.train_agent = True
            self.duration = 30
        elif curr_step_ind == 1:
            self.train_agent = True
            self.duration = 40
        elif curr_step_ind == 2:
            self.train_agent = True
            self.duration = 40
        elif curr_step_ind == 3:
            self.train_agent = True
            self.duration = 50
        elif curr_step_ind == 4:
            self.train_agent = False
            self.duration = 30
        self.dt = 0.005

        # Set up the configuration of the system
        self.system_config = SystemConfig(curr_step_ind)

        # Set up the configuration of the agent
        self.agent_config = AgentConfig(curr_step_ind, self.system_config)

        # Set up the configuration of the environment
        self.env_config = EnvironmentConfig(curr_step_ind)

        # Set up the configuration of the excitation signal
        self.excitation_signal_config = ExcitationSignalConfig(curr_step_ind)

        # Set up the configuration of the reference signal
        self.reference_signal_config = ReferenceSignalConfig(curr_step_ind, self.dt)

        # Set up the configuration of the safety filter
        self.safety_filter_config = SafetyFilterConfig(curr_step_ind, self.dt, self.agent_config.action_bounds)

        # Set up the configuration of the supervisor controller
        self.supervisor_config = SupervisorConfig(curr_step_ind)

        # Set up the configuration of the system
        self.system_config = SystemConfig(curr_step_ind)

        # Set up the configuration of the system identifier
        self.system_id_config = SystemIDConfig(curr_step_ind)

        # Set up the configuration of transfer learning
        self.transfer_learning_config = TransferLearningConfig(curr_step_ind, self.system_config.xi)

        return


####################################### INDIVIDUAL CONFIGS #######################################

# ---> Agent
class AgentConfig(object):
    '''
    Configuration class that is given to the agent class during the set up of the
    curricular step.
    '''

    def __init__(self, curr_step_ind, system_config):
        self.save_H_values = True

        self.learning_rate = 0.8

        if curr_step_ind == 0:
            self.method = 'VI'  # 'PI' also available
            self.sample_factor = 8
            self.action_bounds = np.array([[-20, 20],  # F_t [N]
                                           [-15, 15],  # tau_x [Nm]
                                           [-15, 15],  # tau_y [Nm]
                                           [-20, 20]]) # tau_z [Nm]

        elif curr_step_ind == 1:
            self.method = 'VI'
            self.sample_factor = 8
            self.action_bounds = np.array([[-20, 20],  # F_t [N]
                                           [ -3,  3],  # tau_x [Nm]
                                           [-15, 15],  # tau_y [Nm]
                                           [-20, 20]]) # tau_z [Nm]

        elif curr_step_ind == 2:
            self.method = 'VI'
            self.sample_factor = 8
            self.action_bounds = np.array([[-20, 20],  # F_t [N]
                                           [-15, 15],  # tau_x [Nm]
                                           [-15, 15],    # tau_y [Nm]
                                           [-20, 20]]) # tau_z [Nm]

        elif curr_step_ind == 3:
            self.method = 'VI'
            self.sample_factor = 8
            self.action_bounds = np.array([[-80, 80],  # F_t [N]
                                           [-15, 15],  # tau_x [Nm]
                                           [-15, 15],  # tau_y [Nm]
                                           [-20, 20]]) # tau_z [Nm]

        elif curr_step_ind == 4:
            self.method = 'VI'
            self.sample_factor = 8

            self.action_bounds = np.array([[-2500, 2500],  # Rotor 1 RPM
                                           [-2500, 2500],  # Rotor 2 RPM
                                           [-2500, 2500],  # Rotor 3 RPM
                                           [-2500, 2500]]) # Rotor 4 RPM

            # self.action_bounds = system_config.hover_RPM + self.action_bounds

        else:
            raise ValueError('curricular step {} not implemented'.format(curr_step_ind+1))

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
            self.Q_diag_values = np.array([1e5, 1e2])
            self.R_diag_values = np.array([1e1])
            self.R_diag_value = 1e1

        elif curr_step_ind == 1:
            self.Q_diag_values = np.array([1e5, 1e5, 1e2])
            self.R_diag_values = np.array([1e1, 1e1])
            self.R_diag_value = 1e1

        elif curr_step_ind == 2:
            self.Q_diag_values = np.array([1e5, 1e5, 1e5, 1e2])
            self.R_diag_values = np.array([1e1, 1e1, 1e1])
            self.R_diag_value = 1e1

        elif curr_step_ind == 3:
            self.Q_diag_values = np.array([1e5, 1e5, 1e5, 1e6, 1e4])
            self.R_diag_values = np.array([1e1, 1e1, 1e1, 1e1])
            self.R_diag_value = 1e1

        elif curr_step_ind == 4:
            self.Q_diag_values = np.array([1e5, 1e5, 1e5, 1e5])
            self.R_diag_values = np.array([1e1, 1e1, 1e1, 1e1])
            self.R_diag_value = 1e1

        else:
            raise ValueError('curricular step {} not implemented'.format(curr_step_ind+1))

        self.RMSE_limit = 0.5

        return

# ---> Excitation Signal
class ExcitationSignalConfig(object):
    '''
    Configuration class that is given to the excitation signal class during the set up of the
    curricular step.
    '''

    def __init__(self, curr_step_ind):
        self.signal_type = 'frequency_limited'

        # Yaw angle and rate control
        if curr_step_ind == 0:
            self.means = np.array([0.0, 0.0, 0.0, 0.0])
            self.amplitudes = np.array([10.5, 13, 13, 21.0])
            self.frequency_ranges = np.array([[50, 100],  # [rad/s]
                                              [50, 100],  # [rad/s]
                                              [50, 100],  # [rad/s]
                                              [50, 100]]) # [rad/s]

        # Roll angle and rate control
        elif curr_step_ind == 1:
            self.means = np.array([0.0, 0.0, 0.0, 0.0])
            #self.amplitudes = np.array([10.5, 5.0, 5.0, 21.0])
            self.amplitudes = np.array([10.5, 10.0, 10.0, 21.0])
            self.frequency_ranges = np.array([[50, 100],  # [rad/s]
                                              [50, 100], #[50, 200],#[500, 800],  # [rad/s]
                                              [50, 200],  # [rad/s]
                                              [50, 100]]) # [rad/s]

        # Pitch angle and rate control
        elif curr_step_ind == 2:
            self.means = np.array([0.0, 0.0, 0.0, 0.0])
            self.amplitudes = np.array([10.5, 10.0, 10.0, 21.0])
            self.frequency_ranges = np.array([[50, 100],  # [rad/s]
                                              [50, 100],  # [rad/s]
                                              [50, 200],  # [rad/s]
                                              [50, 100]]) # [rad/s]

        # Altitude control
        elif curr_step_ind == 3:
            self.means = np.array([0.0, 0.0, 0.0, 0.0])
            self.amplitudes = np.array([60.5, 15.0, 15.0, 21.0])
            self.frequency_ranges = np.array([[50, 100],  # [m/s]
                                              [50, 100],  # [rad/s]
                                              [50, 100],  # [rad/s]
                                              [50, 100]]) # [rad/s]

        # Rotor RPM control
        elif curr_step_ind == 4:
            self.means = np.array([0.0, 0.0, 0.0, 0.0])
            self.amplitudes = np.array([1e4, 1e4, 1e4, 1e4])
            self.frequency_ranges = np.array([[100, 150],  # [rad/s]
                                              [100, 150],  # [rad/s]
                                              [100, 150],  # [rad/s]
                                              [100, 150]]) # [rad/s]

        else:
            raise ValueError('curricular step {} not implemented'.format(curr_step_ind+1))
        return

# ---> Reference Signal
class ReferenceSignalConfig(object):
    '''
    Configuration class that is given to the reference signal class during the set up of the
    curricular step.
    '''

    def __init__(self, curr_step_ind, dt):
        # Define dt
        self.dt = dt

        if curr_step_ind == 0:
            self.signal_type = 'sinusoidal'
            self.amplitudes = np.array([np.pi/2])
            # self.phase = np.random.random() * np.pi
            self.phases = np.array([np.pi/2])
            self.frequencies = np.array([3]) * 1/(2*np.pi)

            self.reference_map = np.array([[-1, 0],   # yaw angle
                                           [ 0,-1]])  # yaw rate

        elif curr_step_ind == 1:
            self.signal_type = 'sinusoidal'
            self.amplitudes = np.array([np.pi/18, np.pi/14])
            # self.phase = np.random.random() * np.pi
            self.phases = np.array([np.pi/2, np.pi/2])
            self.frequencies = np.array([2, 5]) * 1/(2*np.pi)

            self.reference_map = np.array([[-1, 0, 0, 0],   # yaw angle
                                           [ 0, 0,-1, 0],   # roll angle
                                           [ 0, 0, 0,-1]])  # roll rate

        elif curr_step_ind == 2:
            self.signal_type = 'sinusoidal'
            self.amplitudes = np.array([np.pi/18, np.pi/24, np.pi/14])
            # self.phase = np.random.random() * np.pi
            self.phases = np.array([np.pi/2, np.pi/2, np.pi/2])
            self.frequencies = np.array([2, 3, 5]) * 1/(2*np.pi)

            self.reference_map = np.array([[-1, 0, 0, 0, 0, 0],   # yaw angle
                                           [ 0, 0,-1, 0, 0, 0],   # roll angle
                                           [ 0, 0, 0, 0,-1, 0],   # pitch angle
                                           [ 0, 0, 0, 0, 0,-1]])  # pitch rate

        elif curr_step_ind == 3:
            self.signal_type = 'sinusoidal'
            self.amplitudes = np.array([np.pi/18, np.pi/24, np.pi/24, 1])
            # self.phase = np.random.random() * np.pi
            self.phases = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2])
            self.frequencies = np.array([2, 3, 5, 1]) * 1/(2*np.pi)

            self.reference_map = np.array([[-1, 0, 0, 0, 0, 0, 0, 0],   # yaw angle
                                           [ 0, 0,-1, 0, 0, 0, 0, 0],   # roll angle
                                           [ 0, 0, 0, 0,-1, 0, 0, 0],   # pitch angle
                                           [ 0, 0, 0, 0, 0, 0,-1, 0],   # z-position
                                           [ 0, 0, 0, 0, 0, 0, 0,-1]])  # z-velocity

        elif curr_step_ind == 4:
            self.signal_type = 'sinusoidal'
            self.amplitudes = np.array([np.pi/4, np.pi/24, np.pi/30, 2])
            # self.phase = np.random.random() * np.pi
            self.phases = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2])
            self.frequencies = np.array([6, 8, 11, 0.8]) * 1/(2*np.pi)

            self.reference_map = np.array([[-1, 0, 0, 0, 0, 0, 0, 0],   # yaw angle
                                           [ 0, 0,-1, 0, 0, 0, 0, 0],   # roll angle
                                           [ 0, 0, 0, 0,-1, 0, 0, 0],   # pitch angle
                                           [ 0, 0, 0, 0, 0, 0,-1, 0]])  # z-velocity

        else:
            raise ValueError('curricular step {} not implemented'.format(curr_step_ind+1))

        return

# ---> Safety Filter
class SafetyFilterConfig(object):
    '''
    Configuration class that is given to the safety filter class during the set up of the
    curricular step.
    '''

    def __init__(self, curr_step_ind, dt, action_bounds):

        # Denotes activation of the safety filter (Allows for activating safety filter for certain curricular steps)
        if curr_step_ind == 0:
            self.activated = False

        elif curr_step_ind == 1:
            self.activated = True

        elif curr_step_ind == 2:
            self.activated = True

        elif curr_step_ind == 3:
            self.activated = False

        elif curr_step_ind == 4:
            self.activated = False

        else:
            raise ValueError('Curricular step {} not implemented'.format(curr_step_ind+1))

        # self.activated = False

        # Only define more detailed attributes when the filter is active
        if self.activated:
            # Setting dt
            self.dt = dt

            # Setting action bounds (taken from agent configuration)
            self.action_bounds = np.copy(action_bounds)

            # Plot policy simulations
            self.plot_backup_policy_projection = True

            # Define what SHERPA does when no backup policy is found an no safe projection can be found.
            self.last_resort_action = 'pass-through'  # also 'random'

            # Define whether the policy search is processed sequentially or in parallel
            self.policy_search_processing = 'parallel'  # also "sequential"

            # Iteration parameters
            self.num_iterations_input = 10  # Number of tries to find a backup policy
            self.num_iterations_backup = 10  # Number of policies to be checked
            self.backup_size = 40  # Number of time steps are followed per policy (when using backup policy)
            self.backup_projection_size = np.array([10, 120])

            # Range in which the parameters of a randomly generated policy (in policy generation)
            self.policy_parameter_range = np.array([-10, 10])

            ## Curricular Step 1
            if curr_step_ind == 0:  # --> safety filter is not active

                self.states_closeness_condition = None
                self.RSS = None
                self.FSS = None
                self.sensor_reach = None
                self.v_epsilon = None

            ## Curricular Step 2
            elif curr_step_ind == 1:
                self.states_closeness_condition = np.array([0, 3, 10, 11])  # roll angle and roll rate with according refs
                self.RSS = np.array([0])
                self.RSS_to_ref_mapping = {0:2}

                self.FSS = np.array([[-np.pi/6, np.pi/6],
                                     [0.0, 0.0],
                                     [0.0, 0.0],
                                     [0.0, 0.0],
                                     [0.0, 0.0],
                                     [0.0, 0.0],
                                     [0.0, 0.0],
                                     [0.0, 0.0]])

                self.sensor_reach = np.array([[-np.pi/24, np.pi/24],  # [rad]
                                              [ 0.0, 0.0],
                                              [ 0.0, 0.0],
                                              [ 0.0, 0.0],
                                              [ 0.0, 0.0],
                                              [ 0.0, 0.0],
                                              [ 0.0, 0.0],
                                              [ 0.0, 0.0]])

                # self.v_epsilon = np.array([[-np.pi/12, np.pi/12],  # Roll angle [rad]
                #                            [-np.pi/12, np.pi/12],  # Pitch angle [rad]
                #                            [-np.pi/8 , np.pi/8 ],  # Yaw angle [rad]
                #                            [-np.pi/3, np.pi/3],  # Roll rate [rad/s]
                #                            [-np.pi/6, np.pi/6],  # Pitch rate [rad/s]
                #                            [-np.pi/8 , np.pi/8 ],  # Yaw rate [rad/s]
                #                            [-0.8, 0.8],  # Vertical velocity (w) [m/s]
                #                            [-0.8, 0.8], # Vertical position (z) [m]
                #                            [-np.pi/4, np.pi/4],  # yaw angle ref
                #                            [-np.pi/4, np.pi/4],  # yaw rate ref
                #                            [-np.pi/6, np.pi/6],  # roll angle ref
                #                            [-np.pi/6, np.pi/6]])  # roll rate ref

                self.v_epsilon = np.array([[-np.pi/36, np.pi/36],  # Roll angle [rad]
                                           [0, 0],  # Pitch angle [rad]
                                           [0, 0],  # Yaw angle [rad]
                                           [-np.pi/24, np.pi/24],  # Roll rate [rad/s]
                                           [0, 0],  # Pitch rate [rad/s]
                                           [0, 0],  # Yaw rate [rad/s]
                                           [0, 0],  # Vertical velocity (w) [m/s]
                                           [0, 0],  # Vertical position (z) [m]
                                           [0, 0],  # yaw angle ref
                                           [0, 0],  # yaw rate ref
                                           [-np.pi/24, np.pi/24],  # roll angle ref
                                           [-np.pi/24, np.pi/24]])  # roll rate ref

            ## Curricular Step 3
            elif curr_step_ind == 2:
                self.states_closeness_condition = np.array([0, 1, 3, 4, 10, 11, 12, 13])  # roll, pitch angle and roll, pitch rate
                # self.states_closeness_condition = np.array([1, 4, 12, 13])  # pitch angle and pitch rate
                # self.states_closeness_condition = np.array([1, 4])  # pitch angle and pitch rate
                self.RSS = np.array([0, 1] )
                self.RSS_to_ref_mapping = {0:2, 1:4}

                self.FSS = np.array([[-np.pi/6, np.pi/6],
                                     [-np.pi/6, np.pi/6],
                                     [0.0, 0.0],
                                     [0.0, 0.0],
                                     [0.0, 0.0],
                                     [0.0, 0.0],
                                     [0.0, 0.0],
                                     [0.0, 0.0]])

                self.sensor_reach = np.array([[-np.pi/24, np.pi/24],  # [rad]
                                              [-np.pi/24, np.pi/24],  # [rad]
                                              [ 0.0, 0.0],
                                              [ 0.0, 0.0],
                                              [ 0.0, 0.0],
                                              [ 0.0, 0.0],
                                              [ 0.0, 0.0],
                                              [ 0.0, 0.0]])

                # self.v_epsilon = np.array([[0, 0],  # Roll angle [rad]
                #                            [-np.pi/36, np.pi/36],  # Pitch angle [rad]
                #                            [0, 0],  # Yaw angle [rad]
                #                            [0, 0],  # Roll rate [rad/s]
                #                            [-np.pi/24, np.pi/24],  # Pitch rate [rad/s]
                #                            [0, 0],  # Yaw rate [rad/s]
                #                            [0, 0],  # Vertical velocity (w) [m/s]
                #                            [0, 0],  # Vertical position (z) [m]
                #                            [0, 0],  # yaw angle ref
                #                            [0, 0],  # yaw rate ref
                #                            [0, 0],  # roll angle ref
                #                            [0, 0],  # roll rate ref
                #                            [-np.pi / 24, np.pi / 24],  # pitch angle ref
                #                            [-np.pi / 24, np.pi / 24]])  # pitch rate ref
                #                            # [-np.pi/24, np.pi/24],  # pitch angle ref
                #                            # [-np.pi/24, np.pi/24]]) # pitch rate ref

                self.v_epsilon = np.array([[-np.pi/36, np.pi/36],  # Roll angle [rad]
                                           [-np.pi/36, np.pi/36],  # Pitch angle [rad]
                                           [0, 0],  # Yaw angle [rad]
                                           [-np.pi/24, np.pi/24],  # Roll rate [rad/s]
                                           [-np.pi/24, np.pi/24],  # Pitch rate [rad/s]
                                           [0, 0],  # Yaw rate [rad/s]
                                           [0, 0],  # Vertical velocity (w) [m/s]
                                           [0, 0],  # Vertical position (z) [m]
                                           [0, 0],  # yaw angle ref
                                           [0, 0],  # yaw rate ref
                                           [-np.pi/6, np.pi/6],  # roll angle ref
                                           [-np.pi/3, np.pi/3],  # roll rate ref
                                           [-np.pi/24, np.pi/24],  # pitch angle ref
                                           [-np.pi/24, np.pi/24]]) # pitch rate ref


            elif curr_step_ind == 3:
                self.states_closeness_condition = None
                self.RSS = None
                self.FSS = None
                self.sensor_reach = None
                self.v_epsilon = None

            elif curr_step_ind == 4:
                self.states_closeness_condition = None
                self.RSS = None
                self.FSS = None
                self.sensor_reach = None
                self.v_epsilon = None

            else:
                raise ValueError('Curricular step {} not implemented'.format(curr_step_ind+1))

        return

class SupervisorConfig(object):
    '''
    Configuration class that is given to the excitation signal class during the set up of the
    curricular step.

    $
        K[0, 8] = -1.1784e+00  # Cancel vertical rate (w)
        K[0,11] = -3.1538e-01  # Cancel vertical position (z)

        K[1, 0] = 1.4996e+00  # Cancel roll (phi)
        K[1, 3] = 4.3504e-01  # Cancel roll rate (p)
        K[1, 7] = 3.0715e-01  # Cancel y-velocity (v)
        K[1,10] = 3.0542e-01  # Cancel y-position (y)

        K[2, 1] = 1.5192e+00  # Cancel pitch (theta)
        K[2, 4] = 4.4361e-01  # Cancel pitch rate (q)
        K[2, 6] = -3.0916e-01  # Cancel x-velocity (u)
        K[2, 9] = -3.0549e-01  # Cancel x-position (x)

        K[3, 2] = 9.9374e-02  # Cancel yaw angle (psi)
        K[3, 5] = 1.6062e-01  # Cancel yaw rate (r)
    $
    '''

    def __init__(self, curr_step_ind):

        if curr_step_ind == 0:
            # Define mask for which actions this controller has a say on
            self.action_mask = np.array([True, True, True, False])

            # # Define Gain matrix
            K = np.zeros((3,10))
            K[0,6] = -1.1784e+00  # Cancel vertical rate (w)
            K[0,7] = -3.1538e-01  # Cancel vertical position (z)
            K[1,0] = 1.4996e+00  # Cancel roll (phi)
            K[1,3] = 4.3504e-01  # Cancel roll rate (p)
            K[2,1] = 1.5192e+00  # Cancel pitch (theta)
            K[2,4] = 4.4361e-01  # Cancel pitch rate (q)

        elif curr_step_ind == 1:
            # Define mask for which actions this controller has a say on
            self.action_mask = np.array([True, False, True, False])

            # # Define Gain matrix
            K = np.zeros((2,12))
            K[0,6] = -1.1784e+00  # Cancel vertical rate (w)
            K[0,7] = -3.1538e-01  # Cancel vertical position (z)
            K[1,1] = 1.5192e+00  # Cancel pitch (theta)
            K[1,4] = 4.4361e-01  # Cancel pitch rate (q)

        elif curr_step_ind == 2:
            # Define mask for which actions this controller has a say on
            self.action_mask = np.array([True, False, False, False])

            # Define Gain matrix
            K = np.zeros((1,14))
            K[0,6] = -1.1784e+00  # Cancel vertical rate (w)
            K[0,7] = -3.1538e-01  # Cancel vertical position (z)

        elif curr_step_ind in [3, 4]:
            # Define mask for which actions this controller has a say on
            self.action_mask = np.array([False, False, False, False])

            # Define Gain matrix
            K = np.zeros((0,16))

        else:
            raise ValueError('curricular step {} not implemented'.format(curr_step_ind+1))

        if np.sum(self.action_mask) == K.shape[0]:
            self.K = np.asmatrix(K)
        else:
            raise ValueError('Supervisor Gain matrix does not have the correct size. According to the action mask, it '+
                             'controls {} actions, however, the gain matrix has {} rows.'.format(np.sum(self.action_mask),
                                                                                                        K.shape[0]))

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

        # Assign curricular step index for model creation
        self.curr_step_ind = curr_step_ind

        # Quadrotor housing dimensions
        self.body_h = 0.10  # [m]
        self.body_w = 0.10  # [m]
        self.body_l = 0.10  # [m]
        self.body_m = 0.4  # [kg]

        # Propeller arm lengths (from geometric center)
        arm_l = 0.3
        self.prop1_arm_l = arm_l  # [m]
        self.prop2_arm_l = arm_l  # [m]
        self.prop3_arm_l = arm_l  # [m]
        self.prop4_arm_l = arm_l  # [m]

        # Propeller arm angle with the x-a xis (Vehicle carried frame of reference)
        self.prop1_arm_alpha =  1/4 * np.pi  # [rad]
        self.prop2_arm_alpha =  3/4 * np.pi  # [rad]
        self.prop3_arm_alpha = -3/4 * np.pi  # [rad]
        self.prop4_arm_alpha = -1/4 * np.pi  # [rad]

        # Propeller arm masses
        arm_mass = 0.1
        self.prop1_arm_m = arm_mass  # [kg]
        self.prop2_arm_m = arm_mass  # [kg]
        self.prop3_arm_m = arm_mass  # [kg]
        self.prop4_arm_m = arm_mass  # [kg]

        # Propeller motor masses
        motor_mass = 0.2
        self.prop1_motor_m = motor_mass  # [kg]
        self.prop2_motor_m = motor_mass  # [kg]
        self.prop3_motor_m = motor_mass  # [kg]
        self.prop4_motor_m = motor_mass  # [kg]

        # Gravitational acceleration
        self.g = 9.80665
        self.b = 4.08732e-3*0.5  # e-3
        self.d = 9.2964264e-5*3  # e-5

        # Compute xi, the force/torque -> RPM mapping matrix
        l1_y = np.sin(self.prop1_arm_alpha) * self.prop1_arm_l
        l1_x = np.cos(self.prop1_arm_alpha) * self.prop1_arm_l

        l2_y = np.sin(self.prop2_arm_alpha) * self.prop2_arm_l
        l2_x = np.cos(self.prop2_arm_alpha) * self.prop2_arm_l

        l3_y = np.sin(self.prop3_arm_alpha) * self.prop3_arm_l
        l3_x = np.cos(self.prop3_arm_alpha) * self.prop3_arm_l

        l4_y = np.sin(self.prop4_arm_alpha) * self.prop4_arm_l
        l4_x = np.cos(self.prop4_arm_alpha) * self.prop4_arm_l

        self.xi = np.array([[self.b, self.b, self.b, self.b],
                            [-self.b*l1_y, -self.b*l2_y, -self.b*l3_y, -self.b*l4_y],
                            [self.b*l1_x, self.b*l2_x, self.b*l3_x, self.b*l4_x],
                            [self.d, -self.d, self.d, -self.d]])

        # Compute RPM for hovering (trim point of the model)
        m = 4 * motor_mass + 4 * arm_mass + self.body_m
        self.hover_RPM = (m * self.g)/self.b

        # Define initial conditions
        self.init_bounds = np.array([[-np.pi/36, np.pi/36],  # [rad]   --> pitch
                                     [-np.pi/36, np.pi/36],  # [rad]   --> roll
                                     [-np.pi/36, np.pi/36],  # [rad]   --> yaw
                                     [-np.pi/24, np.pi/24],  # [rad/s] --> pitch rate
                                     [-np.pi/24, np.pi/24],  # [rad/s] --> roll rate
                                     [-np.pi/24, np.pi/24],  # [rad/s] --> yaw rate
                                     [-0.15, 0.15],          # [m/s]   --> z-velocity
                                     [-0.20, 0.20]])         # [m]     --> z-position

        # Yaw angle control
        if curr_step_ind == 0:
            # Define C matrix
            self.C = np.zeros((2, 8))
            self.C[0, 2] = 1  # --> yaw angle
            self.C[1, 5] = 1  # --> yaw rate

        # Roll angle control
        elif curr_step_ind == 1:
            # Define C matrix
            self.C = np.zeros((3, 8))
            self.C[0,2] = 1  # --> yaw angle
            self.C[1,0] = 1  # --> roll angle
            self.C[2,3] = 1  # --> roll rate

        # Pitch angle control
        elif curr_step_ind == 2:
            # Define C matrix
            self.C = np.zeros((4, 8))
            self.C[0,2] = 1  # --> yaw angle
            self.C[1,0] = 1  # --> roll angle
            self.C[2,1] = 1  # --> pitch angle
            self.C[3,4] = 1  # --> pitch rate

        # Altitude control
        elif curr_step_ind == 3:
            # Define C matrix
            self.C = np.zeros((5, 8))
            self.C[0,2] = 1  # --> yaw angle
            self.C[1,0] = 1  # --> roll angle
            self.C[2,1] = 1  # --> pitch angle
            self.C[3,7] = 1  # --> z-position
            self.C[4,6] = 1  # --> z-velocity

        # Rotor RPM control
        elif curr_step_ind == 4:
            # Define C matrix
            self.C = np.zeros((4, 8))
            self.C[0,2] = 1  # --> yaw angle
            self.C[1,0] = 1  # --> roll angle
            self.C[2,1] = 1  # --> pitch angle
            self.C[3,7] = 1  # --> z-position

        else:
            raise ValueError('curricular step {} not implemented'.format(curr_step_ind+1))

        return

class SystemIDConfig(object):
    '''
    Configuration class that is given to the system ID class during the set up of the
    curricular step.
    '''

    def __init__(self, curr_step_ind):
        # Boolean whether to use sliding window principle or not
        self.use_sliding_window = True

        # Define sample factor for each curricular step
        if curr_step_ind == 0:
            self.sample_factor = 2

        elif curr_step_ind == 1:
            self.sample_factor = 2

        elif curr_step_ind == 2:
            self.sample_factor = 4

        elif curr_step_ind == 3:
            self.sample_factor = 2

        elif curr_step_ind == 4:
            self.sample_factor = 2

        else:
            raise ValueError('curricular step {} not implemented'.format(curr_step_ind))

        return

class TransferLearningConfig(object):
    '''
    Configuration class that is given to the transfer learning class during the set up of the
    curricular step.
    '''

    def __init__(self, curr_step_ind, xi):
        self.curr_step_ind = curr_step_ind

        # Boolean denoting the use of the supervisor policy as initial value for agent policy
        self.use_supervisor_policy_as_basis = True

        # Transformation matrix between rotor RPM inputs and dimension-wise inputs
        self.xi = np.asmatrix(xi)

        return
