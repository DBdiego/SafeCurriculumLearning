import numpy as np
import warnings


class AgentEnvironmentInteraction(object):
    '''
    This class formalises an instance of a Reinforcement Learning agent-environment interaction.
    The agent learns a policy to control the system safely during this process. The task implemented is
    a Linear Quadratic Tracking task.

    Below, the meaning of certain attributes will be provided:

    :attr times: 1-D array of times
    :attr agent: Object of an agent
    :attr env: Obejct of an environment
    :attr excitation_signals: excitement signal for each input
    :attr num_samples: number os samples used in each learning epoch
    :attr safety_filter: class of safety filter to be applied on the agent's action
    :attr system_id:
    :attr model:
    :attr train_agent: Boolean stating whether or not to train the agent
    '''

    def __init__(self):
        # Contains all time steps
        self.times = None

        # Instance of system
        self.system = None

        # Instance of the agent
        self.agent = None

        # Instance of the environment
        self.env = None

        # Time series of excitation signals
        self.excitation_signal = None

        # Instance of the safety filter
        self.safety_filter = None

        # Instance of the system identifier
        self.system_identifier = None

        # Instance of the supervisor controller
        self.supervisor = None

        # Instance of the model used in system identification and safety filter
        self.model = None

        # Instance of reference signal
        self.reference_signal = None

        # Instance of measurement noise
        self.measurement_noise = None

        # Initiate attribute for initial conditions
        self.initial_conditions = None

        self.tran_agent = None      ###########---------> make part of the agent???

        # Arrays that store some of the variables of interest during the simulation
        self.state_storage = None
        self.measured_state_storage = None
        self.action_storage = None
        self.excite_storage = None
        self.cost_storage = None
        self.RMSE_storage = None
        self.real_closed_loop_eigs = None

        # Boolean describing whether a stablising policy was found
        self.stable_policy_found = False

        # Additional attributes for output generation
        self.name = None
        self.folder_name = None
        self.ind = None

    def run_checks(self):
        '''
        This function runs the necessary checks to warn the user of infeasible or undesired combinations
        of attributes an values between the different modules.

        :return: None
        '''

        # Check whether system identification updates more frequently than agent policy
        if self.system_identifier is not None:
            if self.agent.num_samples <= self.system_identifier.num_samples:
                warnings.warn('System identifier has more samples than the agent.')

        return

    def run_interaction(self):
        '''
        This function simulates the agent in the environment. The agent learns during this
        process.

        :return: output_data, all data collected during the simulation
        '''

        # Run checks
        self.run_checks()

        # Create variable that will store KPI's
        self.state_storage = np.zeros((self.env.n+self.env.p, self.times.shape[0]))
        self.measured_state_storage = np.zeros((self.env.n + self.env.p, self.times.shape[0]))
        self.action_storage = np.zeros((self.env.m, self.times.shape[0]))
        self.excite_storage = np.zeros((self.env.m, self.times.shape[0]))
        self.cost_storage = np.zeros(self.times.shape)
        self.RMSE_storage = np.zeros(self.times.shape)
        self.real_closed_loop_eigs = np.empty((self.env.n, self.times.shape[0]))
        self.system_id_uncertainties = np.zeros((0,(self.env.n+self.env.p)*(self.env.n+self.env.m+self.env.p)))

        # Construct the policy parameters of the system by combining the supervisor and agent policies
        policy_params = self.construct_agent_supervisor_policy_params()

        # Appending initial closed loop eigenvalues
        CL_eigs = self.env.closed_loop_eigs(policy_params)
        # self.real_closed_loop_eigs = np.hstack((self.real_closed_loop_eigs, CL_eigs.reshape((self.env.n, 1))))

        # Epoch numbering variables
        num_epochs = int(np.ceil(self.times.shape[0] / self.agent.num_samples)) - 1

        # Generate initial conditions
        self.initial_conditions = self.system.get_initial_conditions()
        X_k = np.vstack((self.initial_conditions,
                         self.reference_signal.x_init))

        # Generate excitation signal time series
        excitation_signals = self.excitation_signal.generate_excitation_time_series(self.env.m, self.times)
        self.excitation_signals = excitation_signals

        # Simulation of Agent-Environment Interaction
        j = 1
        print('\t\tAgent-Environment Interaction:')
        for k, t in enumerate(self.times):

            # Adding measurement noise to state (if desired)
            if self.measurement_noise is not None:
                X_k_measured = self.measurement_noise.add_measurement_noise(X_k, self.env.n, k)
            else:
                X_k_measured = np.asmatrix(np.copy(X_k))

            # Getting excitement signal
            excite_sig = excitation_signals[:,k:(k+1)] * self.train_agent

            # Determine agent action based on measured state
            u_k_agent = self.agent.policy(X_k_measured, self.agent.policy_parameters)
            # u_k_agent = self.agent.policy(X_k, self.agent.policy_parameters)

            # Add excitation signal to agent action
            u_k_agent_excite = u_k_agent + np.asmatrix(excite_sig)[self.agent.action_mask]

            # Update model used in safety filter and pass action through safety filter, if any
            if self.safety_filter is not None:

                # Update safety filter model and policy parameters
                if self.system_identifier.SS_estimate_available:
                    self.safety_filter.bm = self.model
                    agent_policy = np.copy(self.agent.policy_parameters)
                    self.safety_filter.policy_parameters = np.asmatrix(agent_policy) #self.agent.policy_parameters

                # Pass action through filter
                # u_k_excite = self.safety_filter.validate_action(X_k_measured, u_k_excite, k)
                u_k_agent_excite = self.safety_filter.validate_action(X_k_measured, u_k_agent_excite, k)

            # Assemble supervisor and agent actions
            if self.supervisor is not None:
                # Compute action taken by the supervisor
                u_k_supervisor = self.supervisor.supervisor_action(X_k_measured)
                # u_k_supervisor = self.supervisor.supervisor_action(X_k)

                # Add excitation signal to supervisor action
                u_k_supervisor_excite = u_k_supervisor + np.asmatrix(excite_sig)[self.supervisor.action_mask]

                # Adapt the desired action elements
                u_k_excite = np.asmatrix(np.zeros((self.env.m, 1)))
                u_k_excite[self.supervisor.action_mask] = u_k_supervisor_excite
                u_k_excite[self.agent.action_mask] = u_k_agent_excite

            else:
                u_k_excite = np.asmatrix(np.copy(u_k_agent_excite))

            # Derive u_k from u_k_excite
            u_k = u_k_excite - excite_sig

            # Check for input saturation
            u_k_excite = self.agent.action_saturation(u_k_excite)
            u_k = self.agent.action_saturation(u_k)

            # # Add excitement signal to input
            # u_k_excite = u_k + excite_sig
            #
            # # Saturate excited input
            # u_k_excite = self.agent.action_saturation(u_k_excite)

            # Computing cost at time k for selected state-action pair
            cost_k = self.env.calculate_cost(X_k_measured, u_k_excite[self.agent.action_mask])

            ## Collect samples for Q-Learning and System Identification
            self.agent.collect_samples(X_k_measured,
                                       u_k[self.agent.action_mask],
                                       u_k_excite[self.agent.action_mask],
                                       cost_k)

            if self.system_identifier is not None:
                self.system_identifier.collect_samples(X_k_measured, u_k, u_k_excite)

            # Calculating RMSE
            RMSE = self.env.calculate_RMSE(X_k)
            self.RMSE_storage[k] = RMSE

            # Store values of Z, u, and cost at time step k
            self.excite_storage[:,k:k+1] = excite_sig
            self.state_storage[:,k:k+1] = np.copy(X_k_measured) #np.copy(X_k)
            self.measured_state_storage[:,k:k+1] = np.copy(X_k_measured)
            self.action_storage[:,k:k+1] = np.copy(u_k_excite)
            self.cost_storage[k] = cost_k
            self.real_closed_loop_eigs[:,k:k+1] = CL_eigs.reshape((self.env.n, 1))

            # System Identification update and Q-learning kernel update
            if k > 0 and self.train_agent:
                # System Identification
                if self.system_identifier is not None:
                    condition_1 = k % self.system_identifier.num_samples == 0
                    condition_2 = (k > self.system_identifier.num_samples and self.system_identifier.use_sliding_window)
                    if condition_1 or condition_2:
                        # Perform system identification
                        self.system_identifier.identify_system(method='OLS',
                                                               uncertainty_method='uniform',
                                                               uncertainty_type='absolute')

                        # Provide estimates and their uncertainties to the model instance
                        self.model.T_d, self.model.B1_d = [self.system_identifier.T, self.system_identifier.B1]
                        self.model.unc_T, self.model.unc_B1 = [self.system_identifier.T_unc, self.system_identifier.B1_unc]

                # Q-learning
                if k % self.agent.num_samples == 0:
                    print('\t\t\t Epoch {:2d}/{}'.format(j, num_epochs))

                    # Policy evaluation
                    valid_policy = self.agent.evaluate_policy()

                    # Policy Improvement (if valid policy evaluation step)
                    if valid_policy:
                        self.agent.policy_parameters = self.agent.improve_policy()
                    else:
                        break

                    print(self.agent.policy_parameters)

                    # New closed loop eigenvalues (real part)
                    policy_params = self.construct_agent_supervisor_policy_params()
                    CL_eigs = self.env.closed_loop_eigs(policy_params)

                    # # Saving new CL-eigenvalues
                    # self.real_closed_loop_eigs = np.hstack((self.real_closed_loop_eigs,
                    #                                         CL_eigs.reshape((self.env.n, 1))))

                    # Increase j
                    j += 1

            # Propagate state
            X_k = self.env.state_transition_discrete(X_k, u_k_excite)

        # Verify policy stability with closed loop eigenvalues
        if np.all(self.real_closed_loop_eigs[:, -1] <= 0):
            self.stable_policy_found = True

        return

    def construct_agent_supervisor_policy_params(self):
        '''
        Combines the policy parameters of the supervisor and agent into a single gain.
        Mapping of the respective policies is based on the action masks of the supervisor and agent.

        :return: K, the new policy parameters (formulated as a gain for linear systems)
        '''

        # Create policy parameter array
        K = np.zeros((self.env.m, self.env.n+self.env.p))

        # Assign the policy parameters of agent
        K[self.agent.action_mask,:] = self.agent.policy_parameters

        if self.supervisor is not None:
            K[self.supervisor.action_mask,:] = self.supervisor.K

        return K

    def generate_output(self, configuration):
        '''
        Generates all the output files (plots, txt files, etc) for each module that is active.

        :param configuration: Simulation configuration (containing all configs for each module)
        :return: None
        '''

        # Generate output from system class
        print('\t\t\tGenerating System Outputs')
        self.system.generate_output(self, configuration.system_config)

        # Generate output from agent class
        print('\t\t\tGenerating Agent Outputs')
        self.agent.generate_output(self, configuration.agent_config)

        # Generate output from environment class
        print('\t\t\tGenerating Environment Outputs')
        self.env.generate_output(self, configuration.env_config)

        # Generate output from safety filter class
        if self.safety_filter is not None:
            print('\t\t\tGenerating Safety Filter Outputs')
            self.safety_filter.generate_output(self, configuration.safety_filter_config)

        # Generate output from system identifier class
        if self.system_identifier is not None:
            print('\t\t\tGenerating System Identification Outputs')
            self.system_identifier.generate_output(self, configuration.system_id_config)

        # Save configuration
        self.save_configuration(configuration)

        return

    def save_configuration(self, configuration):
        '''
        Constructs a text version of the configuration class.

        :param configuration: Simulation configuration (containing all configs for each module)
        :return: None
        '''

        str2write = ''
        str2write += '*****************\n'
        str2write += '* Configuration *\n'
        str2write += '*****************\n\n'

        # General
        str2write += 'Simulation time: {:.2f} [s]\n'.format(configuration.duration)
        str2write += 'Simulation step size (dt): {:.1e} [s]\n'.format(configuration.dt)
        str2write += '\n'
        str2write += '\n'

        # Agent
        str2write += 'Agent Configuration\n'
        str2write += '-------------------\n'
        str2write += '  Learning rate: {:.2f}\n'.format(configuration.agent_config.learning_rate)
        str2write += '  Sample factor: {:d}\n'.format(configuration.agent_config.sample_factor)
        str2write += '  Action bounds: \n'
        str2write += '       min: {}\n'.format(configuration.agent_config.action_bounds[:,0])
        str2write += '       max: {}\n'.format(configuration.agent_config.action_bounds[:,1])
        str2write += '  Learning method: {}\n'.format(configuration.agent_config.method)
        str2write += '\n'
        str2write += '\n'

        # Environment
        str2write += 'Environment Configuration\n'
        str2write += '-------------------------\n'
        str2write += '  Run system checks: {}\n'.format(configuration.env_config.run_system_checks*'on' +
                                                        (not configuration.env_config.run_system_checks)*'off')
        str2write += '  Q diagonal values: {}\n'.format(configuration.env_config.Q_diag_values)
        str2write += '  R diagonal values: {}\n'.format(configuration.env_config.R_diag_value)
        str2write += '  RMSE limit: {}\n'.format(configuration.env_config.RMSE_limit)
        str2write += '\n'
        str2write += '\n'

        # Excitation signal
        str2write += 'Excitation Signal Configuration\n'
        str2write += '-------------------------------\n'
        str2write += '  Signal Type: {} \n'.format(configuration.excitation_signal_config.signal_type)
        if configuration.excitation_signal_config.signal_type == 'normal':
            str2write += '  Means: {} \n'.format(configuration.excitation_signal_config.means)
            str2write += '  Standard Deviations: {} \n'.format(configuration.excitation_signal_config.standard_deviations)
        elif configuration.excitation_signal_config.signal_type == 'frequency_limited':
            str2write += '  Means: {} \n'.format(configuration.excitation_signal_config.means)
            str2write += '  Amplitudes: {} \n'.format(configuration.excitation_signal_config.amplitudes)
            str2write += '  Frequency ranges [rad/s]: \n'
            str2write += '       min: {}\n'.format(configuration.excitation_signal_config.frequency_ranges[:,0])
            str2write += '       max: {}\n'.format(configuration.excitation_signal_config.frequency_ranges[:,1])
        str2write += '\n'
        str2write += '\n'

        # Measurement Noise
        str2write += 'Measurement Noise Configuration\n'
        str2write += '-------------------------------\n'
        if self.measurement_noise is not None:
            str2write += '  Mean: {} [m or m/s]\n'.format(configuration.measurement_noise_config.mean)
            str2write += '  Standard Deviation: {} [m or m/s]\n'.format(configuration.measurement_noise_config.standard_deviation)
        else:
            str2write += '  No measurement noise used\n'

        str2write += '\n'
        str2write += '\n'

        # Reference signal
        str2write += 'Reference Signal Configuration\n'
        str2write += '------------------------------\n'
        str2write += '  Type: {}\n'.format(configuration.reference_signal_config.signal_type)
        str2write += '  Amplitudes: {} [m]\n'.format(configuration.reference_signal_config.amplitudes)
        str2write += '  Phases: {} [rad]\n'.format(configuration.reference_signal_config.phases)
        str2write += '  Frequencies: {} [Hz]\n'.format(configuration.reference_signal_config.frequencies)
        str2write += '\n'
        str2write += '\n'

        # Safety filter
        str2write += 'Safety Filter Configuration\n'
        str2write += '---------------------------\n'
        if self.safety_filter is not None:
            str2write += '  Number of tries for finding a backup policy: {}\n'.format(
                configuration.safety_filter_config.num_iterations_input)
            str2write += '  Number of policies checked in each try: {}\n'.format(
                configuration.safety_filter_config.num_iterations_backup)
            str2write += '  Number of time steps projected with each policy: {}\n'.format(
                configuration.safety_filter_config.backup_size)
            str2write += '\n'
            str2write += '  Time step size (dt): {} [s]\n'.format(configuration.safety_filter_config.dt)
            str2write += '  Action bounds: \n'
            for i in range(configuration.safety_filter_config.action_bounds.shape[0]):
                str2write += '       {}\n'.format(configuration.safety_filter_config.action_bounds[i,:])
            str2write += '\n'
            str2write += '  Sensor reach: \n'
            for i in range(configuration.safety_filter_config.sensor_reach.shape[0]):
                str2write += '       {}\n'.format(configuration.safety_filter_config.sensor_reach[i,:])
            str2write += '\n'
            str2write += '  RSS: {}\n'.format(configuration.safety_filter_config.RSS)
            str2write += '  FSS bounds:\n'
            for i in range(configuration.safety_filter_config.FSS.shape[0]):
                str2write += '       {}\n'.format(configuration.safety_filter_config.FSS[i,:])
            str2write += '\n'
            str2write += '  Closeness condition:\n'
            for i in range(configuration.safety_filter_config.v_epsilon.shape[0]):
                str2write += '       {}\n'.format(configuration.safety_filter_config.v_epsilon[i,:])
            str2write += '\n'
            on_off_bool = configuration.safety_filter_config.plot_backup_policy_projection
            str2write += '  Plot backup policy projections when used: {}\n'.format(on_off_bool*'on'+(not on_off_bool)*'off')
        else:
            str2write += '  No safety filter used\n'
        str2write += '\n'
        str2write += '\n'

        # System Identification
        str2write += 'System Identification Configuration\n'
        str2write += '-----------------------------------\n'
        if self.system_identifier is not None:
            str2write += '  Sample factor: {}\n'.format(configuration.system_id_config.sample_factor)
        else:
            str2write += '  No system identification used\n'

        # Save text to file
        f = open(self.folder_name +'/' + 'Configuration.txt', 'w')
        f.write(str2write)
        f.close()

        return