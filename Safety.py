import os
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import Interval_arithmetic as itrv_arith
from matplotlib.patches import Polygon, Rectangle


class SHERPA(object):
    """
    Class containing all the functionalities of the SHERPA algorithm presented in:

        T. Mannucci, E.-J. Van Kampen, C. De Visser, and Q. Chu. Safe Exploration Algorithms
        for Reinforcement Learning Controllers. IEEE Transactions on Neural Networks and Learning
        Systems, 29(4):1069–1081, 2018. doi: 10.1109/TNNLS.2017.2654539.


    The algorithm is aware of the state and input dimensions (m,n).
    """

    def __init__(self, interaction_dims, configuration, agent_class, output_folder_directory):

        # Set dt
        self.dt = configuration.dt

        # Range of parameters in a randomly generated policy
        self.policy_parameter_range = configuration.policy_parameter_range

        # Action taken when next step is not safe and no backup policy is found (either 'pass-through' or 'random')
        self.last_resort_action = configuration.last_resort_action

        # Which states to check for closeness condition
        self.modality_states = configuration.states_closeness_condition

        # The minimum and maximum projection time frame when searching for a backup policy
        self.backup_projection_size = configuration.backup_projection_size

        # mapping between the RSS and the reference state vector (for backup policy projection plotting)
        self.RSS_to_ref_mapping = configuration.RSS_to_ref_mapping

        # Whether to plot the simulations of all the policies
        self.plot_backup_policy = configuration.plot_backup_policy_projection

        # String defining the processing type of the policy search (can be 'parallel' or 'sequential')
        self.policy_search_processing = configuration.policy_search_processing

        # Output folder of curricular SHERPA
        if self.plot_backup_policy:
            self.output_folder = output_folder_directory + '/SHERPA'
            os.mkdir(self.output_folder)

        # State and input dimensions
        [self.n, self.m, self.p] = [agent_class.n, agent_class.m, agent_class.p]

        # Assign a bounding model to the SHERPA class
        self.bm = None

        # Definition of the state history database
        self.state_history = np.zeros((self.n+self.p,0))
        self.SSS_storage = np.zeros((int(2*(self.n+self.p)), 0))

        # Storage for time steps with backup policy and random actions
        self.action_suggestion_steps = []
        self.backup_policy_steps = []
        self.random_acitons_steps = []

        # Definition of safety related state spaces
        self.RSS = configuration.RSS
        self.FSS = configuration.FSS  # np.asmatrix(configuration.FSS)
        self.SSS = np.asmatrix(np.zeros((self.n+self.p, 2)))

        # Define variable containing the hypersphere limits
        self.sensor_reach = configuration.sensor_reach
        self.sensor_reach = np.asmatrix(self.sensor_reach)

        # Define closeness condition epsilon
        self.v_epsilon = configuration.v_epsilon
        self.v_epsilon = np.asmatrix(self.v_epsilon)

        # Compute epsilon interval size
        self.v_epsilon_itrv_size = np.asmatrix(np.diff(self.v_epsilon))[self.modality_states,:]

        # Define initial policy function and policy parameters to be None
        self.policy = None
        self.policy_parameters = None

        # Define variable counting the remaining steps to be taken by the backup policy
        self.backup_steps_remaining = None

        # Define backup sequence
        self.backup_policy_parameters = None
        self.backup_policy_final_state = None
        self.arrival_backup_policy_parameters = None
        self.arrival_backup_policy_final_state = None

        # Booleans
        self.backup_taken = False
        self.backup_arrival_found = False
        self.input_action_approved = False

        # # Set the limit number of iterations to find a backup (What is the difference between them?)
        self.num_iterations_input = configuration.num_iterations_input
        self.num_iterations_backup = configuration.num_iterations_backup

        # Define backup band and size (number of time steps projected by each backup policy)
        self.backup_size = configuration.backup_size
        self.arrival_backup_size = configuration.backup_size

        # Define the input saturation limits
        self.action_bounds = np.asmatrix(configuration.action_bounds)[agent_class.action_mask,:]

        # Defining policy function
        self.policy = agent_class.uncertain_policy

        # Define agent action mask
        self.agent_action_mask = agent_class.action_mask

        # Define number of CPU's allocated to SHERPA's internal policy projections
        self.num_cpus = int(1.0 * mp.cpu_count())  # 4

        self.policy_size = agent_class.m * (agent_class.n + agent_class.p)

        self.backup_policy_history = np.zeros((self.policy_size,0))
        self.backup_policy_history_list = []

        self.initial_agent_policy_params = None
        self.initial_backup_size = self.backup_size

        ## !!!!!!!!!!!!!!!! FOR DEBUGGING !!!!!!!!!!!!!!
        self.actual_state_evolution = []
        self.projected_state_evolution = []

        return

    def action_saturation(self, action):
        '''
        Saturates the SHERPA's action based on given bounds and agent's action filter (when supervisor is present)

        :param u_k: desired action of the agent
        :return: saturated action of the agent
        '''

        # Check for max force saturation:
        action = np.minimum(action, self.action_bounds[:,1])

        # Check for min force saturation:
        action = np.maximum(action, self.action_bounds[:,0])

        return action

########################################################################################################################
    def validate_action(self, input_state, input_action, k):
        '''
        Function validating (or not) the action selected by the agent at the current state.

        :param input_state: Augmented state at which action was selected
        :param input_action: selected action by agent
        :return: original action if valid, new action if not valid
        '''

        if self.policy_parameters is not None and self.initial_agent_policy_params is None:
            self.initial_agent_policy_params = self.policy_parameters

        # Reset the action approval boolean
        self.input_action_approved = False

        # Set initial backup policy parameters as the ones of the current agent params
        if (self.backup_policy_parameters is None) and (self.policy_parameters is not None) :
            print('\t\t\t SHERPA: taking agent policy parameters as initial backup')
            self.backup_policy_parameters = self.policy_parameters
            self.arrival_backup_policy_parameters = self.policy_parameters

        # Saturate input action
        input_action = self.action_saturation(input_action)

        # Create output action variable
        output_action = np.asmatrix(np.zeros(input_action.shape))

        # Create interval formulations of state and action
        interval_action = np.hstack((input_action, input_action))
        interval_state = np.hstack((input_state, input_state))

        # Assess presence of risk at this state
        reach, risk = self.assess_risk_of_state(input_state)

        # If no risk is associated: Update Safe State Space Boundaries and State History
        if not risk:
            # Update state history
            self.state_history = np.hstack((self.state_history, input_state))

            # Update Safe State Space (SSS) bounds
            if k == 0:
                self.SSS[self.RSS,0] = np.asarray(reach)[:,0]  # lower SSS bound
                self.SSS[self.RSS,1] = np.asarray(reach)[:,1]  # upper SSS bound
            else:
                SSS_lower_bound = np.minimum(self.SSS[self.RSS,0], reach[:,0])
                SSS_upper_bound = np.maximum(self.SSS[self.RSS,1], reach[:,1])

                self.SSS[self.RSS,0] = np.asarray(SSS_lower_bound)[:,0]
                self.SSS[self.RSS,1] = np.asarray(SSS_upper_bound)[:,0]

        # Store current SSS
        SSS_vector = np.vstack((self.SSS[:,0], self.SSS[:,1]))
        self.SSS_storage = np.hstack((self.SSS_storage, SSS_vector))

        # Bounding model is None until an estimate from system identification is available
        if self.bm is not None:

            # Only take necessary columns from bm.B1_d and bm.unc_B1 (related to inputs controlled by the agent)
            self.bm.B1_d = self.bm.B1_d[:,self.agent_action_mask]
            self.bm.unc_B1 = self.bm.unc_B1[:,self.agent_action_mask]

            # Do this when following backup policy
            if self.backup_taken:
                # !!!!!!!!!!!!!!!!!!! DEBUGGING PURPOSES !!!!!!!!!!!!!!!!!
                self.actual_state_evolution.append(input_state)

                # Save the time step at which the backup policy is started (for plotting purposes)
                self.backup_policy_steps.append(k)

                # Decreasing the amount of backup steps to be taken by 1
                self.backup_steps_remaining -= 1

                # Whilst executing backup policy, try to find a backup policy for the arrival state
                if not self.backup_arrival_found:
                    self.backup_arrival_found, \
                    arrival_backup_policy_parameters, \
                    arrival_backup_policy_final_state,\
                    arrival_backup_size = self.find_backup_policy(self.arrival_backup_policy_final_state,
                                                                  self.backup_policy_parameters,
                                                                  0)

                    # Store backup if found
                    if self.backup_arrival_found:
                        print('\t\t\t SHERPA: k = {:3d} - Found backup policy for arrival state'.format(k))
                        self.arrival_backup_policy_parameters = arrival_backup_policy_parameters
                        self.arrival_backup_policy_final_state = arrival_backup_policy_final_state
                        self.arrival_backup_size = arrival_backup_size

                # Compute first action based on policy with backup policy parameters
                backup_interval_action = self.policy(interval_state, self.backup_policy_parameters)

                # # Saturate backup interval action
                backup_interval_action[:,0] = self.action_saturation(backup_interval_action[:,0])
                backup_interval_action[:,1] = self.action_saturation(backup_interval_action[:,1])

                # Define the action as the middle of the interval formulation (by using the mean)
                backup_action = np.mean(backup_interval_action, axis=1)
                backup_action = np.asmatrix(backup_action)

                # self.action_saturation(backup_action)

                # Assign the backup action to the output action variable
                output_action = np.asmatrix(backup_action)

                if self.backup_steps_remaining == 0:
                    # When no more backup steps are required, shift to backup policy of arrival state
                    self.backup_taken = False

                    if self.backup_arrival_found:
                        self.backup_policy_parameters = self.arrival_backup_policy_parameters
                        self.backup_policy_final_state = self.arrival_backup_policy_final_state
                        self.backup_size = self.arrival_backup_size
                        self.backup_arrival_found = False

                    else:
                        print('\t\t\t SHERPA: k = {:3d} - No backup policy found for arrival state'.format(k))
                        self.backup_size = self.initial_backup_size
                        backup_policy_final_state = np.asmatrix(np.copy(interval_state))

                        for i in range(self.backup_size):
                            backup_policy_final_state = self.simulate_agent_in_uncertain_environment(backup_policy_final_state,
                                                                                                     self.initial_agent_policy_params)

                        self.backup_policy_parameters = self.initial_agent_policy_params
                        self.backup_policy_final_state = backup_policy_final_state

                    self.actual_state_evolution = []

            # If not following a backup policy, find one for this state
            else:
                # Convert model to interval format
                self.bm.create_interval_formulation(unc_type='absolute')

                iteration = 0
                next_step_is_safe = False
                backup_found = False
                safe_status_next_state = []
                while iteration <= self.num_iterations_input and not backup_found:
                    # 1) Propagate state to next time step based input proposed by agent or random action
                    new_interval_state = self.bm.propagate_state(interval_state, interval_action)

                    # 2) Verify that propagated state is in known SSS
                    next_step_is_safe = self.check_interval_state_safety(new_interval_state)

                    # 3) Search for a backup policy from this propagated state
                    if next_step_is_safe:
                        # Check whether propagated state does not belong to LTF states by finding a backup
                        backup_found, \
                        backup_policy_parameters, \
                        backup_policy_final_state,\
                        backup_size = self.find_backup_policy(new_interval_state,
                                                              self.policy_parameters,
                                                              iteration)

                        # Store backup
                        if backup_found:
                            self.backup_policy_parameters = backup_policy_parameters
                            self.backup_policy_final_state = backup_policy_final_state
                            self.backup_size = backup_size

                    # 4) If the propagated state is not in the known SSS or no backup was found, propose another action
                    if not (next_step_is_safe or backup_found):
                        action = self.generate_random_action()
                        interval_action = np.hstack((action, action))

                        # # in case of a last loop, the output action will be random
                        output_action = action

                    # Collect safety status of all the next states tried during iterations
                    safe_status_next_state.append(next_step_is_safe)

                    # Increase iteration index
                    iteration += 1

                # Perform action
                if next_step_is_safe and backup_found:
                    if iteration == 1:
                        self.input_action_approved = True
                    else:
                        print('\t\t\t SHERPA: k = {:3d} - New action suggested (iteration: {})'.format(k, iteration))
                        self.action_suggestion_steps.append(k)

                    # !!!!!!!!!!!!!!!!!!!!! DEBUGGING PURPOSES ONLY !!!!!!!!!!!!!!!!!!!!!!!!!
                    self.projected_next_state = new_interval_state

                    # Define the action that is suggested by the filter
                    output_action = np.mean(interval_action, axis=1)
                    output_action = np.asmatrix(output_action)

                # Take backup
                elif next_step_is_safe and not backup_found:

                    print('\t\t\t SHERPA: k = {:3d} - Using Backup Policy (for {} time steps)'.format(k,self.backup_size))
                    print(self.backup_policy_parameters)

                    # Save the time step at which the backup policy is started (for plotting purposes)
                    self.backup_policy_steps.append(k)

                    # Based on current backup policy, find arrival estimate
                    state_history = [interval_state]
                    backup_policy_final_state = np.asmatrix(np.copy(interval_state))

                    for i in range(self.backup_size):

                        # !!!!!!!!!!! DEBUGGING PURPOSES !!!!!!!!!!!!!!!
                        self.projected_state_evolution.append(np.mean(backup_policy_final_state, axis=1))

                        backup_policy_final_state = self.simulate_agent_in_uncertain_environment(backup_policy_final_state,
                                                                                      self.backup_policy_parameters)


                        # Save projected states for plotting
                        if self.plot_backup_policy:
                            state_history.append(backup_policy_final_state)

                    # !!!!!!!!!!! DEBUGGING PURPOSES !!!!!!!!!!!!!!!
                    self.projected_state_evolution.append(np.mean(backup_policy_final_state, axis=1))

                    if self.plot_backup_policy:
                        times = np.arange(k, k+len(state_history))
                        filename = self.output_folder + '/k_{}_backup_policy_sim'.format(k)
                        self.plot_uncertain_state_space_propagation(times, state_history, file_directory=filename)

                    # Compute back-up for expected arrival interval
                    self.backup_arrival_found,\
                    arrival_backup_policy_parameters,\
                    arrival_backup_policy_final_state,\
                    arrival_backup_size = self.find_backup_policy(backup_policy_final_state,
                                                                  self.backup_policy_parameters,
                                                                  0)

                    # Store back-up if found
                    if self.backup_arrival_found:
                        print('\t\t\t SHERPA: k = {:3d} - Found backup policy for arrival state'.format(k))
                        self.arrival_backup_policy_parameters = arrival_backup_policy_parameters
                        self.arrival_backup_policy_final_state = arrival_backup_policy_final_state
                        self.arrival_backup_size = arrival_backup_size

                    else:
                        self.arrival_backup_policy_parameters = np.asmatrix(np.copy(self.backup_policy_parameters))
                        self.arrival_backup_policy_final_state = self.simulate_agent_in_uncertain_environment(backup_policy_final_state,
                                                                                                      self.backup_policy_parameters)

                    # Change "backup taken" flag
                    self.backup_taken = True

                    # output_action = np.mean(interval_action, axis=1)
                    # output_action = np.asmatrix(output_action)

                    # !!!!!!!!!!!!!!!!!! DEBUGGING PURPOSES !!!!!!!!!!!!!!!!!
                    self.actual_state_evolution.append(input_state)

                    # Compute first action based on policy with backup policy parameters
                    backup_interval_action = self.policy(interval_state, self.backup_policy_parameters)

                    # Saturate backup interval action
                    backup_interval_action[:,0] = self.action_saturation(backup_interval_action[:,0])
                    backup_interval_action[:,1] = self.action_saturation(backup_interval_action[:,1])

                    # Define the action as the middle of the interval formulation (by using the mean)
                    backup_action = np.mean(backup_interval_action, axis=1)

                    # Assign the backup action to the output action variable
                    output_action = self.action_saturation(backup_action)

                    # Set the remaining backup steps to take, to its maximum
                    self.backup_steps_remaining = self.backup_size-1 #len(state_history)

                else:
                    if self.last_resort_action == 'pass-through':
                        print('\t\t\t SHERPA: k = {:3d} - Pass-Through: Unsafe Agent Action...'.format(k))
                        output_action = input_action
                    else:
                        print('\t\t\t SHERPA: k = {:3d} - Random Action...'.format(k))

                    self.random_acitons_steps.append(k)
        else:
            output_action = input_action

        return output_action
########################################################################################################################

    def assess_risk_of_state(self, state):
        '''
        Risk function used to assess the risk associated with a state.
        a 'reach' is determined based on an imaginary sensor range. For so long as
        the reach stays outside the FSS, the risk is said to be zero (0%). If it does fall into
        the FSS, the risk is set to 1 (100%). This is a binary risk-function.

        :param state: state of the agent
        :return reach, risk: Both the reach and the associated risk of the input state.
        '''

        # Computing a state's reach
        reach = state[self.RSS] + self.sensor_reach[self.RSS,:]

        # Checks for breach of FSS, when considering a state's reach
        risk_lower_FSS = np.any(reach[:,0] <= self.FSS[self.RSS][:,0])
        risk_upper_FSS = np.any(reach[:,1] >= self.FSS[self.RSS][:,1])

        # Define risk as a bool describing whether risk is present (True) or not (False)
        risk = risk_lower_FSS or risk_upper_FSS

        return reach, risk

    def augmented_state_to_state_and_reference(self, augmented_state):
        '''
        Separates state and reference signal from an augmented state.

        :param augmented_state: A column vector form of the augmented state
        :return: state, reference, the state and reference indexed from the augmented state
        '''

        # Separate state
        state = augmented_state[:-self.p, 0]

        # Separate reference signal
        reference = augmented_state[-self.p:, 0]

        return state, reference

    def check_closeness_condition(self, x_itrv):
        '''
        Checks if x_itrv is within a given closeness bound from any previously visited states.

        :param x_itrv: state interval
        :return: close, a boolean denoting if a previously visisted state is close to x_itrv
        '''

        # Compute parametric distance between itrv_state and all previously saved states
        lower_x_bound_diffs = x_itrv[:,0] - self.state_history
        upper_x_bound_diffs = x_itrv[:,1] - self.state_history

        # Check individually whether the above distances are within the epsilon bounds
        fulfill_closeness_lower = lower_x_bound_diffs >= self.v_epsilon[:,0]
        fulfill_closeness_upper = upper_x_bound_diffs <= self.v_epsilon[:,1]

        # limit closeness condition to selected states from configuration
        fulfill_closeness_lower = fulfill_closeness_lower[self.modality_states,:]
        fulfill_closeness_upper = fulfill_closeness_upper[self.modality_states,:]

        # Check if combinations of above arrays of booleans (basically an AND check)
        fulfill_closeness = np.multiply(fulfill_closeness_lower, fulfill_closeness_upper)
        #print(np.sum(fulfill_closeness, axis=1))
        fulfill_closeness = np.prod(fulfill_closeness, axis=0)

        # Check if any combination fulfills the closeness condition
        close = np.any(fulfill_closeness)

        return close

    def check_if_backup_policy_in_history(self, backup_policy):
        '''
        Checks whether the input policy has been used before.

        :param backup_policy:
        :return:
        '''

        # Convert backup policy to array format and reshape to vector format
        backup_policy = np.asarray(backup_policy).reshape((self.policy_size, 1))

        # Compute the difference between input policy and all policies in used in the past
        backup_history_diff = self.backup_policy_history - backup_policy

        # Get the absolute value of these differences
        backup_history_diff_abs = np.abs(backup_history_diff)

        # Compute the column wise sum of all the differences
        column_wise_diff_sum = np.sum(backup_history_diff_abs, axis=0)

        # Determine whether policy has been used in the past
        backup_policy_used_before = np.any(column_wise_diff_sum < 1e-18)

        return backup_policy_used_before

    def check_interval_state_safety(self, interval_state):
        '''
        Checks if interval state is within the Safe State Space

        :param interval_state: interval-vector of state
        :return: safe, a boolean stating the safety of the interval state
        '''


        safe_lower = np.all(interval_state[self.RSS][:, 0] >= self.SSS[self.RSS][:, 0])
        safe_upper = np.all(interval_state[self.RSS][:, 1] <= self.SSS[self.RSS][:, 1])
        safe = safe_lower and safe_upper

        return safe


    def generate_policies_from_current_policy(self, policy_parameters, num_policies, sigma=0.05, epsilon=0.1):
        '''
        Defines a strategy closely related to the epsilon-greedy method. Each parameter is varied by a few
        percentage points. The percentage variation, e, is taken from a normal distribution with mean set to 0 and standard
        defined by sigma. Every parameter is varied according to:

            p_new = p_old * (1 - e)

        :param policy_parameters: Vector/matrix of all policy parameters
        :param num_policies: number of policies to be generated
        :param sigma: Standard deviation of distribution for parameter skew
        :param epsilon:
        return: Set of new policy parameters
        '''

        if np.all(np.abs(policy_parameters) < 1e-8):
            policy_parameter_scale = np.diff(self.policy_parameter_range)[0]
            epsilon_rnd_policy = 0.7  # Augment the chance of random policies
        else:
            policy_parameter_scale = np.abs(np.max(policy_parameters) - np.min(policy_parameters))
            epsilon_rnd_policy = epsilon

        # Generate array of booleans defining whether a random policy is created
        gen_random_policies = np.random.choice([True, False],
                                               num_policies,
                                               p=[epsilon_rnd_policy, 1-epsilon_rnd_policy])

        new_policies_set = []
        for i in range(num_policies):

            # Generate a completely random policy
            if gen_random_policies[i]:
                new_policy_parameters = (np.random.random(policy_parameters.shape)-0.5)*(policy_parameter_scale)

            else:
                # Define zero and non-zero masks
                non_zeros_mask = np.abs(policy_parameters) > 1e-8
                zeros_mask = np.invert(non_zeros_mask)

                # Generate random numbers
                random_elements = np.random.normal(0, sigma, size=policy_parameters.shape)

                # Define the factors of each policy parameter
                param_factors = np.ones(policy_parameters.shape) - random_elements

                # Compute parameters of new policy
                new_policy_parameters = np.multiply(policy_parameters, param_factors)
                new_policy_parameters[zeros_mask] = random_elements[zeros_mask] * policy_parameter_scale

            # Save new parameters
            new_policies_set.append(np.asmatrix(new_policy_parameters))

        return new_policies_set

    def generate_random_action(self):
        '''
        Generates random action vector within the action limits set in self.initialise()

        :return: action, a random action vector
        '''

        # Define normal distribution mean (center of normal distribution bell)
        means = np.mean(self.action_bounds, axis=1)

        # Define standard deviation (to contain 99.7% of possible inputs within action_bounds)
        standard_deviations = 1/3 * (self.action_bounds[:,1] - means)

        # Define random action vector
        action = np.random.normal(means, standard_deviations, (self.m, 1))

        # Convert to matrix format
        action = np.asmatrix(action)

        # Saturate action if necessary
        action = self.action_saturation(action)

        return action

    def find_backup_policy(self, state, initial_policy_parameters, iteration):
        '''
        Finds a backup policy for the sate in which the agent finds itself.

        :param state: initial interval state
        :param initial_policy_parameters:
        :param initial_policy_parameters:
        :param iteration:
        :return: The parameters of the backup policy
        '''

        # Generate set of new policies from initial policy parameters
        policy_set = self.generate_policies_from_current_policy(initial_policy_parameters, self.num_iterations_backup-1)

        # Combine generated policies with current agent policy and backup policy history
        policy_set = [self.policy_parameters] + self.backup_policy_history_list + policy_set

        # Process policy search in multiprocessing module
        if self.policy_search_processing in ['parallel', 'p']:
            search_output = self.find_backup_policy_parallel(state, policy_set)

        # Process policy search in sequential order
        elif self.policy_search_processing in ['sequential', 's']:
            search_output = self.find_backup_policy_sequential(state, policy_set)

        else:
            raise ValueError('Policy search processing type "{}" not implemented'.format(self.policy_search_processing))

        # Unfold search output variables
        found_backup, backup_policy_parameters, backup_policy_arrival_state, backup_size = search_output

        return found_backup, backup_policy_parameters, backup_policy_arrival_state, backup_size

    def find_backup_policy_parallel(self, state, policy_set):
        '''
        Searches for a backup policy by going through the given set of policies in batches of N policies.
        N is defined as the number of CPU cores on the machine.

        :param state: initial state from which the projections are started
        :param policy_set: Set of policy parameters to be tested
        :return: found_backup, backup_policy_parameters, backup_policy_arrival_state, backup_size
        '''

        # Initialise values for output parameters
        backup_policy_parameters = np.asmatrix(np.zeros(policy_set[0].shape))
        policy_projections_output = []
        backup_found_booleans = []
        backup_policy_arrival_state = np.asmatrix(np.copy(state))
        found_backup = False
        backup_size = 0

        # Define the batch size
        num_parallel_policies = self.num_cpus

        # Compute the number of batches that have to be crated
        num_policy_batches = 1 + int(len(policy_set) / num_parallel_policies)

        # Create processing pool
        with mp.Pool(processes=self.num_cpus) as worker_pool:

            # Search for policies
            i = 0
            while i < num_policy_batches and not found_backup:
                # Reset evaluated state interval
                x_eval = state

                # Determine how many policies are contained in current batch
                if (i + 1) * num_parallel_policies > len(policy_set):
                    num_policies_in_batch = int(len(policy_set) - i * num_parallel_policies)
                else:
                    num_policies_in_batch = num_parallel_policies

                # Generate input data for next batch of potential backup policies
                policy_projections_input = []
                for j in range(num_policies_in_batch):
                    policy_ind = int(i * num_parallel_policies + j)
                    policy = policy_set[policy_ind]
                    policy_projections_input.append([x_eval, policy])

                # Policy projection (Propagate state using the policies in the batch)
                policy_projections_output = worker_pool.map(self.policy_projections, policy_projections_input)

                # Check if backup policy was found
                backup_found_booleans = np.array([x[0] for x in policy_projections_output])
                found_backup = np.any(backup_found_booleans == 1)

                # Increment the batch index
                i += 1

        # # Close worker pool
        worker_pool.close()
        worker_pool.terminate()

        # Determine which of the policies of the batch is the backup one
        if found_backup:
            # Find the index of the first possible backup policy
            batch_backup_policy_ind = list(backup_found_booleans).index(1)
            backup_policy_ind = int((i - 1) * num_parallel_policies)
            backup_policy_ind += batch_backup_policy_ind

            # Select the backup policy outputs
            backup_policy_outputs = policy_projections_output[batch_backup_policy_ind]

            # Get the correct policy parameters for this policy
            backup_policy_parameters = policy_set[backup_policy_ind]

            # Get the number of time steps that were required to confirm the backup policy validity
            backup_size = backup_policy_outputs[1]
            # self.backup_size = backup_policy_outputs[1]

            # Arrival state in policy projection
            backup_policy_arrival_state = backup_policy_outputs[2]

            # Check if in backup policy history
            in_backup_policy_history = self.check_if_backup_policy_in_history(backup_policy_parameters)

            # Add it to policy history if not used before
            if not in_backup_policy_history:
                backup_policy_parameters_vector = np.asarray(backup_policy_parameters).reshape((self.policy_size, 1))
                self.backup_policy_history = np.hstack((self.backup_policy_history, backup_policy_parameters_vector))
                self.backup_policy_history_list.append(backup_policy_parameters)

        return found_backup, backup_policy_parameters, backup_policy_arrival_state, backup_size

    def find_backup_policy_sequential(self, state, policy_set):
        '''
        Searches for a backup policy by going through the given set of policies sequentially.

        :param state: initial state from which the projections are started
        :param policy_set: Set of policy parameters to be tested
        :return: found_backup, backup_policy_parameters, backup_policy_arrival_state, backup_size
        '''

        # Initialise values for output parameters
        backup_policy_parameters = np.asmatrix(np.zeros(policy_set[0].shape))
        backup_policy_arrival_state = np.asmatrix(np.copy(state))
        found_backup = False
        backup_size = 0

        # Initialise other variables in while loop
        policy_parameters = None
        i = 0

        # Go over all policies in the policy set (until a valid backup is found)
        while i < len(policy_set) and not found_backup:
            # Select one policy from the policy set
            policy_parameters = policy_set[i]

            # Compose the inputs for the policy projection function
            projection_inputs = [state, policy_parameters]

            # Project policy
            projection_outputs = self.policy_projections(projection_inputs)

            # unfold projection outputs in respective variables
            found_backup, backup_size, backup_policy_arrival_state = projection_outputs

            # Increment the loop index
            i += 1

        # Determine which of the policies of the batch is the backup one
        if found_backup:

            # Define current policy parameters as backup policy parameters
            backup_policy_parameters = policy_parameters

            # Check if in backup policy history
            in_backup_policy_history = self.check_if_backup_policy_in_history(backup_policy_parameters)

            # Add it to policy history if not used before
            if not in_backup_policy_history:
                backup_policy_parameters_vector = np.asarray(backup_policy_parameters).reshape((self.policy_size,1))
                self.backup_policy_history = np.hstack((self.backup_policy_history,
                                                        backup_policy_parameters_vector))
                self.backup_policy_history_list.append(backup_policy_parameters)

        return found_backup, backup_policy_parameters, backup_policy_arrival_state, backup_size

    def policy_projections(self, inputs):
        '''
        Projects uncertain state over time for a given policy, initial state and initial action.

        This function can either be used in (parallel) multiprocessing or sequentially.
        :return:
        '''

        # Extract inputs from input list
        state, policy_parameters = inputs

        # Initialise variables
        j = 0
        safe = True
        close = False
        found_backup = False
        x_eval_itrv_smaller_than_epsilon_itrv = True
        backup_size = 0

        # Define initial state
        x_eval = np.asmatrix(np.copy(state))

        # Backup policy projection
        while j <= self.backup_projection_size[1] and safe and x_eval_itrv_smaller_than_epsilon_itrv and not close:

            # Simulate agent in an uncertain environment
            x_eval = self.simulate_agent_in_uncertain_environment(x_eval, policy_parameters)

            # Check if state interval is larger than epsilon interval
            if np.any(np.diff(x_eval[self.modality_states,:]) > self.v_epsilon_itrv_size):
                x_eval_itrv_smaller_than_epsilon_itrv = False

            # Check safety of newly propagated state
            safe = self.check_interval_state_safety(x_eval)

            # Check closeness condition only if the state propagation was safe and the state interval is smaller
            #     than the epsilon interval
            if safe and x_eval_itrv_smaller_than_epsilon_itrv and (self.backup_projection_size[0] <= j):
                close = self.check_closeness_condition(x_eval)

            # increment counter
            j += 1

        # Check whether all condition for a valid backup policy are satisfied
        if safe and x_eval_itrv_smaller_than_epsilon_itrv and close:
            found_backup = True
            backup_size = j

        # Collect projection results
        projection_results = [found_backup, backup_size, x_eval]

        return projection_results

    def simulate_agent_in_uncertain_environment(self, state, policy_parameters):
        '''
        Simulates the agent-environment interaction over one time step
            1) construct augmented state
            2) compute action to be taken at state based on given policy parameters
            3) propagate state through bounding model

        :param state: interval formulation of the state
        :param policy_parameters: policy parameters fitting the requirements of the uncertain policy function
        :return: new_state, the state at the next time step
        '''

        # Determine action based on given policy parameters and policy function
        action = self.policy(state, policy_parameters)

        # Check saturation of action on both lower and upper interval bound
        action[:,0] = self.action_saturation(action[:,0])
        action[:,1] = self.action_saturation(action[:,1])

        # Propagate state
        new_state = self.bm.propagate_state(state, action)

        return new_state

    def generate_output(self, curricular_step, configuration):
        '''
        Generates all the desired outputs for the SHERPA safety filter.

        :param curricular_step: Simulated instance of the curricular step
        :param configuration: configuration of the safety filter
        :return: None
        '''

        # Define parent directory
        parent_dir = curricular_step.folder_name

        # Plot evolution of SSS with SHERPA interventions
        SSS_evo_filename = 'SSS_evolution_' + curricular_step.name + curricular_step.stable_policy_found * '_converged'
        SSS_evo_directory = parent_dir + '/' + SSS_evo_filename
        self.plot_SSS_evolution(curricular_step.times,
                                curricular_step.state_storage,
                                curricular_step.system_identifier.num_samples,
                                filename=SSS_evo_directory)
        return

    def plot_uncertain_state_space_propagation(self, times, interval_states, file_directory=None):
        '''
        Plots the propagation of uncertain states through an uncertain state space

        :param times: list or array of time steps
        :param interval_states: list of interval state arrays (columns representing lower and upper bounds)
        :param file_directory: optional, None (default) when not saving file. Otherwise,
                               contains the directory of where the file should be saved
        :return: None
        '''

        # Convert time step to time in seconds
        times = times * self.dt

        # Index of desired state element
        desired_state_indices = np.array(self.RSS)

        if desired_state_indices.shape[0] > 1:
            # Create folder directory for backup policy plots of all elements
            os.mkdir(file_directory)

            # Derive filename basis
            filename = file_directory.split('/')[-1]

            # reconstruct file directory
            file_directory = file_directory + '/' + filename

        # Convert list of interval states into a 3D array
        states = np.asarray(interval_states)

        # Lower and upper bounds of all states
        lower_bounds = states[:,:,0]
        upper_bounds = states[:,:,1]

        for state_index in desired_state_indices:

            # Upper and lower bound of desired state element
            lower_bounds_state = lower_bounds[:,state_index]
            upper_bounds_state = upper_bounds[:,state_index]
            all_x = np.mean(states, axis=2)[:,state_index]

            # Upper and lower bound of reference state
            # lower_bounds_reference = lower_bounds[:,-self.p]
            # upper_bounds_reference = upper_bounds[:,-self.p]
            ref_index = self.RSS_to_ref_mapping[state_index]
            all_ref_x = np.mean(states, axis=2)[:,(-self.p + ref_index)]

            # compute coordinates of uncertainty propagation of reference
            # lower_coords_ref = [*zip(times, lower_bounds_reference)]
            # upper_coords_ref = [*zip(times, upper_bounds_reference)][::-1]
            # all_polygon_coords_ref = lower_coords_ref + upper_coords_ref

            # compute coordinates of uncertainty propagation of state
            lower_coords_x = [*zip(times, lower_bounds_state)]
            upper_coords_x = [*zip(times, upper_bounds_state)][::-1]
            all_polygon_coords_state = lower_coords_x + upper_coords_x

            # Initiate figure instance
            fig = plt.figure(figsize=(8.3, 3))
            ax1 = fig.add_subplot(1,1,1)

            # Drawing the grid lines
            ax1.grid(b=True, color='#888888', which='major', alpha=0.6, linewidth=0.2)
            ax1.grid(b=True, color='#888888', which='minor', alpha=0.6, linewidth=0.2)

            # Plot FSS bounds
            rect_lower_FSS = Rectangle((times[0], self.FSS[state_index, 0]),
                                       (times[-1] - times[0]),
                                       1.5*self.FSS[state_index,0],
                                       lw=1, ec='none', fc='#ff0000', alpha=0.5, label='FSS')
            rect_upper_FSS = Rectangle((times[0], self.FSS[state_index, 1]),
                                       (times[-1] - times[0]),
                                       1.5*self.FSS[state_index,1],
                                       lw=1, ec='none', fc='#ff0000', alpha=0.5)
            ax1.add_patch(rect_lower_FSS)
            ax1.add_patch(rect_upper_FSS)
            # # plot uncertainty propagation of reference
            # poly = Polygon(all_polygon_coords_ref, fc='#adacac', ec='#828282', alpha=0.7, lw=1, ls='--', label=r'$x^{r}_{1}$')
            # ax1.add_patch(poly)

            # plot state of reference position
            ax1.plot(times, all_ref_x, c='#525252', lw=0.6, ls='--', label=r'$x^{r}_{'+'{}'.format(state_index+1)+'}$')

            # plot uncertainty propagation of state
            label = r'$x_{'+r'{}'.format(state_index+1) + r',unc. bound}$'
            poly = Polygon(all_polygon_coords_state, fc='#2459e0', ec='#113ba6', alpha=0.3, lw=0.4, ls='-', label=label)
            ax1.add_patch(poly)

            # plot state of last mass
            label = r'$x_{'+r'{}'.format(state_index+1) + r',unc}$'
            ax1.plot(times, all_x, c='b', lw=0.6, ls='-.', label=label)

            # Set plot axis labels and title
            ax1.set_title('Backup Policy Projection of Uncertain State vs Time')
            ax1.set_ylabel(r'$x_{'+'{}'.format(state_index+1)+'}$')
            ax1.set_xlabel('Time [s]')

            # Set plot limits
            ax1.set_xlim([times[0], times[-1]])
            ax1.set_ylim(np.asarray(self.FSS)[state_index,:] * 1.2)

            # Set plot legend
            ax1.legend(loc='upper left', fontsize=15, bbox_to_anchor=(1.01, 1.03))

            # Add figure padding
            plt.subplots_adjust(left=0.1, bottom=0.15, right=0.75, top=0.9, wspace=0.1, hspace=0.4)

            # Save figure
            if file_directory is not None:
                plt.savefig(file_directory + '_element_{}.pdf'.format(state_index+1))

            plt.close()

        return

    def plot_SSS_evolution(self, times, states, sys_id_num_samples, filename=None):
        '''
        Plotting the evolution of the safe state space over time along with interventions of SHERPA.
        The Safe State Space of each state that has received Fatal State Space bounds, is plotted.

        :param times: time series (in seconds)
        :param states: array of horizontally stacked states
        :param filename: name of the file to which the plot will be saved as pdf
        :return: None
        '''

        FSS = np.asarray(self.FSS)

        # Define the states that are to be plotted
        desired_state_indices = self.RSS

        # Define upper and lower bounds of SSS
        SSS_lower_bounds = self.SSS_storage[:(self.n+self.p),:]
        SSS_upper_bounds = self.SSS_storage[(self.n+self.p):,:]

        # Create masks for areas designating action of SHERPA
        suggest_action_steps_mask = np.zeros(times.shape).astype('bool')
        backup_policy_steps_mask = np.zeros(times.shape).astype('bool')
        random_action_steps_mask = np.zeros(times.shape).astype('bool')

        # Define time steps where new actions were suggested
        if len(self.action_suggestion_steps) > 0:
            suggest_action_steps_mask[np.array(self.action_suggestion_steps)] = 1

            suggest_action_lower_bound = np.zeros(times.shape)
            suggest_action_upper_bound = np.zeros(times.shape)
            suggest_action_lower_bound[suggest_action_steps_mask] = 1
            suggest_action_upper_bound[suggest_action_steps_mask] = 1

        # Define time steps where backup policy is used
        if len(self.backup_policy_steps) > 0:
            backup_policy_steps_mask[np.array(self.backup_policy_steps)] = 1

            backup_policy_lower_bound = np.zeros(times.shape)
            backup_policy_upper_bound = np.zeros(times.shape)
            backup_policy_lower_bound[backup_policy_steps_mask] = 1
            backup_policy_upper_bound[backup_policy_steps_mask] = 1

        # Define time steps where random action is used
        if len(self.random_acitons_steps) > 0:
            random_action_steps_mask[np.array(self.random_acitons_steps)] = 1

            random_action_lower_bound = np.zeros(times.shape)
            random_action_upper_bound = np.zeros(times.shape)
            random_action_lower_bound[random_action_steps_mask] = 1
            random_action_upper_bound[random_action_steps_mask] = 1

        # Plot SSS and state evolution of each state element (that is desired)
        for state_index in desired_state_indices:

            # Compute coordinates for area showing where backup policy was used
            if len(self.action_suggestion_steps) > 0:
                suggest_action_lower_bound_state = suggest_action_lower_bound * FSS[state_index,0]
                suggest_action_upper_bound_state = suggest_action_upper_bound * FSS[state_index,1]
                suggest_action_lower_coors = [*zip(times, suggest_action_lower_bound_state)]
                suggest_action_upper_coors = [*zip(times, suggest_action_upper_bound_state)][::-1]
                suggest_action_poly_coors = suggest_action_lower_coors + suggest_action_upper_coors

            # Compute coordinates for area showing where backup policy was used
            if len(self.backup_policy_steps) > 0:
                backup_policy_lower_bound_state = backup_policy_lower_bound * FSS[state_index,0]
                backup_policy_upper_bound_state = backup_policy_upper_bound * FSS[state_index,1]
                backup_policy_lower_coors = [*zip(times, backup_policy_lower_bound_state)]
                backup_policy_upper_coors = [*zip(times, backup_policy_upper_bound_state)][::-1]
                backup_policy_poly_coors = backup_policy_lower_coors + backup_policy_upper_coors

            # Compute coordinates for area showing where random action was used
            if len(self.random_acitons_steps) > 0:
                random_action_lower_bound_state = random_action_lower_bound * FSS[state_index,0]
                random_action_upper_bound_state = random_action_upper_bound * FSS[state_index,1]
                random_action_lower_coors = [*zip(times, random_action_lower_bound_state)]
                random_action_upper_coors = [*zip(times, random_action_upper_bound_state)][::-1]
                random_action_poly_coors = random_action_lower_coors + random_action_upper_coors

            # Derive upper and lower bounds of RSS from SSS
            RSS_lower_bounds = np.asarray(SSS_lower_bounds[state_index,:])
            RSS_upper_bounds = np.asarray(SSS_upper_bounds[state_index,:])

            RSS_lower_bounds = RSS_lower_bounds.reshape((RSS_lower_bounds.shape[1],))
            RSS_upper_bounds = RSS_upper_bounds.reshape((RSS_upper_bounds.shape[1],))

            RSS_lower_coords = [*zip(times, RSS_lower_bounds)]
            RSS_upper_coords = [*zip(times, RSS_upper_bounds)][::-1]
            all_polygon_coords_RSS = RSS_lower_coords + RSS_upper_coords

            # Derive state to be plotted
            all_x = states[state_index,:].reshape((times.shape[0],))

            # Initiate figure instance
            fig = plt.figure(figsize=(14, 5))
            ax1 = fig.add_subplot(1,1,1)

            # Drawing the grid lines
            ax1.grid(b=True, color='#888888', which='major', linewidth=0.2)
            ax1.grid(b=True, color='#888888', which='minor', linewidth=0.2)

            # Plot SHERPA inactivity time interval
            rect = Rectangle((times[0], FSS[state_index,0]),
                                       (times[sys_id_num_samples-1] - times[0]),
                                       np.diff(FSS[state_index,:])[0],
                                       lw=1, ec='none', fc='#c0c0c0', alpha=0.5, label='SHERPA Deactivated')
            ax1.add_patch(rect)

            # Plot suggestions of new actions  polygon
            if len(self.action_suggestion_steps) > 0:
                poly = Polygon(suggest_action_poly_coors, fc='#ecf00e', ec='#ecf00e', alpha=0.6, lw=0.4, ls='--',
                               label='Action Suggestion')
                ax1.add_patch(poly)

            # Plot backup policy usage polygon
            if len(self.backup_policy_steps) > 0:
                poly = Polygon(backup_policy_poly_coors, fc='#fc8b00', ec='#fc8b00', alpha=0.6, lw=0.4, ls='--',
                               label='Backup Policy Use')
                ax1.add_patch(poly)

            # Plot random action usage polygon
            if len(self.random_acitons_steps) > 0:
                poly = Polygon(random_action_poly_coors, fc='#ff0000', ec='#ff0000', alpha=0.6, lw=0.4, ls='--',
                               label='Random Action')
                ax1.add_patch(poly)

            # Plot zero line and SSS bounds
            lower_FSS_bounds = [FSS[state_index,:][0], FSS[state_index,:][0]]
            upper_FSS_bounds = [FSS[state_index,:][1], FSS[state_index,:][1]]
            ax1.plot([times[0], times[-1]], [0, 0], lw=1, c='k')
            ax1.plot([times[0], times[-1]], lower_FSS_bounds, lw=0.8, c='k', ls='--', label='FSS Bounds')
            ax1.plot([times[0], times[-1]], upper_FSS_bounds, lw=0.8, c='k', ls='--')

            # Plot uncertainty propagation of reference
            poly = Polygon(all_polygon_coords_RSS, fc='#36c95e', ec='#2b8c46', alpha=0.3, lw=0.4, ls='--', label='SSS')
            ax1.add_patch(poly)

            # Plot state of last mass
            ax1.plot(times, all_x, c='b', lw=0.5, label=r'$x_{}$'.format(state_index+1))

            # Set plot axis labels and title
            ax1.set_title(r'State Propagation $x_{}$ with SHERPA interventions'.format(state_index+1)).set_fontsize(20)
            ax1.set_ylabel(r'$x_{}$'.format(state_index+1)).set_fontsize(15)
            ax1.set_xlabel('Time [s]').set_fontsize(15)

            # Set plot limits
            ax1.set_xlim([times[0], times[-1]])
            ax1.set_ylim(FSS[state_index,:]*1.2)

            # Set plot legend
            ax1.legend(loc='upper left', fontsize=15, bbox_to_anchor=(1.01, 1.03))

            # Add figure padding
            plt.subplots_adjust(left=0.05, bottom=0.1, right=0.75, top=0.92, wspace=0.1, hspace=0.4)

            # Save figure
            if filename is not None:
                plt.savefig(filename + '_state_element_{}.pdf'.format(state_index+1))

            plt.close()

        return


class optiSHERPA(object):
    def __init__(self):
        self.iter = 5













