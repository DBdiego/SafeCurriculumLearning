import numpy as np
import scipy
import warnings
import Interval_arithmetic as itrv_arith
import matplotlib.pyplot as plt

class Agent(object):

    def __init__(self, SS_dimensions, supervisor_action_mask, configuration, H=None):

        # Extract configuration parameters
        # Define method to be used in Q-learning. Either Value Iteration (VI) or Policy Iteration (PI)
        self.method = configuration.method

        # Learning parameters
        self.gamma = configuration.learning_rate

        # Sample factor
        self.sample_factor = configuration.sample_factor

        # State Space System dimensions
        self.n, self.m, self.p = SS_dimensions

        # Define action mask of agent (i.e. which inputs it controls)
        self.action_mask = np.ones(self.m).astype('bool')

        # Adapt agent characteristics if a supervisor is present
        if supervisor_action_mask is not None:
            # Correct value of m, if a supervisor is controlling some inputs
            self.m -= np.sum(supervisor_action_mask)

            # Create action mask
            self.action_mask = np.invert(supervisor_action_mask)

        # Number of samples collected for updating H
        min_num_samples = (self.n+self.m+self.p)*(self.n+self.m+self.p+1) / 2
        self.num_samples = int(self.sample_factor * min_num_samples)

        # Initialise Z vector
        self.Z_shape = (self.n + self.m + self.p, 1)
        self.Z_k = None
        self.Z_k_excite = None
        self.Z_k_1 = None
        self.Z_k_1_excite = None
        self.Z_kron_mask = np.mask_indices(self.n+self.m+self.p, np.triu)

        # Create saturation of action amplitude
        self.action_bounds = configuration.action_bounds
        self.min_input = self.action_bounds[:,0].reshape((self.action_mask.shape[0], 1))
        self.max_input = self.action_bounds[:,1].reshape((self.action_mask.shape[0], 1))

        # Kernel initialisation
        self.H_shape = ((self.n+self.m+self.p), (self.n+self.m+self.p))
        self.num_independent_vals_H = int((self.n+self.m+self.p)*(self.n+self.m+self.p+1)/2)
        self.H_j = None
        self.H_j_vector = None

        # Stabilising control gain initialisation
        self.policy_parameters = None

        # Initialise agent cost
        self.cost_k = None
        self.cost_k_1 = None

        # Compute initial H and K1 matrices
        if H is not None:
            self.H_j = H
            self.policy_parameters = self.calculate_K1()
        else:
            self.define_H_K1()
        self.H_j_vector = self.H_to_H_1D()

        # Create array to store all kernel values
        self.H_storage = np.copy(self.H_j_vector)
        self.Q_storage = np.zeros((1,0))
        self.u_excite_storage = None

        # Define sample collectors
        # --> Policy Iteration
        self.regr_m_PI = None
        self.regr_v_PI = None

        # --> Value Iteration
        self.regr_m_VI = None
        self.regr_v_VI = None

        # Reset all sample collector arrays
        self.reset_sample_collector_arrays()

        return

    ############################################# - Mendatory Functions - #############################################
    def action_saturation(self, u_k):
        '''
        Saturates the agent's action based on given bounds

        :param u_k: desired action of the agent
        :return: saturated action of the agent
        '''

        # Check for max force saturation:
        u_k = np.minimum(u_k, self.max_input)

        # Check for min force saturation:
        u_k = np.maximum(u_k, self.min_input)

        return u_k


    def collect_samples(self, X_k_msrd, u_k, u_k_excite, cost_k):
        '''
        Collect samples to construct regression matrices and vectors for both VI and PI. This function
        shifts variables of current time step to past time step as well.

        :param X_k: Augmented state at time step k
        :param u_k: action vector at time step k
        :param u_k_excite: action vector with excitation signal at time step k
        :param cost_k: cost at time step k
        :return: None
        '''

        # Shifting variables by one time step
        self.Z_k_1 = self.Z_k
        self.Z_k_1_excite = self.Z_k_excite
        self.cost_k_1 = self.cost_k

        # Assigning new values to current-time-step attributes
        self.Z_k = self.construct_Z(X_k_msrd, u_k)
        self.Z_k_excite = self.construct_Z(X_k_msrd, u_k_excite)
        self.cost_k = cost_k

        # Collect excited input for PE condition
        self.u_excite_storage = np.hstack((self.u_excite_storage, u_k_excite))
        self.Q_storage = np.hstack((self.Q_storage, 0.5*np.matmul(np.matmul(self.Z_k.T,self.H_j), self.Z_k)))

        # Append to regression matrices and vectors if not the first time step (i.e., when k_1 are not None)
        if self.Z_k_1 is not None:
            # Compute all the desired values of Z_kron for all combinations of Z_k, Z_k_1
            Z_k_1_kron_noise_desired = self.calc_desired_Z_kron_values_k(self.Z_k_1_excite)
            Z_k_kron_desired = self.calc_desired_Z_kron_values_k(self.Z_k)

            # Regression Matrix
            regr_m_values_PI = Z_k_1_kron_noise_desired - self.gamma * Z_k_kron_desired
            regr_m_values_VI = Z_k_1_kron_noise_desired

            self.regr_m_PI = np.vstack((self.regr_m_PI, regr_m_values_PI))
            self.regr_m_VI = np.vstack((self.regr_m_VI, regr_m_values_VI))

            # Regression Vector
            regr_v_value_PI = self.cost_k_1
            regr_v_value_VI = self.cost_k_1 + self.gamma * self.Z_k.T * self.H_j * self.Z_k

            self.regr_v_PI = np.vstack((self.regr_v_PI, regr_v_value_PI))
            self.regr_v_VI = np.vstack((self.regr_v_VI, regr_v_value_VI))

        return

    def evaluate_policy(self):
        '''
        Evaluates the current policy by finding a new kernel matrix H. The kernel
        matrix H is then used during the simulation to compute the Q value of a given
        state.

        :return: None
        '''

        # Boolean denoting whether policy evaluation succeeded
        policy_evaluation_success = True

        # Check if persistence of excitation condition is fulfilled
        PE_is_ok = self.check_persistence_excitation(self.u_excite_storage)

        # Perform Policy evaluation
        if PE_is_ok:

            # Peform Batch Ordinary Least Squares
            if self.method == 'VI':
                H_j_plus_1_vector = self.batch_ordinary_least_squares(self.regr_m_VI,  self.regr_v_VI)
            elif self.method == 'PI':
                H_j_plus_1_vector = self.batch_ordinary_least_squares(self.regr_m_PI, self.regr_v_PI)
            else:
                raise ValueError('agent.method should be VI or PI')

            # Converting output vector into a matrix
            H = self.H_vector_to_H_matrix(H_j_plus_1_vector, self.H_shape)
            H_j_plus_1 = np.asmatrix(np.copy(H))

            # Warning when incorrect H was found (negative value on the diagonal)
            if np.any([val < 0 for val in np.diag(H_j_plus_1)]):
                warnings.warn('Negative value along diagonal of H')

            if np.isnan(np.sum(H_j_plus_1)):
                warnings.warn('H contains NaN values')
                policy_evaluation_success = False

            # Computing norm of the difference between old H and new H
            self.H_diff_norm = np.linalg.norm(self.H_j-H_j_plus_1)

            # Updating H_j to new value of kernel
            self.H_j = np.asmatrix(np.copy(H_j_plus_1))
            self.H_j_vector = self.H_to_H_1D()

            # Save kernel values
            self.H_storage = np.hstack((self.H_storage, self.H_j_vector))

            # Reset sample collectors after policy evaluation
            self.reset_sample_collector_arrays()

        else:
            warnings.warn('Persistence of Excitation condition not fulfilled, H update negated')

        return policy_evaluation_success

    def improve_policy(self):
        '''
        Improve policy of the agent by adapting the stabilizing control
        matrix K1, based on the newly found kernel matrix H.

        :return: None
        '''

        # Saving Previous K1 for comparison
        K1_j_1 = np.asmatrix(np.copy(self.policy_parameters))

        # Compute new K1
        K1_j = self.calculate_K1()

        # Compute norm of the difference between old and new K1
        K1_diff_norm = np.linalg.norm(K1_j_1 - K1_j)

        return K1_j

    def policy(self, state, policy_parameters):
        '''
        Tranformation of state to action, based on the policy parameters given as input

        :param state: the state in which the agent finds itself
        :param policy_parameters: the parameters defining the policy
        :return: action, action taken by the agent
        '''

        # Computing control force
        action = -np.matmul(policy_parameters, state)

        return action

    def uncertain_policy(self, interval_state, policy_parameters):
        '''
        Computes the interval formulation of the action from an state formulated in interval form and
        policy parameters (which are assumed to have a uncertainty of zero)

        :param interval_state: interval formulation of the state
        :param policy_parameters: certain policy parameters
        :return: interval_action, the interval formulation of the action
        '''

        # Create an interval formulation for the policy parameters
        policy_params_uncertainty = np.zeros(policy_parameters.shape)
        interval_policy_parameters = itrv_arith.construct_uncertainty_matrix(policy_parameters,
                                                                             policy_params_uncertainty,
                                                                             uncertainty_type='relative')

        # Determine the action
        interval_action = - itrv_arith.interval_matmul(interval_policy_parameters, interval_state)

        return interval_action

    ###################################################################################################################

    def batch_ordinary_least_squares(self, regr_m, regr_v):
        '''
        Normalized Batch Least Square implementation solving
            Ax = b
        by using
            x = (A' * A)^-1 * A' * b

        :param regression_matrix:
        :param regression_vector:
        :return:
        '''

        # Computing regression matrix dot product
        regr_m_prod = regr_m.T * regr_m

        # Computing Pseudo inverse of dot product
        regr_v_final = regr_m.T * regr_v
        # regr_m_prod_inv = np.linalg.inv(regr_m_prod)
        print(np.linalg.det(regr_m_prod), np.any(np.isinf(regr_m_prod)), np.average(regr_m_prod))

        # Multiplying with transpose of regression matrix and with regression vector
        BLS_result = np.linalg.pinv(regr_m_prod, rcond=1e-20) * regr_v_final

        return BLS_result

    def calc_desired_Z_kron_values_k(self, Z):
        '''

        :param Z:
        :return:
        '''

        # Compute kronecker product of input Z
        Z_kron = np.kron(Z, Z)

        # Reshaping Z_kron into a 2D array
        Z_kron_2D = Z_kron.reshape((self.Z_shape[0], self.Z_shape[0]))

        # Take on only top right triangular part of Z_kron 2D array
        Z_kron_out = Z_kron_2D[self.Z_kron_mask]

        return np.asmatrix(np.array(Z_kron_out))

    def calculate_K1(self):
        '''
        Compute K1 matrix from H at optimisation epoch j

        :param H_j:
        :return:
        '''

        # Find Hux and Huu from H_j
        HuX = np.asmatrix(self.H_j[-self.m:, :-self.m])
        Huu = np.asmatrix(self.H_j[-self.m:, -self.m:])

        # Invert Huu
        Huu_inv = scipy.linalg.inv(Huu)

        # Computed new K1
        K1 = np.dot(Huu_inv, HuX)

        return K1

    def check_persistence_excitation(self, actions):
        '''
        Verifies that all the actions given in actions are fulfilling the PE condition

        :param actions: time series of each action
        :return: PE_condition_passed, a bool defining a pass or no pass of the condition for all actions
        '''

        PE_condition_passed = False
        action_wise_sum = np.sum(actions, axis=1)
        if np.all(np.abs(action_wise_sum) > 1e-10):
            PE_condition_passed = True

        return PE_condition_passed

    def construct_Z(self, X, u):
        '''
        Construct the Z vector by vertically stacking X and u.

        :param X: Augmented state vector
        :param u: input vector
        :return: Z
        '''

        # Stack X and u
        Z = np.vstack((X, u))

        return Z

    def create_augmented_state(self, x_k, x_ref):
        '''
        Creates augmented state matrix at time step k: X_k. This matrix is composed
        of the system state and the reference state, which are vertically stacked,
        and given as X_k = [x_k, x_k^ref]^T

        :param x_ref: reference state
        :return: X_k, the augmented state
        '''

        # Stack x and x_ref
        X_k = np.asmatrix(np.vstack((x_k, x_ref)))

        return X_k

    def define_H_K1(self):
        '''
        Initialising kernel matrix H and stabilising control gain matrix K1

        :return: None
        '''

        # Initialise H
        self.H_j = np.identity(self.H_shape[0])
        self.H_j = np.asmatrix(self.H_j)

        # Initialise K1
        self.policy_parameters = self.calculate_K1()

        return

    def H_to_H_1D(self):
        '''
        Reshapes the H matrix to a column vector
        :return: H_j_vector, the vectorised formulation of kernel H
        '''

        H_j_vector = self.H_j.reshape((self.H_shape[0]**2, 1))

        return H_j_vector

    def H_vector_to_H_matrix(self, H_vector, H_shape):
        '''
        Fit values of estimated H vector into the kernel matrix

        :param H_shape:
        :return:
        '''
        H = np.zeros(H_shape)
        k = 0
        for i in range(H_shape[0]):
            for j in range(i, H_shape[0]):
                if i != j:
                    H[i, j] = H_vector[k] / 2
                    H[j, i] = H_vector[k] / 2
                else:
                    H[i, j] = H_vector[k]
                k += 1
        return H

    def import_external_H(self, H_file):
        '''
        Import kernel matrix from external source file H_file

        :param H_file: file directory from whih to import kernel matrix H
        :return: None
        '''

        print('\n\tImporting Kernel Matrix H from: '+H_file+'\n')

        # Importing H
        H_history = np.genfromtxt(H_file, delimiter=',')
        H = H_history[:,-1].reshape(self.H_shape)

        # Save H to agent class attributes
        self.H_j = np.asmatrix(H)

        # Compute according K1
        self.policy_parameters = self.calculate_K1()

        return

    def reset_sample_collector_arrays(self):
        '''
        Reset sample collectors to empty arrays of the correct size.

        :return: None
        '''

        # --> Policy Iteration
        self.regr_m_PI = np.zeros((0, self.num_independent_vals_H))
        self.regr_v_PI = np.zeros((0,1))

        # --> Value Iteration
        self.regr_m_VI = np.zeros((0, self.num_independent_vals_H))
        self.regr_v_VI = np.zeros((0,1))

        # reset storage of action used in the PE condition
        self.u_excite_storage = np.zeros((self.m, 0))

        return

    def generate_output(self, curricular_step, configuration):
        '''

        :param curricular_step:
        :param configuration:
        :return:
        '''

        parent_dir = curricular_step.folder_name
        # Saving H_diag values
        if configuration.save_H_values and curricular_step.train_agent:
            H_data_filename = 'Kernel_evo_data_' + curricular_step.name + curricular_step.stable_policy_found*'_converged'+'.csv'
            H_data_directory = parent_dir + '/' + H_data_filename

            # Removing last column if it is filled with NaN
            if np.isnan(self.H_storage[0, -1]):
                self.H_storage = self.H_storage[:, :-1]
            np.savetxt(H_data_directory, self.H_storage, delimiter=",")

            # Plot evolution of H diagonal values
            H_evo_filename = 'Kernel_evo_plot_'+ curricular_step.name + curricular_step.stable_policy_found*'_converged'
            H_evo_directory = parent_dir + '/' + H_evo_filename
            self.plot_kernel_diagonal(curricular_step.times,
                                     self.H_storage,
                                     self.num_samples,
                                     filename=H_evo_directory)

        # Plotting Q-values
        Q_evo_filename = 'Q_value_evo_plot_'+ curricular_step.name + curricular_step.stable_policy_found*'_converged'
        Q_evo_directory = parent_dir + '/' + Q_evo_filename
        self.plot_Q_values(curricular_step.times,
                           np.array(self.Q_storage).reshape((self.Q_storage.shape[1],)),
                           filename=Q_evo_directory)


        return

    def plot_kernel_diagonal(self, times, kernel_storage, n_samples, filename=None):
        '''

        :param times:
        :param kernel_storage:
        :param n_samples:
        :param main_title:
        :param filename:
        :return:
        '''

        n_samples = int(n_samples)

        # Create figure
        fig = plt.figure(figsize=(11, 8))
        ax1 = fig.add_subplot(1,1,1)

        # Plot grid
        ax1.grid(b=True, color='#888888', which='major', linewidth=0.2)
        ax1.grid(b=True, color='#888888', which='minor', linewidth=0.2)

        # Reconstructing the time-evolution of diagonal values of H matrtix
        n_rows = int(np.sqrt(kernel_storage.shape[0]))
        H_shape = (n_rows, n_rows)
        H_diag_storage = np.empty((n_rows, times.shape[0]))

        for i in range(kernel_storage.shape[1]):
            H = kernel_storage[:,i].reshape((H_shape))

            for j in range(i*n_samples, np.min([(i+1)*n_samples, times.shape[0]])):
                H_diag_storage[:,j] = np.diag(H)

        # Plotting evolution of values on diagonal of H matrix
        for i in range(H_diag_storage.shape[0]):
            label = '$H_{'+str(i+1)+','+str(i+1)+'}$'
            label = r''+label
            ax1.plot(times, H_diag_storage[i,:], label=label)

        # Plot characteristics of ax1
        ax1.set_title('Kernel Diagonal Values vs Time').set_fontsize(20)
        ax1.set_xlim([times[0], times[-1]])
        ax1.set_yscale('log')
        ax1.set_ylabel(r'$H_{diag}$ [-]').set_fontsize(15)
        ax1.set_xlabel('Time [s]').set_fontsize(15)
        ax1.legend(loc='upper left', fontsize=15, bbox_to_anchor=(1.01, 1.03))

        # Figure padding
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.95, wspace=0.1, hspace=0.4)

        # Saving plot
        if filename is not None:
            plt.savefig(filename+'.pdf')

        plt.close()

        return

    def plot_Q_values(self, times, Q_values, filename=None):
        '''

        :param times:
        :param Q_values:
        :return:
        '''

        # Create figure
        fig = plt.figure(figsize=(11, 8))
        ax1 = fig.add_subplot(1,1,1)

        # Plot grid
        ax1.grid(b=True, color='#888888', which='major', linewidth=0.2)
        ax1.grid(b=True, color='#888888', which='minor', linewidth=0.2)

        # Plotting evolution of Q-value
        ax1.plot(times, Q_values, lw=0.8, label='Q-value')

        # Plot characteristics of ax1
        ax1.set_title('Q-value Evolution vs Time').set_fontsize(20)
        ax1.set_xlim([times[0], times[-1]])
        ax1.set_yscale('symlog')
        ax1.set_ylabel('Q [-]').set_fontsize(15)
        ax1.set_xlabel('Time [s]').set_fontsize(15)
        ax1.legend(loc='upper left', fontsize=15, bbox_to_anchor=(1.01, 1.03))

        # Figure padding
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.83, top=0.95, wspace=0.1, hspace=0.4)

        # Saving plot
        if filename is not None:
            plt.savefig(filename+'.pdf')

        plt.close()

        return

