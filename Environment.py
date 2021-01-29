import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
import scipy.integrate
import scipy.signal
import matplotlib.pyplot as plt

class Environment(object):
    def __init__(self, SS, F, supervisor_action_mask, dt, configuration, reference_map):

        # system name
        self.ref_map = reference_map

        # Loading State Space Matrices
        self.A, self.B, self.C, self.D = SS

        self.F = F
        self.SS = SS

        # Loading State Space dimensions
        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self.p = self.F.shape[0]

        # Compute number of inputs being controlled by a supervisork
        if supervisor_action_mask  is None:
            num_inputs_supervisor = 0
        else:
            num_inputs_supervisor = np.sum(supervisor_action_mask)

        # LQT Weighing matrices
        self.Q = np.asmatrix(np.diag(configuration.Q_diag_values))
        self.R = np.asmatrix(np.identity(self.m - num_inputs_supervisor) * configuration.R_diag_value)

        # Initiate attributes of C1 and Q1
        self.C1 = None
        self.Q1 = None

        # Converting State Space to discrete
        self.convert_ss_to_discrete(dt)

        # Create Augmented (SS) matrices (T, B1, C1, Q1)
        self.create_augmented_SS_LQT()

        # Computed MSD system characteristics
        self.calc_system_characteristics()

        if configuration.run_system_checks:
            print('\t\tSystem Analysis:')
            self.check_system_stability()
            # self.check_controllability()
            # self.check_observability()
            print()


    def convert_ss_to_discrete(self, dt):
        '''
        Converting State Space system form continuous to discrete

        :param dt: size of time steps
        :return: None
        '''

        # Discretise state space
        ss_discrete = scipy.signal.cont2discrete(self.SS, dt)
        A_d, B_d, C_d, D_d, dt = ss_discrete
        self.A_d = np.asmatrix(A_d)
        self.B_d = np.asmatrix(B_d)
        self.C_d = np.asmatrix(C_d)
        self.D_d = np.asmatrix(D_d)

        # Discretise F matrix of ref signal
        A_ref = self.F
        B_ref = np.asmatrix(np.zeros((A_ref.shape[0], 1)))
        C_ref = np.asmatrix(np.zeros((1, A_ref.shape[0])))
        D_ref = np.asmatrix(np.zeros(1))

        self.F_d, B_ref_d, C_ref_d, D_ref_d, dt = scipy.signal.cont2discrete([A_ref, B_ref, C_ref, D_ref], dt)

        return

    def create_augmented_SS_LQT(self):
        '''
        Augmented matrices for LQT tasks based on:
        Bahare Kiumarsi, Frank L.Lewis, Hamidreza Modares, Ali Karimpour, and Mohammad - Bagher Naghibi -
        Sistani.Reinforcement Q-learning for optimal tracking control of linear discrete-time systems with
        unknown dynamics.Automatica, 50(4):1167 – 1175, 2014. doi: https://doi.org/10.1016/j.automatica.2014.02.015.

        :return: None
        '''

        # Compute augmented A matrix, which is called T
        self.T = scipy.linalg.block_diag(self.A_d, self.F_d)

        # Define C1 matrix
        self.C1 = np.hstack((self.C_d, self.ref_map))
        self.C1 = np.asmatrix(self.C1)

        # Compute Q1 matrix
        Q1 = np.matmul(np.matmul(self.C1.T, self.Q), self.C1)
        self.Q1 = np.asmatrix(Q1)

        # Construct B1 matrix
        B1_zeros = np.zeros((self.p, self.B_d.shape[1]))
        self.B1 = np.vstack((self.B_d, B1_zeros))

        return


    def check_system_stability(self):
        '''
        Computes the environment system's stability

        :return: None
        '''
        self.is_stable = all([x <= 1e-18 for x in self.OL_eigvals])
        print('\t\t\t --> ' + (not self.is_stable)*'un' + 'stable')

        return


    def calc_system_characteristics(self):
        '''
        Finds the characterstics of the environment's system

        :return: None
        '''
        eig_vals_OL = np.linalg.eigvals(self.A)
        self.OL_eigvals = np.real(eig_vals_OL)

        return


    def closed_loop_eigs(self, K):
        '''
        Find the real part of the closed-loop eigenvalues of the system

        :param K: control gain
        :return: eig_vals_CL_out,  the real part of the closed-loop eigenvalues
        '''

        eig_vals_CL = np.linalg.eigvals(self.A-self.B*K[:,:self.n])
        eig_vals_CL_out = np.real(eig_vals_CL)
        return eig_vals_CL_out


    def ODE_state(self, state, t, action):
        '''
        Ordinary Differential Equation propagation of the state.
        x_dot = A * x + B * u

        :param state: state vector at time t
        :param t: value denoting the time
        :param action: action vector at time t
        :return:
        '''

        state = np.asmatrix(state.reshape((state.shape[0], 1)))
        action = np.asmatrix(action)
        x_dot = self.A*state + self.B*action
        x_dot = np.asarray(x_dot).reshape((state.shape[0],))
        return x_dot


    def ODE_reference(self, x_r_in, t, action):
        '''
        Ordinary Differential Equation propagation of the reference.
        x_dot = A * x + B * u

        :param x_r_in: reference state vector at time t
        :param t: value denoting the time
        :param action: action vector at time t
        :return:
        '''

        x_r_in = np.asmatrix(x_r_in.reshape((x_r_in.shape[0], 1)))
        action = np.asmatrix(action)
        B_r = np.asmatrix(np.zeros((x_r_in.shape[0], 1)))
        x_r_dot = self.F*x_r_in + B_r*action
        x_r_dot = np.asarray(x_r_dot).reshape((x_r_in.shape[0],))
        return x_r_dot


    def state_transition_continuous(self, X_t, u_t, dt):
        '''
        Find the state transition of the augmented state X in continuous
        fashion, by solving the ODE of both the reference and the system
        state.

        :param X_t: augmented state at time t
        :param u_t: action at time t
        :param dt: size of time step
        :return: X_t_plus_1, the augmented state at time (t+1)
        '''

        x_t = X_t[:-2]
        x_r_t = X_t[-2:]
        u_r_t = np.asmatrix(np.zeros((x_r_t.shape[1], 1)))

        # Find next state
        y0 = np.asarray(x_t).reshape((self.n,))
        t = [0, dt]
        x_t_plus_1 = scipy.integrate.odeint(self.ODE_state, y0, t, args=(u_t,))[1,:]

        # Formatting newly found state
        x_t_plus_1 = np.asarray(x_t_plus_1).reshape((self.n,1))
        x_t_plus_1 = np.asmatrix(x_t_plus_1)

        # ref signal
        y0_r = np.asarray(x_r_t).reshape((x_r_t.shape[0],))
        t_r = [0, dt]
        x_r_t_plus_1 = scipy.integrate.odeint(self.ODE_reference, y0_r, t_r, args=(u_r_t,))[1,:]
        x_r_t_plus_1 = np.asarray(x_r_t_plus_1).reshape((x_r_t.shape[0], 1))
        x_r_t_plus_1 = np.asmatrix(x_r_t_plus_1)

        X_t_plus_1 = np.asmatrix(np.vstack((x_t_plus_1, x_r_t_plus_1)))

        return X_t_plus_1

    def state_transition_discrete(self, X_k, u_k):
        '''
        Find the state transition of the augmented state X in discrete
        fashion.

        :param X_k: augmented state at time step k
        :param u_k: action at time step k
        :return: X_k_plus_1, the augmented state at time step (k+1)
        '''

        # Compute state at next time step
        X_k_plus_1 = np.matmul(self.T, X_k) + np.matmul(self.B1, u_k)

        return X_k_plus_1

    def calculate_RMSE(self, X):
        '''
        Compute the RMSE from the system based on the C1 matrix

        :param X: augmented state
        :return: RMSE, the root mean square error of respective state parameters
        '''

        # Find error of all output states
        ref_delta = self.C1 * np.asmatrix(X)
        ref_delta = np.asarray(ref_delta)

        # Compute RMSE
        RMSE = np.sqrt(np.sum(ref_delta**2)/ref_delta.shape[0])

        return RMSE

    def calculate_cost(self, X, u):
        '''
        Compute cost of the current state-action pair

        :param X: augmented state
        :param u: action
        :return: cost, the cost of given state and input
        '''

        # Compute cost of state
        state_cost = np.dot(np.dot(X.T, self.Q1), X)

        # Compute cost of action
        input_cost = np.dot(np.dot(u.T, self.R), u)

        # Compute total cost of state-action pair
        cost = state_cost + input_cost

        return cost

    def generate_output(self, curricular_step, configuration):
        '''
        Generates all the output plots and files related to the environment.

        :param curricular_step: Simulated curricular step class instance
        :param configuration:  Environment configuration class (from configuration file)
        :return: None
        '''

        # Define parent directory of output files
        parent_dir = curricular_step.folder_name

        # Plotting Evolution of cost
        cost_evo_filename = 'Cost_'+ curricular_step.name + curricular_step.stable_policy_found*'_converged'
        cost_evo_directory = parent_dir + '/' + cost_evo_filename
        self.plot_cost_evolution(curricular_step.times,
                                 curricular_step.cost_storage,
                                 filename=cost_evo_directory)

        # Plotting evolution of closed-loop eigenvalues
        CL_eigs_evo_filename = 'CL_eigs_'+ curricular_step.name + curricular_step.stable_policy_found*'_converged'
        CL_eigs_evo_directory = parent_dir + '/' + CL_eigs_evo_filename
        self.plot_closed_loop_eigs(curricular_step.times,
                                   curricular_step.real_closed_loop_eigs,
                                   filename=CL_eigs_evo_directory)

        # Plotting evolution of RMSE
        RMSE_evo_filename = 'RMSE_evo_'+ curricular_step.name + curricular_step.stable_policy_found*'_converged'
        RMSE_evo_directory = parent_dir + '/' + RMSE_evo_filename
        self.plot_RMSE_evolution(curricular_step.times,
                                 curricular_step.RMSE_storage,
                                 configuration.RMSE_limit,
                                 filename=RMSE_evo_directory)

        return

    def plot_cost_evolution(self, times, cost_storage, filename=None):
        '''
        Plots the evolution of the cost/reward over time

        :param times: Array of times (seconds)
        :param cost_storage: Array of the cost values
        :param filename: The name of the file containing the plot in pdf format
        :return: None
        '''

        # Create figure
        fig = plt.figure(figsize=(11, 8))
        ax1 = fig.add_subplot(1, 1, 1)

        # Plot grid
        ax1.grid(b=True, color='#888888', which='major', linewidth=0.2)
        ax1.grid(b=True, color='#888888', which='minor', linewidth=0.2)

        # Plot cost evolution vs time
        ax1.plot(times, cost_storage, lw=1, label='$Cost = $X_{k}^{T}Q_{1}X_{k} + u_{k}^{T}Ru_{k}$')

        # Plot characteristics of ax1
        ax1.set_title('Cost vs Time').set_fontsize(20)
        ax1.set_ylabel('Cost [-]').set_fontsize(15)
        ax1.set_xlabel('Time [s]').set_fontsize(15)
        ax1.set_xlim([times[0], times[-1]])
        ax1.set_yscale('log')

        # Save figure
        if filename is not None:
            plt.savefig(filename+'.pdf')

        plt.close()

        return

    def plot_closed_loop_eigs(self, times, eigvals, filename=None):
        '''
        Plots the evolution of the real part of the closed-loop eigenvalues over time

        :param times: Array of times (seconds)
        :param eigvals: Closed-loop eigenvalues (real part)
        :param n_samples: Number of samples collected for a sample update
        :param filename: The name of the file containing the plot in pdf format
        :return: None
        '''

        # Create figure
        fig = plt.figure(figsize=(11, 8))
        ax1 = fig.add_subplot(1, 1, 1)

        # Plotting grid
        ax1.grid(b=True, color='#888888', which='major')
        ax1.grid(b=True, color='#888888', which='minor', linewidth=0.2)

        # Plot zero line
        ax1.plot([times[0], times[-1]], [0,0], c='k', lw=1)

        for i in range(eigvals.shape[0]):
            ax1.plot(times, eigvals[i,:], lw=1)

        # Plot characteristics of ax1
        ax1.set_title('Real Part of Closed-Loop Eigenvalues vs Time').set_fontsize(20)
        ax1.set_ylabel(r'Re($\lambda_{CL}$) [-]').set_fontsize(15)
        ax1.set_xlabel('Time [s]').set_fontsize(15)
        ax1.set_xlim([times[0], times[-1]])
        ax1.set_yscale('symlog')

        # Save figure
        if filename is not None:
            plt.savefig(filename+'.pdf')

        plt.close()

        return

    def plot_RMSE_evolution(self, times, RMSE_data, RMSE_limit, filename=None):
        '''

        :param times:
        :param RMSE_data:
        :param filename:
        :return:
        '''

        # Create figure
        fig = plt.figure(figsize=(11, 5))
        ax1 = fig.add_subplot(1, 1, 1)

        # Plot grid
        ax1.grid(b=True, color='#888888', which='major', linewidth=0.2)
        ax1.grid(b=True, color='#888888', which='minor', linewidth=0.2)

        # Plot RMSE vs time
        ax1.plot(times, RMSE_data, lw=0.8, label='RMSE')

        # Plot RMSE limit
        ax1.plot([times[0], times[-1]], [RMSE_limit, RMSE_limit], lw=1, ls='--', c='k', label='RMSE limit')

        # Plot Characteristics of ax1
        ax1.set_title('RMSE vs Time').set_fontsize(20)
        ax1.set_ylabel('RMSE [-]').set_fontsize(15)
        ax1.set_xlabel('Time [s]').set_fontsize(15)
        ax1.set_xlim([times[0], times[-1]])
        ax1.set_yscale('log')
        ax1.legend(loc='upper left', fontsize=15, bbox_to_anchor=(1.01, 1.03))

        # Figure padding
        plt.subplots_adjust(left=0.07, bottom=0.1, right=0.8, top=0.9, wspace=0.1, hspace=0.4)

        # Save figure
        if filename is not None:
            plt.savefig(filename+'.pdf')

        plt.close()

        return

