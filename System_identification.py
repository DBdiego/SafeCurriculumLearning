import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg


class SystemIdentificationSS(object):
    '''
    Class containing all the necessary functionalities for identifying systems and providing a
    parametric absolute uncertainty of the identified system. The identification uses linear OLS as
    the estimator.
    '''

    def __init__(self, interaction_dimensions, configuration):

        # Booleans
        self.use_sliding_window = configuration.use_sliding_window

        # Allocate input dimensions to class attributes
        [self.n, self.m, self.p] = interaction_dimensions

        # Define number of samples required before identification
        min_num_samples = self.p*self.p + self.n*(self.n + self.m)
        self.num_samples = int(configuration.sample_factor * min_num_samples)

        # Booleans
        self.SS_estimate_available = False

        # Defining system matrices
        self.T  = np.zeros(((self.n+self.p), (self.n+self.p)))
        self.B1 = np.zeros(((self.n+self.p), self.m))

        # Define matrix uncertainties (relative uncertainties)
        self.T_unc = np.zeros(((self.n+self.p), (self.n+self.p)))
        self.B1_unc = np.zeros(((self.n+self.p), self.m))

        # Sampling variables ("..._k_1" stands for time step (k-1) )
        self.X_k_1 = None
        self.u_k_1 = None
        self.u_k_1_excite = None
        self.X_k = None
        self.u_k = None
        self.u_k_excite = None

        # Defining regression matrix and vector
        self.regression_matrix = np.zeros((0, self.n+self.m+self.p))
        self.regression_vector = np.zeros((0, self.n+self.p))

        # Define attribute for 1D array with all the uncertainties
        self.uncertainties = None
        self.uncertainty_hist = np.zeros((0,min_num_samples))

        return

    def uniform_distr_uncertainty(self, variance):
        '''
        Derive absolute uncertainty from parametric variance for a continuous uniform distribution

        :param variance: element-wise parametric variance
        :return: absolute_uncertainty, the element-wise positive absolute uncertainty
        '''

        # Conversion to numpy array format
        variance = np.asarray(variance)

        # Compute absolute uncertainty
        absolute_uncertainty = np.sqrt(np.multiply(3, variance))

        return absolute_uncertainty

    def normal_distr_uncertainty_bounds(self, variance, std_factor):
        '''
        Derive absolute uncertainty from parametric variance for a continuous normal distribution

        :param variance: element-wise parametric variance
        :param std_factor: multiplier of the standar deviation (postivie int)
        :return: absolute_uncertainty, the element-wise positive absolute uncertainty
        '''

        # Conversion to numpy array format
        variance = np.asarray(variance)

        # Compute absolute uncertainty
        absolute_uncertainty = np.multiply(std_factor, variance)

        return absolute_uncertainty

    def calculate_parametric_variances(self, A, b, theta_transposed, cov_matrix):
        '''
        Computes uncertainty of the individual parameters

        :param A:
        :param b:
        :param theta_transposed:
        :param cov_matrix:
        :return: parametric_variance, the parametric variance of theta
        '''

        #Computing the error
        epsilon = np.matmul(A, theta_transposed) - b

        # Computing variance w.r.t. to each error vector (columns of epsilon)
        residual_variance = (epsilon.T * epsilon) / (epsilon.shape[0] - theta_transposed.size)

        # Compute covariance
        covariance = np.kron(residual_variance, cov_matrix)

        # Computing variance of each parameter
        parameter_variance = np.diag(covariance)

        # Shape parameter variances to the TB1 matrix shape
        parameter_variance = parameter_variance.reshape(theta_transposed.T.shape)

        return parameter_variance

    def Ordinary_Least_Squares(self, A, b):
        '''
        Executes an Ordinary Least Squares regression:
            Theta = (A^T*A)^(-1) * A^T * b

        Theta consists of all parameters to be estimated. The function further derives
        state space matrices A, B and F from Theta.

        :param A: regression matrix
        :param b: regression vector
        :return: theta, parametric_variances: the estimates along with their respective parametric variance
        '''

        # Compute normalising part
        covariance_matrix = np.linalg.inv(A.T * A)

        # Estimate parameters
        theta_transposed = covariance_matrix * A.T * b
        theta = theta_transposed.T

        # Compute the uncertainties of the parameters
        parametric_variances = self.calculate_parametric_variances(A, b, theta_transposed, covariance_matrix)

        return theta, parametric_variances

    def collect_samples(self, X_k, u_k, u_k_excite):
        '''
        Collects samples for a linear system identification regression using Ordinary Least Squares (OLS)

        :param X_k: Augmented state vector at time step k
        :param u_k: action vector at time step k
        :param u_k: action vector with excitation signal at time step k
        :return: None
        '''

        # Shift variables by on time step
        self.X_k_1 = self.X_k
        self.u_k_1 = self.u_k
        self.u_k_1_excite = self.u_k_excite

        # Assigning new values to current-time-step attributes
        self.X_k = X_k
        self.u_k = u_k
        self.u_k_excite = u_k_excite

        # Append to regression matrices and vectors if not the first time step (i.e., when k_1 are not None)
        if self.X_k_1 is not None:
            # Construct Z_k_1 (Z at time step (k-1) )
            Z_k_1_excite = np.asmatrix(np.vstack((self.X_k_1, self.u_k_1_excite)))

            # Complete the regression matrix with state measurements at current time step
            self.regression_matrix = np.vstack((self.regression_matrix, Z_k_1_excite.T))

            # Complete the regression vector with state measurements at next time step
            self.regression_vector = np.vstack((self.regression_vector, np.asmatrix(X_k).T))

        return

    def identify_system(self, method='OLS', uncertainty_method='uniform', uncertainty_type='absolute'):
        '''
        Identifies system dynamids, input dynamics and reference signal dynamics.

        :return: None
        '''

        # Signal estimate availability
        self.SS_estimate_available = True

        # Estimate parameters
        if method == 'OLS':
            # State Space estimation
            AB_regr_m = np.hstack((self.regression_matrix[:,:self.n], self.regression_matrix[:,-self.m:]))
            AB_regr_v = self.regression_vector[:,:self.n]
            AB, AB_param_variances = self.Ordinary_Least_Squares(AB_regr_m, AB_regr_v)

            # Reference estimation
            F_ref_regr_m = self.regression_matrix[:,self.n:(self.n+self.p)]
            F_ref_regr_v = self.regression_vector[:,-self.p:]

            # Add small amount of noise, due to the reference being deterministic
            # F_ref_regr_m += np.random.normal(0, 1e-1, F_ref_regr_m.shape)

            # Estimate F and variances
            F, F_param_variances = self.Ordinary_Least_Squares(F_ref_regr_m, F_ref_regr_v)

        else:
            raise ValueError('method "' + method + '" not implemented')

        # Construct TB1
        T = scipy.linalg.block_diag(AB[:self.n, :self.n], F)
        B1 = np.vstack((AB[:,-self.m:], np.zeros((self.p, self.m))))
        TB1 = np.hstack((T, B1))

        # Construct TB1 variances
        T_param_variances = scipy.linalg.block_diag(AB_param_variances[:self.n, :self.n], F_param_variances)
        B1_param_variances = np.vstack((AB_param_variances[:,-self.m:], np.zeros((self.p, self.m))))
        TB1_param_variances = np.hstack((T_param_variances, B1_param_variances))

        # Compute parameter uncertainties based on variances
        if uncertainty_method == 'uniform':
            abs_uncertainty = self.uniform_distr_uncertainty(TB1_param_variances)
        elif uncertainty_method == 'normal':
            abs_uncertainty = self.normal_distr_uncertainty_bounds(TB1_param_variances, std_factor=3)
        else:
            raise ValueError('uncertainty_method "' + uncertainty_method + '" not implemented.')

        # Define uncertainty in the desired type (relative or absolute)
        if uncertainty_type == 'absolute':
            uncertainties = abs_uncertainty
        elif uncertainty_type == 'relative':
            TB1_array = np.asarray(TB1)
            uncertainties = abs_uncertainty/TB1_array
        else:
            raise ValueError('Input uncertainty_type not valid. Should be "relative" or "absolute".')

        # Derive A, B and F from estimated parameters
        self.T  = TB1[:,:(self.n+self.p)]
        self.B1 = TB1[:,-self.m:]

        # Derive A, B and F uncertainties from estimated parametric uncertainties
        self.T_unc = uncertainties[:,:(self.n+self.p)]
        self.B1_unc = uncertainties[:,-self.m:]

        # Shape uncertainties to a 1 by ... array
        T_unc_1D = self.T_unc.reshape((1, self.T_unc.size))
        B1_unc_1D = self.B1_unc.reshape((1, self.B1_unc.size))
        self.uncertainties = np.hstack((T_unc_1D, B1_unc_1D))

        # Save uncertainties for plotting
        uncertainties_A = self.T_unc[:self.n, :self.n].reshape((1,(self.n * self.n)))
        uncertainties_F = self.T_unc[self.n:, self.n:].reshape((1,(self.p * self.p)))
        uncertainties_B = self.B1_unc[:self.n, :].reshape((1,(self.n * self.m)))
        model_uncertainties = np.hstack((uncertainties_A, uncertainties_B, uncertainties_F))
        self.uncertainty_hist = np.vstack((self.uncertainty_hist, model_uncertainties))

        return

    def generate_output(self, curricular_step, configuration):
        '''

        :param curricular_step:
        :param configuration:
        :return:
        '''


        if curricular_step.train_agent:
            parent_dir = curricular_step.folder_name

            # Plot uncertainties over time
            Unc_evo_filename = 'Uncertainty_evo_'+ curricular_step.name + curricular_step.stable_policy_found*'_converged'
            Unc_evo_directory = parent_dir + '/' + Unc_evo_filename
            self.plot_uncertainties_over_time(curricular_step.times,
                                              self.uncertainty_hist,
                                              filename=Unc_evo_directory)

        return

    def plot_uncertainties_over_time(self, times, uncertainties, filename=None):
        '''

        :param times:
        :param uncertainties:
        :return: None
        '''

        # Initiate uncertainty data array
        populated_uncertainties = np.zeros((0, uncertainties.shape[1]))

        if self.use_sliding_window:
            populated_uncertainties = np.copy(uncertainties)

        else:
            # Populate uncertainties to full time range
            for unc_row in range(uncertainties.shape[0]):
                # Populating update period with same uncertainties
                populated_row = np.tile(uncertainties[unc_row,:], (self.num_samples, 1))

                # Construction total data array
                populated_uncertainties = np.vstack((populated_uncertainties, populated_row))

            # Removing excess population at the end
            populated_uncertainties = populated_uncertainties[:times.shape[0]-self.num_samples,:]

        # Create figure
        fig = plt.figure(figsize=(15,11))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0,:])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[1,1])

        # Plot grid
        ax1.grid(b=True, color='#888888', which='major', linewidth=0.2)
        ax1.grid(b=True, color='#888888', which='minor', linewidth=0.2)

        ax2.grid(b=True, color='#888888', which='major', linewidth=0.2)
        ax2.grid(b=True, color='#888888', which='minor', linewidth=0.2)

        ax3.grid(b=True, color='#888888', which='major', linewidth=0.2)
        ax3.grid(b=True, color='#888888', which='minor', linewidth=0.2)

        # Plot uncertainty of diagonal elements from A
        diagonal_indices = np.identity(self.n).astype('bool')
        diagonal_indices = diagonal_indices.reshape((diagonal_indices.size,))
        A_diagonal_unc = populated_uncertainties[:,:self.n*self.n][:,diagonal_indices]
        for i in range(self.n):
            label = r'$a_{'+'{},{}'.format(i+1, i+1)+ '}$'
            ax1.plot(times[self.num_samples:], A_diagonal_unc[:,i], label=label)

        # Plot uncertainty of elements from B
        for i in range(self.n*self.m):
            B_row_ind = int(i/self.m)
            B_col_ind = int(i - B_row_ind*self.m)
            i = int(self.n*self.n) + i

            label = r'$b_{' + '{},{}'.format(B_row_ind+1, B_col_ind+1) + '}$'
            ax2.plot(times[self.num_samples:], populated_uncertainties[:,i], label=label)

        # Plot uncertainty of elements from F
        for i in range(self.p*self.p):
            F_row_ind = int(i/self.p)
            F_col_ind = int(i - F_row_ind*self.p)
            i = int((self.n*self.n)+(self.n*self.m)) + i

            label = r'$f_{' + '{},{}'.format(F_row_ind+1, F_col_ind+1) + '}$'
            ax3.plot(times[self.num_samples:], populated_uncertainties[:, i], label=label)

        # Plot Characteristics of ax1
        ax1.set_title('Uncertainty of A diagonal elements vs Time').set_fontsize(20)
        ax1.set_xlim([times[0], times[-1]])
        ax1.set_ylabel('Uncertainty [-]').set_fontsize(15)
        ax1.set_xlabel('Time [s]').set_fontsize(15)
        ax1.set_yscale('log')
        ax1.legend(loc='upper left', fontsize=15, bbox_to_anchor=(1.01, 1.03))

        # Plot Characteristics of ax2
        ax2.set_title('Uncertainty of B elements vs Time').set_fontsize(20)
        ax2.set_xlim([times[0], times[-1]])
        ax2.set_ylabel('Uncertainty [-]').set_fontsize(15)
        ax2.set_xlabel('Time [s]').set_fontsize(15)
        ax2.set_yscale('log')
        ax2.legend(loc='upper left', fontsize=15, bbox_to_anchor=(1.01, 1.03))

        # Plot Characteristics of ax3
        ax3.set_title('Uncertainty of F elements vs Time').set_fontsize(20)
        ax3.set_xlim([times[0], times[-1]])
        ax3.set_ylabel('Uncertainty [-]').set_fontsize(15)
        ax3.set_xlabel('Time [s]').set_fontsize(15)
        ax3.set_yscale('log')
        ax3.legend(loc='upper left', fontsize=15, bbox_to_anchor=(1.01, 1.03))

        # Figure padding
        plt.subplots_adjust(left=0.08, bottom=0.1, right=0.9, top=0.95, wspace=0.5, hspace=0.4)

        # Save figure
        if filename is not None:
            plt.savefig(filename + '.pdf')

        plt.close()

        return