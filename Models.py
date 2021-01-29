import numpy as np
import Interval_arithmetic as itrv_arith



class SS_model(object):
    def __init__(self, n, m, p):

        # Save state space dimensions
        self.n = n  # state vector dimensions (n x 1)
        self.m = m  # input vector dimensions (m x 1)
        self.p = p  # reference state vector dimensions (p x 1)

        # Initialise state space matrices
        self.T_d = np.zeros((self.n+self.p, self.n+self.p))
        self.B1_d = np.zeros((self.n+self.p, self.m))

        # Initialise initial values for uncertainties of T and B1
        self.unc_T = None
        self.unc_B1 = None

        # Initialise state and input vectors
        self.state = np.zeros((self.n,1))
        self.action = np.zeros((self.n,1))

        # Initialise interval formulations
        self.interval_T_d = None
        self.interval_B1_d = None

        return

    def create_interval_formulation(self, unc_type='absolute'):
        '''
        Construct uncertain state space matrices using the interval format used in this code.

        :param unc_type: type of uncertainty provided. Can be 'absolute' or 'relative'
        :return: None
        '''

        self.interval_T_d = itrv_arith.construct_uncertainty_matrix(self.T_d, self.unc_T, uncertainty_type=unc_type)
        self.interval_B1_d = itrv_arith.construct_uncertainty_matrix(self.B1_d, self.unc_B1, uncertainty_type=unc_type)

        return

    def propagate_state(self, interval_X, interval_u):
        '''
        Propagates augmented state through uncertain state space.

        :param interval_X: augmented state (in interval vector format)
        :param interval_u: input (in interval vector format)

        :return interval_x_new: state at next time step (in interval vector format)
        '''

        # Initial check for dimensions
        if int(self.interval_T_d.shape[1]/2) != interval_X.shape[0] or self.interval_T_d.shape[0] != interval_X.shape[0]:
            raise ValueError('Interval matrix SS_T and interval vector X have inconsistent dimensions')

        if self.interval_T_d.shape[0] != self.interval_B1_d.shape[0]:
            raise ValueError('Interval matrix SS_T and interval matrix SS_B1 have inconsistent dimensions')

        if int(self.interval_B1_d.shape[1]/2) != interval_u.shape[0]:
            raise ValueError('Interval matrix SS_B and interval vector u have inconsistent dimensions')

        # Compute multiplication of interval matrix A and interval state vector x (i.e. A*x)
        new_state_interval = itrv_arith.interval_matmul(self.interval_T_d, interval_X)

        # Compute multiplication of interval matrix B and interval input vector u (i.e. B*u)
        new_input_interval = itrv_arith.interval_matmul(self.interval_B1_d, interval_u)

        # Compute new state interval vector x_new (i.e. x_new = A*x + B*u)
        interval_X_new = new_state_interval + new_input_interval

        return interval_X_new