import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class ProppellorArmAndMotor(object):
    def __init__(self, arm_length, angle_with_x, arm_mass, motor_mass):
        self.l = arm_length
        self.alpha = angle_with_x
        self.m_arm = arm_mass
        self.m_motor = motor_mass
        self.m = self.m_arm + self.m_motor

        self.l_x = np.cos(self.alpha) * self.l
        self.l_y = np.sin(self.alpha) * self.l

        # Compute moments of inertia
        self.compute_moments_of_inertias()

        # Compute centroid position
        self.compute_centroid_position()

    def compute_moments_of_inertias(self, print_inertias=False):
        '''
        Computes the mass moment of inertia of the arm w.r.t. geometric center of the body.

        :return: None
        '''

        I_xx_arm = self.m_arm / 2 * self.l_y ** 2
        I_yy_arm = self.m_arm / 2 * self.l_x ** 2
        I_zz_arm = self.m_arm / 2 * self.l ** 2

        I_xx_motor = self.m_motor * self.l_y ** 2
        I_yy_motor = self.m_motor * self.l_x ** 2
        I_zz_motor = self.m_motor * self.l ** 2

        self.I_xx = I_xx_arm + I_xx_motor
        self.I_yy = I_yy_arm + I_yy_motor
        self.I_zz = I_zz_arm + I_zz_motor

        if print_inertias:
            print('I_xx =', self.I_xx)
            print('I_yy =', self.I_yy)
            print('I_zz =', self.I_zz)

        return

    def compute_centroid_position(self, print_pos=False):
        '''
        Computes the centroid location w.r.t geometric center of body.

        :return: None
        '''

        # Centroid arm
        c_x_arm = self.l_x/2
        c_y_arm = self.l_y/2
        c_z_arm = 0

        # Centroid motor
        c_x_motor = self.l_x
        c_y_motor = self.l_y
        c_z_motor = 0

        # Centroid Arm-Motor Assembly
        self.c_x = (c_x_arm * self.m_arm) + (c_x_motor * self.m_motor)
        self.c_y = (c_y_arm * self.m_arm) + (c_y_motor * self.m_motor)
        self.c_z = (c_z_arm * self.m_arm) + (c_z_motor * self.m_motor)

        if print_pos:
            print('c_x =', self.c_x)
            print('c_y =', self.c_y)
            print('c_z =', self.c_z)

        return

class Body(object):
    '''
    Class containing attributes of the quadcopter's central body element (assumed to be a rectangular block)
    '''

    def __init__(self, h, w, l, m):
        # Geometric dimensions
        self.h = h
        self.w = w
        self.l = l

        # Body mass
        self.m = m

        # Initiating Moment of Inertia attributes
        self.I_xx = None
        self.I_yy = None
        self.I_zz = None

        # Initiating centroid location attributes
        self.c_x = None
        self.c_y = None
        self.c_z = None

        # Compute moments of inertia
        self.compute_moments_of_inertias()

        # Compute centroid position
        self.compute_centroid_position()

        return

    def compute_moments_of_inertias(self):
        '''
        Computes the mass moment of inertia of the central body w.r.t. geometric center of the body.

        :return: None
        '''

        self.I_xx = self.m / 12 * (self.w ** 2 + self.h ** 2)
        self.I_yy = self.m / 12 * (self.h ** 2 + self.l ** 2)
        self.I_zz = self.m / 12 * (self.l ** 2 + self.w ** 2)

        return

    def compute_centroid_position(self):
        '''
        Computes centroid location of the central body w.r.t. the geometric center of the body

        :return: None
        '''

        self.c_x = 0
        self.c_y = 0
        self.c_z = 0

        return

class QuadrotorSystem(object):
    '''
    Class defining the quad rotor system. The numbering of the arms has been done in a clockwise direction starting
    with the front right motor. Thus, the arm mapping is:
        arm 1: front right
        arm 2: rear right
        arm 3: rear left
        arm 4: front left
    '''

    def __init__(self, configuration):
        # System Name
        self.name = 'Quadrotor'

        self.curr_step_ind = configuration.curr_step_ind

        # Initialise state space system matrices
        self.A = None
        self.B = None
        self.C = configuration.C
        self.D = None

        # Initial condition bounds
        self.init_bounds = np.asmatrix(configuration.init_bounds)

        # Quadrotor body dimensions
        body_h = configuration.body_h  # [m]
        body_w = configuration.body_w  # [m]
        body_l = configuration.body_l  # [m]
        body_m = configuration.body_m  # [kg]

        # Propeller arm lengths (from geometric center)
        prop1_arm_l = configuration.prop1_arm_l  # [m]
        prop2_arm_l = configuration.prop2_arm_l  # [m]
        prop3_arm_l = configuration.prop3_arm_l  # [m]
        prop4_arm_l = configuration.prop4_arm_l  # [m]

        # Propeller arm angle with the x-axis (Vehicle carried frame of reference)
        prop1_arm_alpha = configuration.prop1_arm_alpha  # [rad]
        prop2_arm_alpha = configuration.prop2_arm_alpha  # [rad]
        prop3_arm_alpha = configuration.prop3_arm_alpha  # [rad]
        prop4_arm_alpha = configuration.prop4_arm_alpha  # [rad]

        # Propeller arm masses
        prop1_arm_m = configuration.prop1_arm_m  # [kg]
        prop2_arm_m = configuration.prop2_arm_m  # [kg]
        prop3_arm_m = configuration.prop3_arm_m  # [kg]
        prop4_arm_m = configuration.prop4_arm_m  # [kg]

        # Propeller motor masses
        prop1_motor_m = configuration.prop1_motor_m  # [kg]
        prop2_motor_m = configuration.prop2_motor_m  # [kg]
        prop3_motor_m = configuration.prop3_motor_m  # [kg]
        prop4_motor_m = configuration.prop4_motor_m  # [kg]

        # Gravitational acceleration
        self.g = configuration.g
        self.b = configuration.b
        self.d = configuration.d

        # Define the body characteristics of the quadrotor (assumed to be a rectangular block)
        self.body = Body(body_h, body_w, body_l, body_m)

        # Compute characteristics of each arm based on the configuration parameters given
        self.arm1 = ProppellorArmAndMotor(prop1_arm_l, prop1_arm_alpha, prop1_arm_m, prop1_motor_m)
        self.arm2 = ProppellorArmAndMotor(prop2_arm_l, prop2_arm_alpha, prop2_arm_m, prop2_motor_m)
        self.arm3 = ProppellorArmAndMotor(prop3_arm_l, prop3_arm_alpha, prop3_arm_m, prop3_motor_m)
        self.arm4 = ProppellorArmAndMotor(prop4_arm_l, prop4_arm_alpha, prop4_arm_m, prop4_motor_m)

        # Compute the total mass of the quadrotor
        self.m = self.body.m + self.arm1.m + self.arm2.m + self.arm3.m + self.arm4.m

        # Initiating Moment of Inertia attributes
        self.I_xx = 0
        self.I_yy = 0
        self.I_zz = 0

        # Compute moments of inertia
        self.compute_moments_of_inertias()

        # Initiating centroid location attributes
        self.c_x = 0
        self.c_y = 0
        self.c_z = 0

        # Compute centroid position
        self.compute_centroid_position()

        # Create continuous-time state space system
        self.state_space = self.create_SS_matrices()

        # Compute RPM for hovering (trim point of the model)
        self.hover_RPM = (self.m * self.g)/self.b

        return

    def compute_moments_of_inertias(self):
        '''
        Computes the mass moment of inertia of the quadcopter w.r.t. geometric center of the body.

        :return:
        '''

        self.I_xx = self.body.I_xx + self.arm1.I_xx + self.arm2.I_xx + self.arm3.I_xx + self.arm4.I_xx
        self.I_yy = self.body.I_yy + self.arm1.I_yy + self.arm2.I_yy + self.arm3.I_yy + self.arm4.I_yy
        self.I_zz = self.body.I_zz + self.arm1.I_zz + self.arm2.I_zz + self.arm3.I_zz + self.arm4.I_zz

        # print('I_xx =', self.I_xx)
        # print('I_yy =', self.I_yy)
        # print('I_zz =', self.I_zz)

        return

    def compute_centroid_position(self):
        '''
        Computes position of the centroid relative to the geometric center of the rectangular center portion of the
        quadcopter.

        :return: None
        '''
        # Coordinates are computed relative to the geometric center of the body
        self.c_x = (self.body.c_x * self.body.m) + (self.arm1.c_x * self.arm1.m) + (self.arm2.c_x * self.arm2.m) + \
                   (self.arm3.c_x * self.arm3.m) + (self.arm4.c_x * self.arm4.m)

        self.c_y = (self.body.c_y * self.body.m) + (self.arm1.c_y * self.arm1.m) + (self.arm2.c_y * self.arm2.m) + \
                   (self.arm3.c_y * self.arm3.m) + (self.arm4.c_y * self.arm4.m)

        self.c_z = (self.body.c_z * self.body.m) + (self.arm1.c_z * self.arm1.m) + (self.arm2.c_z * self.arm2.m) + \
                   (self.arm3.c_z * self.arm3.m) + (self.arm4.c_z * self.arm4.m)

        # print('c_x =', self.c_x)
        # print('c_y =', self.c_y)
        # print('c_z =', self.c_z)

        return

    def get_initial_conditions(self):
        '''
        Defining initial state of the quadrotor

        :return: state_init, the initial state of the system
        '''

        # Generate random initial conditions for position and velocity of all masses
        random_numbers = np.random.random((self.init_bounds.shape[0],1))
        state_init = self.init_bounds[:,0] + np.multiply(random_numbers, np.diff(self.init_bounds, axis=1))

        print('\t\tInitial Conditions:')
        print('\t\t\t', state_init.T)

        return state_init

    def create_SS_matrices(self):
        '''
        The state space is created with the state having the following order:
            x = [phi, theta, psi, p, q, r, u, v, w, x, y, z]^T

        The input vector contains the RPM of each motor, which have been assumed to be related in a linear
        fashion to thrust (with factor b) and torque (with factor d).

        :return: state space
        '''

        # A rows for phi, theta and psi
        A_phi_theta_psi_rows = np.hstack((np.zeros((3, 3)), np.identity(3), np.zeros((3, 6))))

        # A rows for p, q, r
        A_pqr_rows = np.zeros((3, 12))

        # A rows for u, v, w
        A_uvw_rows = np.zeros((3, 12))
        A_uvw_rows[0, 1] = -self.g
        A_uvw_rows[1, 0] = self.g

        # A rows for x, y, z
        A_xyz_rows = np.hstack((np.zeros((3, 6)), np.identity(3), np.zeros((3, 3))))

        # Construct A
        A = np.vstack((A_phi_theta_psi_rows, A_pqr_rows, A_uvw_rows, A_xyz_rows))

        # Input forces and moments
        if self.curr_step_ind in [0, 1, 2, 3]:
            # Create B matrix for F, tau_x, tau_y, tau_z as input
            B = np.zeros((12, 4))
            B[3, 1] = 1 / self.I_xx
            B[4, 2] = 1 / self.I_yy
            B[5, 3] = 1 / self.I_zz
            B[8, 0] = -1 / self.m

            # Construct mask for A
            A_mask = np.ones(A.shape).astype('bool')
            A_mask[6: 8,:] = 0
            A_mask[9:11,:] = 0
            A_mask[:,6: 8] = 0
            A_mask[:,9:11] = 0

            # Construct mask for B
            B_mask = np.ones(B.shape).astype('bool')
            B_mask[6: 8,:] = 0
            B_mask[9:11,:] = 0

            self.A = A[A_mask].reshape((8,8))
            self.B = B[B_mask].reshape((8,4))

        # Switch to rotor RPMs
        elif self.curr_step_ind == 4:
            # Create constants for the B matrix
            b_over_Ixx = self.b / self.I_xx
            b_over_Iyy = self.b / self.I_yy
            d_over_Izz = self.d / self.I_zz
            b_over_m = self.b / self.m

            # Create B matrix for rotor RPMs as input vector
            B = np.zeros((12, 4))
            B[3,:] = np.array([-b_over_Ixx * self.arm1.l_y, -b_over_Ixx * self.arm2.l_y,
                               -b_over_Ixx * self.arm3.l_y, -b_over_Ixx * self.arm4.l_y])

            B[4,:] = np.array([b_over_Iyy * self.arm1.l_x, b_over_Iyy * self.arm2.l_x,
                               b_over_Iyy * self.arm3.l_x, b_over_Iyy * self.arm4.l_x])

            B[5,:] = np.array([d_over_Izz, -d_over_Izz, d_over_Izz, -d_over_Izz])
            B[8,:] = np.array([-b_over_m, -b_over_m, -b_over_m, -b_over_m])

            # Construct mask for A
            A_mask = np.ones(A.shape).astype('bool')
            A_mask[6: 8,:] = 0
            A_mask[9:11,:] = 0
            A_mask[:,6: 8] = 0
            A_mask[:,9:11] = 0

            # Construct mask for B
            B_mask = np.ones(B.shape).astype('bool')
            B_mask[6:8,:] = 0
            B_mask[9:11,:] = 0

            self.A = A[A_mask].reshape((8,8))
            self.B = B[B_mask].reshape((8,4))

        else:
            raise ValueError('No SS implemented for curricular step {}'.format(self.curr_step_ind+1))

        # Convert A and B arrays to matrices
        self.A = np.asmatrix(self.A)
        self.B = np.asmatrix(self.B)

        # Create C
        self.C = np.asmatrix(self.C)

        ## Create D matrix
        D = np.zeros((self.C.shape[0],self.B.shape[1]))
        self.D = np.asmatrix(D)


        self.print_system_info()

        state_space = [self.A, self.B, self.C, self.D]

        return state_space

    def print_system_info(self):
        print('\t\tSystem Info:')
        print('\t\t\t Quadrotor mass: {:.2f} kg'.format(self.m))
        print('\t\t\t Ixx = {:.8f}'.format(self.I_xx))
        print('\t\t\t Iyy = {:.8f}'.format(self.I_yy))
        print('\t\t\t Izz = {:.8f}'.format(self.I_zz))

        return

    def generate_output(self, curricular_step, configuration):
        '''

        :param curricular_step:
        :param configuration:
        :return:
        '''

        parent_dir = curricular_step.folder_name

        if curricular_step.train_agent:
            n_samples = curricular_step.agent.num_samples
        else:
            n_samples = None


        FSS_bounds = np.array([[-30, 30],
                               [-30, 30],
                               [0, 0],
                               [0, 0],
                               [0, 0],
                               [0, 0],
                               [0, 0],
                               [0, 0],
                               [0, 0],
                               [0, 0],
                               [0, 0],
                               [0, 0]])

        state_evo_filename = 'State_evo_' + curricular_step.name + curricular_step.stable_policy_found * '_converged'
        state_evo_directory =  parent_dir + '/' + state_evo_filename

        self.plot_state_evolution(curricular_step.times,
                                  curricular_step.state_storage,
                                  curricular_step.action_storage,
                                  curricular_step.env.C1,
                                  configuration.curr_step_ind,
                                  n_samples,
                                  action_bounds=curricular_step.agent.action_bounds,
                                  FSS_bounds=FSS_bounds,
                                  filename=state_evo_directory)

        # Plotting final policy values
        final_policy_filename = 'Final_policy_params_'+ curricular_step.name \
                                + curricular_step.stable_policy_found*'_converged'
        final_policy_directory = parent_dir + '/' + final_policy_filename

        agent = curricular_step.agent
        full_policy = np.zeros((agent.action_mask.shape[0], agent.policy_parameters.shape[1]))

        full_policy[agent.action_mask,:] = agent.policy_parameters

        if curricular_step.supervisor is not None:
            supervisor = curricular_step.supervisor

            full_policy[supervisor.action_mask,:] = supervisor.K



        self.plot_final_policy_values(full_policy,
                                      curricular_step.ind,
                                      filename=final_policy_directory)

        return


    def plot_state_evolution(self, times, states, actions, C1, curr_step_ind, n_samples,
                             action_bounds=None, FSS_bounds=None, filename=None):
        '''

        :param times:
        :param states:
        :param filename:
        :return:
        '''

        RSS = np.array([0, 1])

        if curr_step_ind in [0, 1, 2, 3, 4]:
            # plot_map = np.array([1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1]).astype('bool')
            plot_map = np.array([1, 1, 1, 1, 1, 1, 1, 1]).astype('bool') #, 0, 0, 0, 0]).astype('bool')
            state_map = np.array([0, 1, 2, 7, 3, 4, 5, 6])
            input_map = np.array([1, 1, 1, 1]).astype('bool')

        else:
            raise ValueError('No state and input maps implemented for curricular step {}'.format(curr_step_ind))




        ########################
        #        STATES        #
        ########################
        fontsize_states = 20
        # Define bounds of each plot
        plot_bounds = np.array([[-60, 60], # roll
                                [-60, 60], # pitch
                                [-100, 100], # yaw
                                [-100, 100], # roll rate
                                [-100, 100], # pitch rate
                                [-360, 360], # yaw rate
                                # [-5, 5], # u
                                # [-5, 5], # v
                                [-5, 5], # w
                                # [-20, 20], # x
                                # [-25, 25], # y
                                [-7, 7]]) # z

        # Define labels of each state plot
        labels = np.array([[r'$\phi_{ref}$'        , r'$\phi$'        , r'Roll $\phi$ [degs]'                ],
                           [r'$\theta_{ref}$'      , r'$\theta$'      , r'Pitch $\theta$ [degs]'             ],
                           [r'$\psi_{ref}$'        , r'$\psi$'        , r'Yaw $\psi$ [degs]'                 ],
                           [r'$\dot{\phi}_{ref}$'  , r'$\dot{\phi}$'  , r'Roll rate $\dot{\phi}$'+'\n [degs/s]'   ],
                           [r'$\dot{\theta}_{ref}$', r'$\dot{\theta}$', r'Pitch rate $\dot{\theta}$'+'\n [degs/s]'],
                           [r'$\dot{\psi}_{ref}$'  , r'$\dot{\psi}$'  , r'Yaw rate $\dot{\psi}$'+'\n [degs/s]'    ],
                           # [r'$u_{ref}$'           , 'u'              , 'u [m/s]'                            ],
                           # [r'$v_{ref}$'           , 'v'              , 'v [m/s]'                            ],
                           [r'$w_{ref}$'           , 'w'              , 'w [m/s]'                            ],
                           # [r'$x_{ref}$'           , 'x'              , 'x [m]'                              ],
                           # [r'$y_{ref}$'           , 'y'              , 'y [m]'                              ],
                           [r'$z_{ref}$'           , 'z'              , 'z [m]'                              ]])

        # Define filenames of figures containing state evolutions
        filenames = [filename + '_phi_theta_psi_z.pdf',
                     filename + '_phi_theta_psi_dot_w.pdf',
                     filename + '_u_v_w.pdf',
                     filename + '_x_y_z.pdf']


        plot_bounds = plot_bounds[plot_map,:]
        labels = labels[plot_map, :]

        # Separate C and reference_map from C1
        C = C1[:, :self.C.shape[1]]
        ref_map = C1[:, self.C.shape[1]:]

        # Derive times at which kernel matrix is updated
        if n_samples is not None:
            H_update_times = []
            for i in range(int(times.shape[0]/n_samples)):
                t = times[int((i+1)*n_samples)]
                H_update_times.append([t,t])

        # Plot state evolution
        k = 0
        for i in range(2):
            if np.sum(plot_map[i*4:(i+1)*4]) > 0:
                fig = plt.figure(figsize=(14, 14))

                # Create sub plots in figure
                for j in range(4):

                    # Define state index to plot
                    plot_index = int((i*4)+j)

                    if plot_map[plot_index]:
                        # Create plot
                        ax = fig.add_subplot(4, 1, int(j+1))

                        state_index = state_map[k]
                        k += 1

                        # Plot grid
                        ax.grid(b=True, color='#888888', which='major', linewidth=0.2)
                        ax.grid(b=True, color='#888888', which='minor', linewidth=0.2)

                        # Plot kernel matrix update times
                        if n_samples is not None:
                            H_update_label = r'$H^{j} \leftarrow H^{j+1}$'
                            for H_update_time in H_update_times:
                                ax.plot(H_update_time, plot_bounds[state_index,:],
                                        c='k', ls='-.', lw='1.5', label=H_update_label)
                                H_update_label = None

                        # Plot FSS bounds
                        if (state_index in RSS) and (FSS_bounds is not None):
                            # Add lower and upper FSS rectangles
                            rect_lower_FSS = Rectangle((times[0], FSS_bounds[state_index,0]),
                                                       (times[-1] - times[0]),
                                                       FSS_bounds[state_index,0]*1.5,
                                                       lw=1, ec='none', fc='r', alpha=0.5, label='FSS')
                            rect_upper_FSS = Rectangle((times[0], FSS_bounds[state_index,1]),
                                                       (times[-1] - times[0]),
                                                       FSS_bounds[state_index,1]*1.5,
                                                       lw=1, ec='r', fc='r', alpha=0.5)
                            ax.add_patch(rect_lower_FSS)
                            ax.add_patch(rect_upper_FSS)

                        # Plot reference signal
                        if np.sum(C[:,state_index]) > 0:
                            state_row = self.ref_index_to_plot(C, ref_map, state_index)
                            if state_row is not None:
                                ref_time_series = states[state_row,:]

                                # Convert rads to degrees for attitude states
                                if plot_index in [0,1,2,4,5,6]:
                                    ref_time_series = np.degrees(ref_time_series)

                                ax.plot(times, ref_time_series, c='k', ls='--', lw=1.5, label=labels[state_index][0])
                            else:
                                ax.plot([times[0], times[-1]], [0, 0], c='k', ls='--', lw=1.5, label=labels[state_index][0])

                        state_time_series = states[state_index, :]
                        if plot_index in [0,1,2,4,5,6]:
                            state_time_series = np.degrees(state_time_series)

                        # Plot state propagation
                        ax.plot(times, state_time_series, lw=1.8, label=labels[state_index][1])

                        # Plot Characteristics
                        ax.set_xlabel('Time [s]').set_fontsize(fontsize_states)
                        ax.set_ylabel(labels[state_index][2]).set_fontsize(fontsize_states)
                        ax.set_xlim([times[0], times[-1]])
                        ax.set_ylim(plot_bounds[state_index,:])

                        # Invert y axis for vertical position and velocity
                        if state_index in [8,11]:
                            ax.invert_yaxis()

                        ax.tick_params(axis='both', labelsize=fontsize_states)
                        ax.legend(loc='upper left', fontsize=fontsize_states, bbox_to_anchor=(1.01, 1.02))

                # Figure padding
                fig.subplots_adjust(left=0.12, bottom=0.07, right=0.8, top=0.97, wspace=0.1, hspace=0.4)

                # Save figure
                if filename is not None:
                    fig.savefig(filenames[i])

                plt.close(fig)


        ##################
        #     INPUTS     #
        ##################
        fontsize_inputs = 20

        # Define input labels
        if curr_step_ind in [0,1,2,3]:
            input_labels = np.array([[r'$f_{t}$'   , r'$f_{t}$ [N]'    ],
                                     [r'$\tau_{x}$', r'$\tau_{x}$ [Nm]'],
                                     [r'$\tau_{y}$', r'$\tau_{y}$ [Nm]'],
                                     [r'$\tau_{y}$', r'$\tau_{z}$ [Nm]']])
            plot_name = filename + '_f_tau_xyz.pdf'
            lw = 0.4
        else:
            input_labels = np.array([[r'$\Omega_{1}$', r'$\Omega_{1}$ [RPM]'],
                                     [r'$\Omega_{2}$', r'$\Omega_{2}$ [RPM]'],
                                     [r'$\Omega_{3}$', r'$\Omega_{3}$ [RPM]'],
                                     [r'$\Omega_{4}$', r'$\Omega_{4}$ [RPM]']])
            plot_name = filename + '_rotor_RPMs.pdf'

            # Correction RPMS to be around hover point
            actions = self.hover_RPM + actions

            # correct action bounds
            if action_bounds is not None:
                action_bounds = self.hover_RPM + action_bounds

            lw = 1.0

        input_labels = input_labels[input_map]

        # y-limit for plots
        if action_bounds is None:
            min_action_value = np.min(actions, axis=1, keepdims=True)
            max_action_value = np.max(actions, axis=1, keepdims=True)
            center_y_range = np.mean(np.hstack((min_action_value, max_action_value)), axis=1, keepdims=True)
            y_range = np.diff(np.hstack((min_action_value, max_action_value)), axis=1) * 1.5
            ax_ylims = np.hstack((center_y_range-y_range/2, center_y_range+y_range/2))
        else:

            if np.all(action_bounds[:,0] < 0):
                ax_ylims = action_bounds * 1.5
            else:
                ax_ylims = np.copy(action_bounds)
                value_to_add_or_substract = np.mean(ax_ylims, axis=1) * 0.15
                ax_ylims[:,0] -= value_to_add_or_substract
                ax_ylims[:,1] += value_to_add_or_substract


        # Plot Input evolution
        fig = plt.figure(figsize=(14, 12))
        for i in range(int(np.sum(input_map))):
            ax = fig.add_subplot(np.sum(input_map), 1, int(i+1))

            # Plot grid
            ax.grid(b=True, color='#888888', which='major', linewidth=0.2)
            ax.grid(b=True, color='#888888', which='minor', linewidth=0.2)

            # Plot kernel matrix update times
            if n_samples is not None:
                H_update_label = r'$H^{j} \leftarrow H^{j+1}$'
                for H_update_time in H_update_times:
                    ax.plot(H_update_time, ax_ylims[i,:], c='k', ls='-.', lw='1.5', label=H_update_label)
                    H_update_label = None

            # Plot action bounds
            if action_bounds is not None:
                ax.plot([times[0], times[-1]], [action_bounds[i,0], action_bounds[i,0]], c='k', lw=1, ls='--')
                ax.plot([times[0], times[-1]], [action_bounds[i,1], action_bounds[i,1]], c='k', lw=1, ls='--')

            # plot action evolution
            ax.plot(times, actions[i,:], lw=lw, label=input_labels[i][0])

            # Plot characteristics
            ax.set_xlabel('Time [s]').set_fontsize(fontsize_inputs)
            ax.set_ylabel(input_labels[i][1]).set_fontsize(fontsize_inputs)
            ax.set_xlim([times[0], times[-1]])
            ax.set_ylim(ax_ylims[i,:])

            ax.tick_params(axis='both', labelsize=fontsize_inputs)

            ax.legend(loc='upper left', fontsize=fontsize_inputs, bbox_to_anchor=(1.01, 1.02))

        # Figure padding
        fig.subplots_adjust(left=0.1, bottom=0.07, right=0.8, top=0.97, wspace=0.1, hspace=0.4)

        # Save figure
        if filename is not None:
            fig.savefig(plot_name)

        plt.close(fig)

        return


    def plot_final_policy_values(self, policy_parameters, curr_step_ind, filename=None):
        '''

        :return:
        '''

        # Define color map
        color_map = 'PuOr'
        fontsize = 20


        states = [r'$\phi$', r'$\theta$', r'$\psi$', r'$\dot{\phi}$', r'$\dot{\theta}$', r'$\dot{\psi}$', r'w', r'z']
        if curr_step_ind == 0:
            references = [r'$\psi^{r}$', r'$\dot{\psi}^{r}$']
            inputs = [r'$f^{*}_{t}$', r'$\tau^{*}_{x}$', r'$\tau^{*}_{y}$', r'$\tau_{z}$']

        elif curr_step_ind == 1:
            references = [r'$\psi^{r}$', r'$\dot{\psi}^{r}$', r'$\phi^{r}$', r'$\dot{\phi}^{r}$']
            inputs = [r'$f^{*}_{t}$', r'$\tau_{x}$', r'$\tau^{*}_{y}$', r'$\tau_{z}$']

        elif curr_step_ind == 2:
            references = [r'$\psi^{r}$', r'$\dot{\psi}^{r}$', r'$\phi^{r}$', r'$\dot{\phi}^{r}$', r'$\theta^{r}$',
                          r'$\dot{\theta}^{r}$']
            inputs = [r'$f^{*}_{t}$', r'$\tau_{x}$', r'$\tau_{y}$', r'$\tau_{z}$']
            fontsize = 30

        elif curr_step_ind == 3:
            references = [r'$\psi^{r}$', r'$\dot{\psi}^{r}$', r'$\phi^{r}$', r'$\dot{\phi}^{r}$', r'$\theta^{r}$',
                          r'$\dot{\theta}^{r}$', r'$z^{r}$', r'$w^{r}$']
            inputs = [r'$f_{t}$', r'$\tau_{x}$', r'$\tau_{y}$', r'$\tau_{z}$']
            fontsize = 30

        elif curr_step_ind == 4:
            references = [r'$\psi^{r}$', r'$\dot{\psi}^{r}$', r'$\phi^{r}$', r'$\dot{\phi}^{r}$', r'$\theta^{r}$',
                          r'$\dot{\theta}^{r}$', r'$z^{r}$', r'$w^{r}$']
            inputs = [r'$\Omega_{1}$', r'$\Omega_{2}$', r'$\Omega_{3}$', r'$\Omega_{4}$']

        else:
            raise ValueError('The "references" and "inputs" variables are not defined for curr_step_ind {}'.format(curr_step_ind))

        # Convert policy params to their log value, whilst keeping the sign
        use_log_scale = False
        if use_log_scale:
            sign_array = np.ones(policy_parameters.shape)
            sign_array[policy_parameters < 0] = -1

            policy_parameters = np.log(np.abs(policy_parameters))
            policy_parameters = np.multiply(sign_array, policy_parameters)

        # Construct the tick label lists
        x_tick_labels = states + references
        y_tick_labels = inputs

        # Create figure
        fig = plt.figure(figsize=(policy_parameters.shape[1], 6))
        ax1 = fig.add_subplot(1,1,1)



        # Plot gray scale "heat map" of policy matrix
        im1 = ax1.imshow(policy_parameters, cmap=color_map)

        # Move x-axis tick labels to the top
        ax1.xaxis.tick_top()

        # Define the tick positions
        ax1.set_xticks(np.arange(policy_parameters.shape[1]).astype('int'))
        ax1.set_yticks(np.arange(policy_parameters.shape[0]).astype('int'))

        # Define the minor tick positions and draw according grid
        ax1.set_xticks(np.arange(-0.5, policy_parameters.shape[1], 1), minor=True)
        ax1.set_yticks(np.arange(-0.5, policy_parameters.shape[0], 1), minor=True)
        ax1.grid(which='minor', color='k', linestyle='-', linewidth=1)

        # Assign the new labels to the tick positions set earlier
        ax1.set_xticklabels(x_tick_labels[:policy_parameters.shape[1]])
        ax1.set_yticklabels(y_tick_labels[:policy_parameters.shape[0]])

        # Set font size and other tick label parameters
        ax1.tick_params(axis='both', labelsize=fontsize, pad=5)

        # Add color bar (as legend)
        cax = fig.add_axes([0.05, 0.15, 0.8, 0.05])
        cax.tick_params(labelsize=fontsize)
        fig.colorbar(im1, cax=cax, orientation='horizontal')

        # Figure padding
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.85)

        # Saving plot
        if filename is not None:
            plt.savefig(filename+'.pdf')

        plt.close()

        return

    def ref_index_to_plot(self, C, ref_map, ind):
        '''

        :param C:
        :param ref_map:
        :param ind:
        :return:
        '''

        # Finding the row of the reference map that corresponds to the examined state
        ref_row = np.array(ref_map)[list(C[:,ind]).index(1),:]

        # Find index of reference state to take
        if np.any(ref_row == -1):
            ref_col_ind = list(ref_row).index(-1)
            state_row = -ref_map.shape[1] + ref_col_ind
        else:
            state_row = None

        return state_row

