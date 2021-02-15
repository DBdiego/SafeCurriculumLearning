import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

class N_MSD(object):
    '''
    Class defining the MSD system
    '''

    def __init__(self, configuration):
        self.name = 'MSD'

        # initialise system characteristics
        self.masses = configuration.masses
        self.all_k = configuration.all_k
        self.all_c = configuration.all_c
        self.n_masses = self.masses.shape[0]

        # Define initial condition boundaries
        self.init_bounds_x = configuration.init_bounds_x
        self.init_bounds_x_dot = configuration.init_bounds_x_dot

        # Initialise state space system matrices
        self.A = None
        self.B = None
        self.C = configuration.C
        self.D = None

        # Create continuous-time state space system
        self.state_space = self.create_SS_matrices()

        return

    def get_initial_conditions(self):
        '''
        Defining initial state of the system, namely position and velocity of each mass

        :return: state_init, the initial state of the system
        '''

        # Generate random initial conditions for position and velocity of all masses
        x0 = self.init_bounds_x[0] + np.random.random((self.n_masses,1))* np.diff(self.init_bounds_x)
        x0_d = self.init_bounds_x_dot[0] + np.random.random((self.n_masses,1)) * np.diff(self.init_bounds_x_dot)

        print('\t\tInitital Conditions:')
        print('\t\t\t position:', x0.T)
        print('\t\t\t velocity:', x0_d.T)

        state_init = np.vstack((x0, x0_d))

        return state_init

    def create_SS_matrices(self):
        '''
        Construction of the state space continuous time state space matrices

        :return: state_space, list of all state space matrices: A, B, C, D
        '''

        ## Create A matrix
        quadrant_shape = (self.n_masses, self.n_masses)
        A = np.zeros(quadrant_shape)
        A = np.hstack((A, np.identity(self.n_masses)))

        # Spring and Damper forces
        A_spr = np.zeros(quadrant_shape)
        A_dmp = np.zeros(quadrant_shape)
        for i in range(self.n_masses - 1):
            # Spring sub-matrix
            A_spr[i, i] = -(self.all_k[i] + self.all_k[i + 1]) / self.masses[i]
            A_spr[i, i + 1] = self.all_k[i + 1] / self.masses[i]
            A_spr[i + 1, i] = self.all_k[i + 1] / self.masses[i + 1]

            # Damper sub-matrix
            A_dmp[i, i] = -(self.all_c[i] + self.all_c[i + 1]) / self.masses[i]
            A_dmp[i, i + 1] = self.all_c[i + 1] / self.masses[i]
            A_dmp[i + 1, i] = self.all_c[i + 1] / self.masses[i + 1]

        A_spr[-1, -1] = -self.all_k[-1] / self.masses[-1]
        A_dmp[-1, -1] = -self.all_c[-1] / self.masses[-1]

        A_eom = np.hstack((A_spr, A_dmp))
        self.A = np.asmatrix(np.vstack((A, A_eom)))

        ## Create B matrix
        B = np.zeros(quadrant_shape)
        B = np.vstack((B, np.identity(self.n_masses) * 1 / self.masses))
        self.B = np.asmatrix(B)

        # Create C matrix
        C = np.hstack((np.identity(self.n_masses),
                       np.zeros((self.n_masses, self.n_masses))))

        self.C = np.asmatrix(C)

        ## Create D matrix
        D = np.zeros((C.shape[0], B.shape[1]))
        self.D = np.asmatrix(D)

        self.print_system_info()

        state_space = [self.A, self.B, self.C, self.D]

        return state_space


    def print_system_info(self):
        print('\t\tSystem Info:')
        print('\t\t\t # all_masses   : ' + str(self.masses))
        print('\t\t\t spring constant: ' + str(self.all_k))
        print('\t\t\t damper constant: ' + str(self.all_c))


    def create_file_name(self):
        filename = ''
        filename += str(self.n_masses) + 'MSD_'
        filename += 'k[' + ','.join([str(k) for k in self.all_k]) + ']_'
        filename += 'c[' + ','.join([str(c) for c in self.all_c]) + ']_'
        filename += 'm[' + ','.join([str(m) for m in self.masses]) + ']_'
        return filename


    def generate_output(self, curricular_step, configuration):
        num_masses = int(curricular_step.env.n/2)

        # Define FSS width if no safety filter is given
        if curricular_step.safety_filter is None:
            FSS_bounds = None
        else:
            FSS_bounds = curricular_step.safety_filter.FSS[num_masses-1,:]

        # Define y limits for position plot if any are given in the config
        if configuration.position_plot_bounds is not None:
            position_plot_bounds = configuration.position_plot_bounds
        else:
            position_plot_bounds = None

        # Plot static (non-animated) evolution of mass positions
        if configuration.plot_static:
            # Define directory of static plot
            static_plot_filename = 'Pos_evo_' + curricular_step.name + curricular_step.stable_policy_found * '_converged'
            static_plot_directory = curricular_step.folder_name + '/' + static_plot_filename

            # Generate the static plot
            self.N_MSD_static(curricular_step.times,
                              curricular_step.state_storage,
                              curricular_step.action_storage,
                              num_masses,
                              curricular_step.agent.num_samples,
                              action_bounds=curricular_step.agent.action_bounds,
                              FSS_bounds=FSS_bounds,
                              position_plot_bounds=position_plot_bounds,
                              filename=static_plot_directory)

        # Plot an animation of the evolution of the mass positions
        if 1 or configuration.plot_animation:
            # Define directory of animated plot
            ani_plot_filename = 'Pos_evo_' + curricular_step.name + curricular_step.stable_policy_found * '_converged'
            ani_plot_directory = curricular_step.folder_name + '/' + ani_plot_filename

            # Generate the plot
            ani = self.N_MSD_animation(curricular_step.times,
                                       curricular_step.state_storage,
                                       curricular_step.action_storage,
                                       int(curricular_step.env.n/2),
                                       FSS=None,
                                       main_title='',
                                       filename=ani_plot_directory)

        # Save system characteristics and initial conditions
        sys_char_filename = 'System_characteristics' + curricular_step.name + curricular_step.stable_policy_found * '_converged'
        sys_char_directory = curricular_step.folder_name + '/' + sys_char_filename

        self.save_system_characteristics(curricular_step.initial_conditions,
                                         filename=sys_char_directory)

        return

    def save_system_characteristics(self, initial_conditions, filename=''):
        '''

        :param initial_conditions:
        :param filename:
        :return:
        '''

        initial_positions = list(initial_conditions[:self.n_masses,0])
        initial_velocities = list(initial_conditions[self.n_masses:2*self.n_masses,0])

        string = ''

        string += 'System built:\n'
        string += '  number of masses: {:2d}'.format(self.n_masses) + '\n'
        string += '  m: ' + ', '.join(['{:2.1f}'.format(m) for m in self.masses]) + '\n'
        string += '  k: ' + ', '.join(['{:2.1f}'.format(k) for k in self.all_k]) + '\n'
        string += '  c: ' + ', '.join(['{:2.1f}'.format(c) for c in self.all_c]) + '\n\n'

        string += 'Initial conditions:\n'
        string += '  Initial position bounds: [' + ', '.join(['{:2.2}'.format(i) for i in self.init_bounds_x])+']\n'
        string += '  Initial velocity bounds: [' + ', '.join(['{:2.2}'.format(i) for i in self.init_bounds_x_dot])+']\n\n'

        string += '  Initial mass positions : [' + ', '.join(['{:2.10f}'.format(pos) for pos in initial_positions])+']\n'
        string += '  Initial mass velocities: [' + ', '.join(['{:2.10f}'.format(vel) for vel in initial_velocities])+']\n'

        f = open(filename+'.txt', 'w')
        f.write(string)
        f.close()

        return

    def N_MSD_static(self, times, all_states, all_actions, n_masses, n_samples, action_bounds=None,
                     FSS_bounds=None, position_plot_bounds=None, filename=None):
        '''

        :param times:
        :param all_states:
        :param all_actions:
        :param n_masses:
        :param n_samples:
        :param action_bounds:
        :param FSS_bounds:
        :param position_plot_bounds:
        :param filename:
        :return:
        '''

        # Window creation
        fig = plt.figure(figsize=(14, 8))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        # Plot y-limits of ax1
        if FSS_bounds is not None:
            ax1_ylims = FSS_bounds * 1.5

        elif position_plot_bounds is not None:
            ax1_ylims = position_plot_bounds * 1.5

        else:
            min_state_value = np.min(all_states[:n_masses,:])
            max_state_value = np.max(all_states[:n_masses,:])
            center_y_range = np.mean([min_state_value, max_state_value])
            y_range = np.diff([min_state_value, max_state_value])[0]*1.5
            ax1_ylims = np.array([center_y_range-y_range/2, center_y_range+y_range/2])

        # Plot y-limits of ax2
        if action_bounds is None:
            min_action_value = np.min(all_actions)
            max_action_value = np.max(all_actions)
            center_y_range = np.mean([min_action_value, max_action_value])
            y_range = np.diff([min_action_value, max_action_value])[0]*1.5
            ax2_ylims = np.array([center_y_range-y_range/2, center_y_range+y_range/2])
        else:
            ax2_ylims = np.array(action_bounds[0,:]) * 1.5

        # Drawing the grid lines
        ax1.grid(b=True, color='#888888', which='major', linewidth=0.2)
        ax1.grid(b=True, color='#888888', which='minor', linewidth=0.2)

        ax2.grid(b=True, color='#888888', which='major', linewidth=0.2)
        ax2.grid(b=True, color='#888888', which='minor', linewidth=0.2)

        # Add kernel matrix update times
        label = r'$H^{j} \leftarrow H^{j+1}$'
        for i in range(int(times.shape[0]/n_samples)):
            ind = (i+1)*n_samples

            if ind < times.shape[0]:
                ax1.plot([times[ind], times[ind]], ax1_ylims, c='k', ls='-.', lw='1.5', label=label)
                ax2.plot([times[ind], times[ind]], ax2_ylims, c='k', ls='-.', lw='1.5', label=label)

            label = None

        # Plot zeros line (for reference)
        ax1.plot([times[0], times[-1]], [0, 0], c='k', lw='0.7')
        ax2.plot([times[0], times[-1]], [0, 0], c='k', lw='0.7')

        # Plot SSS or FSS (when defined)
        if FSS_bounds is not None:
            # Add lower and upper FSS rectangles
            rect_lower_FSS = Rectangle((times[0], FSS_bounds[0]),
                                       (times[-1] - times[0]),
                                       FSS_bounds[0]*1.5, lw=1, ec='none', fc='r', alpha=0.5,
                                       label=r'FSS $x_{'+'{}'.format(n_masses)+'}$')
            rect_upper_FSS = Rectangle((times[0], FSS_bounds[1]),
                                       (times[-1] - times[0]),
                                       FSS_bounds[1]*1.5, lw=1, ec='r', fc='r', alpha=0.5)
            ax1.add_patch(rect_lower_FSS)
            ax1.add_patch(rect_upper_FSS)

        elif position_plot_bounds is not None:
            # Add SSS rectangle
            SSS = Rectangle((times[0], position_plot_bounds[0]),
                            (times[-1] - times[0]),
                            float(np.diff(position_plot_bounds)), color='green', alpha=0.2,
                            label=r'SSS $x_{'+'{}'.format(n_masses)+'}$')
            ax1.add_patch(SSS)

        # Plot reference position
        ax1.plot(times, all_states[-2,:], lw=0.7, ls='--', c='k', alpha=0.7, label=r'$x^{r}_{'+'{}'.format(n_masses)+'}$')

        # Plot control limits
        ax2.plot([times[0], times[-1]], [action_bounds[0,0], action_bounds[0,0]], ls='--', lw=1, c='k')
        ax2.plot([times[0], times[-1]], [action_bounds[0,1], action_bounds[0,1]], ls='--', lw=1, c='k')

        # Plot state and action evolution over time
        for i in range(n_masses):
            ax1.plot(times, all_states[i,:], lw=1 , label=r'$x_{:d}$'.format(i+1))
            ax2.plot(times, all_actions[-n_masses+i,:], lw=1, label=r'$u_{:d}$'.format(i+1))

        # Plot characteristics of ax1
        ax1.set_title('Mass Position vs Time').set_fontsize(20)
        ax1.set_xlim([times[0], times[-1]])
        ax1.set_ylim(ax1_ylims)
        ax1.set_xlabel('Time [s]').set_fontsize(15)
        ax1.set_ylabel('$x_{(•)} [m]$').set_fontsize(15)
        ax1.legend(loc='upper left', fontsize=15, bbox_to_anchor=(1.01, 1.03))

        # Plot characteristics of ax2
        ax2.set_title('Mass Control Force vs Time').set_fontsize(20)
        ax2.set_xlim([times[0], times[-1]])
        ax2.set_ylim(ax2_ylims)
        ax2.set_xlabel('Time [s]').set_fontsize(15)
        ax2.set_ylabel('Control Force [N]').set_fontsize(15)
        ax2.legend(loc='upper left', fontsize=15, bbox_to_anchor=(1.01, 1.03))

        # Figure padding
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.95, wspace=0.1, hspace=0.4)

        # Saving file
        if filename is not None:
            plt.savefig(filename+'.pdf')

        plt.close()

        return


    def N_MSD_animation(self, times, all_states, all_actions, n_masses, FSS=None, main_title='', filename=''):
        '''

        :param all_states:
        :param all_actions:
        :param n_masses:
        :param RSSS_w:
        :param main_title:
        :param filename:
        :return:
        '''

        spring_attach_rod_l = 0.03
        num_spring_coils = 20
        spring_h = 0.01
        damper_h = 0.01
        damper_w = 0.04
        dist_between_m = 0.25
        mass_w, mass_h = [0.05, 0.06]

        # Window creation
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)

        ax1.set_xlim([times[0],times[-1]])

        ax2.set_xlim([times[0],times[-1]])

        ax3.set_xlim([0,(n_masses+1)*0.25])
        ax3.set_ylim([0.0, 0.1])

        if FSS is not None:
            # Plot safe state space (SSS) region based on FSS
            raise ValueError('SSS Plotting not implemented')
            #
            # RSS_width = RSSS_w + mass_w
            # RSS_height = ax3.get_ylim()[1]
            # start_x_RSSS = n_masses * dist_between_m - RSS_width/2
            #
            # ax3.add_patch(Rectangle((start_x_RSSS, 0), RSS_width, RSS_height, color='green', alpha=0.3))

        lines = []
        ax1_lines = []
        ax2_lines = []
        mass_blocks = []
        springs = []
        dampers = []
        control_rods = []
        control_forces = []

        # Plotting the x-positions of the masses
        colors = []
        for i in range(n_masses):
            line, = ax1.plot(times, all_states[i,:], lw=1)
            colors.append(line.get_color())
            ax1_lines.append(line)

        # Plotting the control forces on the masses
        for i in range(n_masses):
            line, = ax2.plot(times, all_actions[i,:], lw=0.7, ls='--', c=colors[i])
            ax2_lines.append(line)

        # Create basis for masses
        for i in range(n_masses):
            m = ax3.add_patch(Rectangle((0,0), mass_w, mass_h, color=colors[i]))
            mass_blocks.append(m)

        # Create basis for springs
        spring_attach_h = 2/3*mass_h
        spring_y_coors = 2*[spring_attach_h]
        spring_y_coors = spring_y_coors + [spring_attach_h + (-1)**i *(spring_h/2) for i in range(2*num_spring_coils)]
        spring_y_coors = spring_y_coors + 2*[spring_attach_h]
        for i in range(n_masses):
            spr, = ax3.plot(np.arange(4+2*num_spring_coils), spring_y_coors, c='k', lw=0.7)
            springs.append(spr)

        # Create basis for dampers
        damper_attach_h = 1/3*mass_h
        damper_y_coors = 2*[damper_attach_h]
        damper_y_coors = damper_y_coors + 3*[damper_attach_h+damper_h/2]
        damper_y_coors = damper_y_coors + 3*[damper_attach_h-damper_h/2]
        damper_y_coors = damper_y_coors + [damper_attach_h + damper_h/2]
        damper_y_coors = damper_y_coors + 2*[damper_attach_h]
        for i in range(n_masses):
            dmpr, = ax3.plot(np.arange(11), damper_y_coors, c='k', lw=0.7)
            dampers.append(dmpr)

        # Create basis for control forces
        for i in range(n_masses):
            ctrl_rod, = ax3.plot([0,0], [mass_h/2, 1.3*mass_h], lw=3, c='k')
            control_rods.append(ctrl_rod)

        # Create basis for control force vector
        largest_action = np.max(all_actions)
        if largest_action > 1e-10:
            for i in range(n_masses):
                ctrl_vect, = ax3.plot([0,0], [1.2*mass_h, 1.2*mass_h], lw=2, c=colors[i])
                control_forces.append(ctrl_vect)

        # Show progress line on x-position plot
        progress_line_ax1, = ax1.plot([0,0], [0,0], lw='0.8', ls='--', c='k')
        progress_line_ax2, = ax2.plot([0,0], [0,0], lw='0.8', ls='--', c='k')
        progress_lines = [progress_line_ax1, progress_line_ax2]


        # Define what to animate
        def animate_MSD(ani_ind):

            # Find mass positions at current time step
            mass_positions_t = all_states[:n_masses, ani_ind]
            mass_positions_t = np.asarray(mass_positions_t).reshape((n_masses,))
            masses_x = np.hstack(([0], mass_positions_t))
            masses_x += np.arange(0, dist_between_m*(n_masses+1), dist_between_m)

            for i in range(n_masses):

                # Animate Position Graph lines
                ax1_lines[i].set_xdata(times[:ani_ind])
                ax1_lines[i].set_ydata(all_states[i,:][:ani_ind])

                # Animate Control Graph lines
                ax2_lines[i].set_xdata(times[:ani_ind])
                ax2_lines[i].set_ydata(all_actions[i,:ani_ind])

                # Animate Mass blocks
                mass_blocks[i].set_x(masses_x[i+1]-mass_w/2)

                # Animate Springs
                if i == 0:
                    m_x_ind = masses_x[i+1] - mass_w/2
                    m_x_ind_1 = 0
                else:
                    m_x_ind = masses_x[i+1] - mass_w/2
                    m_x_ind_1 = masses_x[i] + mass_w/2

                coil_point_dist = (m_x_ind - m_x_ind_1 - 2*spring_attach_rod_l)/(2*num_spring_coils+1)
                x_positions = [m_x_ind_1, m_x_ind_1 + spring_attach_rod_l]
                x_positions = x_positions + [x_positions[1] + n*coil_point_dist for n in range(1, (2*num_spring_coils)+1)]
                x_positions = x_positions + [m_x_ind-spring_attach_rod_l, m_x_ind]

                springs[i].set_xdata([x_positions])


                # Animate Dampers
                if i == 0:
                    m_x_ind = masses_x[i+1] - mass_w/2
                    m_x_ind_1 = 0
                else:
                    m_x_ind = masses_x[i+1] - mass_w/2
                    m_x_ind_1 = masses_x[i] + mass_w/2

                mass_dist = m_x_ind - m_x_ind_1
                mass_dist_fact = mass_dist * ((0.5*damper_w)/dist_between_m)
                bar_1 = [m_x_ind_1, m_x_ind_1+mass_dist/2-damper_w/2]
                cup_up = [bar_1[1], bar_1[1]+damper_w, bar_1[1]]
                cup_down = cup_up[:-1]
                piston = 2*[bar_1[1]+mass_dist_fact]
                bar_2 = [piston[0], m_x_ind]
                x_positions = bar_1 + cup_up + cup_down + piston + bar_2

                dampers[i].set_xdata([x_positions])


                # Animate control rods
                control_rods[i].set_xdata([masses_x[i+1], masses_x[i+1]])


                # Animate control forces
                if largest_action > 1e-10:
                    action = all_actions[i, ani_ind]
                    vector = [masses_x[i+1], masses_x[i+1] + (action/largest_action)*(mass_w)]
                    control_forces[i].set_xdata(vector)


            # Move progress lines
            progress_lines[0].set_xdata([times[ani_ind], times[ani_ind]])
            progress_lines[1].set_xdata([times[ani_ind], times[ani_ind]])

            progress_lines[0].set_ydata(ax1.get_ylim())
            progress_lines[1].set_ydata(ax2.get_ylim())


            return

        if main_title != '':
            fig.suptitle(main_title)

        # Plot 1 (Mass Positions)
        ax1.plot([times[0], times[-1]], [0, 0], lw=1, c='k')
        ax1.set_title('Time vs x-position of masses')
        ax1.set_xlabel('time [s]')
        ax1.set_ylabel('position [m]')
        ax1.grid(color='gray')

        # Plot 2 (Control Inputs)
        ax2.plot([times[0], times[-1]], [0, 0], lw=1, c='k')
        ax2.set_title('Time vs Control inputs forces on masses')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Control force [N]')
        ax2.grid(color='gray')

        # Plot 3 (MSD System visualisation)
        ax3.plot([0, ax3.get_xlim()[1]], [0,0], lw=2, c='k')
        ax3.plot([0, 0], [0, ax3.get_ylim()[1]], lw=4, c='k')
        ax3.set_title('animation of masses')
        ax3.set_xlabel('x-position [m]')
        ax3.get_yaxis().set_visible(False)
        ax3.minorticks_on()
        ax3.grid(b=True, color='#dddddd', which='major')
        ax3.grid(b=True, color='#dddddd', which='minor', linestyle='--', linewidth=0.2)
        ax3.set_axisbelow(True)

        plt.subplots_adjust(left=0.055, bottom=0.06, right=0.95, top=0.92, wspace=0.1, hspace=0.5)

        ani = animation.FuncAnimation(fig, animate_MSD, frames=times.shape[0], interval=1, repeat=False)

        if len(filename)>0:
            dt = np.diff(times[:2])[0]
            fps = int(1 / dt)
            writer = animation.FFMpegWriter(fps=fps, codec=None, bitrate=None, extra_args=None, metadata=None)
            ani.save(filename+'.mp4', writer=writer)

        return ani
