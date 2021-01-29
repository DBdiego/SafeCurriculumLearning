import numpy as np
import os

# Home Made
import Excitement_signals_generator as EXCITE
import Agent_Env_Interaction as AgEnvInt
import Output_Generator as OutGen


class Curriculum(object):
    '''
    Class containing the construction and simulation of a curriculum.
    '''

    def __init__(self, folder_name, num_curricular_steps, transfer_learning_class=None):
        # Directory in which all the results will be saved
        self.folder_name = folder_name

        # Number of curricular steps within the curriculum
        self.num_curr_steps = num_curricular_steps

        # Instance of transfer learning class
        if transfer_learning_class is not None:
            self.transfer_learning = transfer_learning_class

        # List that will contain all the test-setups of the curricular steps
        self.curricular_steps_setup = []

        # List that will contain all the configurations of the curricular steps
        self.curricular_steps_configs = []

        # List that will contain all the simulated curricular steps
        self.curricular_steps_results = []

        # Attribute containing the simulation that is currently run
        self.current_step = None

        # Boolean denoting the successful convergence of all steps within the curriculum
        self.converged = False

        # list saving booleans denoting the successful convergence to a stable policy
        self.step_convergence = []

        return

    def construct(self, curricular_step_folders, configuration_class, system_class, agent_class, environment_class,
                  model_class, reference_class, excitation_class, measurement_noise_class=None,
                  safety_filter_class=None, system_id_class=None, supervisor_class=None):
        '''
        Builds a setup for each curricular step.

        !! All classes that are provided as input have to be uninitiated !!

        :param curricular_step_folders:
        :param configuration_class:
        :param system_class:
        :param agent_class:
        :param environment_class:
        :param model_class:
        :param reference_class:
        :param excitation_class:
        :param measurement_noise_class:
        :param safety_filter_class:
        :param system_id_class:
        :param supervisor_class:
        :return: None
        '''

        print('  ***************************')
        print('  * Constructing Curriculum *')
        print('  ***************************')

        for curr_step_ind in range(self.num_curr_steps):
            print('\t Step {}/{}'.format(curr_step_ind+1, self.num_curr_steps))

            # Get configuration
            sim_config = configuration_class(curr_step_ind)
            print('\t\tLearning Method: {}'.format(sim_config.agent_config.method))

            # Run configuration checks
            self.run_configuration_checks(sim_config)

            # Get curricular folder directory
            curricular_step_folder = curricular_step_folders[curr_step_ind]

            # Generate time steps
            times = np.arange(0, sim_config.duration, sim_config.dt)

            ## Create all required instances of the curricular step
            # --> Create Reference Signal
            reference_signal = reference_class(sim_config.reference_signal_config)

            # --> Excitation signal
            excitation_signal = excitation_class(sim_config.excitation_signal_config)

            # --> Measurement noise
            if measurement_noise_class is not None:
                measurement_noise = measurement_noise_class(sim_config.measurement_noise_config)
            else:
                measurement_noise = None

            # --> Create supervisor_controller
            if supervisor_class is not None:
                supervisor = supervisor_class(sim_config.supervisor_config)
                supervisor_action_mask = supervisor.action_mask
            else:
                supervisor = None
                supervisor_action_mask = None

            # --> Create System
            system = system_class(sim_config.system_config)

            # --> Create Environment
            env = environment_class(system.state_space, reference_signal.F, supervisor_action_mask, sim_config.dt,
                                    sim_config.env_config, reference_signal.reference_map)
            interaction_dimensions = [env.n, env.m, env.p]

            # --> Create agent
            agent = agent_class(interaction_dimensions, supervisor_action_mask, sim_config.agent_config)

            # --> Creating Bounding Model instance
            model = model_class(*interaction_dimensions)

            # --> Creating the safety filter instance
            if safety_filter_class is not None and sim_config.safety_filter_config.activated:
                safety_filter = safety_filter_class(interaction_dimensions,
                                                    sim_config.safety_filter_config,
                                                    agent, curricular_step_folder)
            else:
                safety_filter = None

            # --> Create the system identifier
            if system_id_class is not None:
                system_identifier = system_id_class(interaction_dimensions, sim_config.system_id_config)
            else:
                system_identifier = None

            ## Set up the curricular step agent-environment interaction
            # --> Create instance
            curricular_step = AgEnvInt.AgentEnvironmentInteraction()

            # --> Give the curricular step a name
            curricular_step.name = 'step{}'.format(curr_step_ind+1)

            # --> Give the index of the curricular step
            curricular_step.ind = curr_step_ind

            # --> Assign directory for output results
            curricular_step.folder_name = curricular_step_folder

            # --> Initial conditions
            curricular_step.get_initial_conditions = system.get_initial_conditions

            # --> Reference Signal
            curricular_step.reference_signal = reference_signal

            # --> Excitation Signal
            curricular_step.excitation_signal = excitation_signal

            # --> Measurement Noise
            curricular_step.measurement_noise = measurement_noise

            # --> Define time steps
            curricular_step.times = times

            # --> Assign system instance
            curricular_step.system = system

            # --> Assign agent instance
            curricular_step.agent = agent

            # --> Assign Supervisor controller instance
            curricular_step.supervisor = supervisor

            # --> Assign environment instance
            curricular_step.env = env

            # --> Assign safety filter instance
            curricular_step.safety_filter = safety_filter

            # --> Assign bounding model instance
            curricular_step.model = model

            # --> Assign system identification instance
            curricular_step.system_identifier = system_identifier

            # --> Bool denoting whether to train the agent
            curricular_step.train_agent = sim_config.train_agent

            # Append curricular step instances along with their respective simulation configurations
            self.curricular_steps_setup.append(curricular_step)
            self.curricular_steps_configs.append(sim_config)

        return

    def run_simulation(self):
        '''
        Simulates all curricular steps sequentially.

        :return: None
        '''

        print('  ************************')
        print('  * Executing Curriculum *')
        print('  ************************')

        # Simulate agent in environment
        for i, curricular_step in enumerate(self.curricular_steps_setup):
            print('\t Step {}/{}'.format(i+1, len(self.curricular_steps_setup)))

            self.current_step = curricular_step

            print(self.current_step.agent.num_samples)
            print(self.current_step.system_identifier.num_samples)

            # Get simulation configuration
            sim_config = self.curricular_steps_configs[i]

            if i > 0 and self.transfer_learning is not None:
                # Initialise transfer learning between step i-1 and step i
                transfer_learning = self.transfer_learning(sim_config.transfer_learning_config)

                # Define source agent
                source_agent = self.curricular_steps_results[i-1].agent
                source_supervisor = self.curricular_steps_results[i-1].supervisor

                # Transfer Knowledge to current agent
                self.current_step.agent = transfer_learning.transfer_knowledge(source_agent,
                                                                               self.current_step.agent,
                                                                               source_supervisor)

            # Run the curricular step
            self.current_step.run_interaction()

            # Generate output files
            print('\t\tGenerate output files in: {}'.format(curricular_step.folder_name))
            self.current_step.generate_output(sim_config)
            print()

            # Saving simulated curricular step
            self.curricular_steps_results.append(self.current_step)

            # Rename curricular step folder, if step converged to a stable policy
            if self.current_step.stable_policy_found:
                self.rename_coverged_sim_folder(self.current_step.folder_name)

            # Save convergence boolean, for final assessment of the curriculum as a whole
            self.step_convergence.append(self.current_step.stable_policy_found)

        # Save curriculum stats
        stats_directory = self.folder_name + '/Curriculum_statistics'
        OutGen.save_statistics(np.sum(self.step_convergence), len(self.step_convergence), filename=stats_directory)

        if np.all(self.step_convergence):
            # Switch convergence boolean to true
            self.converged = True

            # Rename folder of curriculum to show "Converged"
            self.rename_coverged_sim_folder(self.folder_name)

        return

    def rename_coverged_sim_folder(self, folder_name):
        '''
        Renames the converged folder by adding " - Converged" at the end of the folder name
        :param folder_name:
        :return:
        '''

        new_folder_name = folder_name + ' - Converged'
        os.rename(folder_name, new_folder_name)

        return

    def run_configuration_checks(self, configuration):
        '''
        Checks if the configuration parameters have the correct sizes, values etc. before running the simulation.
        This provides a more accurate feedback for the user and promotes debugging efficiency and user-friendly usage
        of the algorithm.

        Raises ValueErrors when a configuration parameter does not comform to the requirements

        :param configuration: Instance of the configuration class
        :return: None
        '''

        num_tracked_states = configuration.reference_signal_config.reference_map.shape[0]

        # Check size of reference map and C matrix
        if configuration.system_config.C.shape[0] != num_tracked_states:
            raise ValueError("Configuration: C matrix and Reference map don't have same number of rows")


        # Safety filter checks
        if configuration.safety_filter_config.activated:
            safety_filter_config = configuration.safety_filter_config

            if safety_filter_config.RSS.shape[0] != len(safety_filter_config.RSS_to_ref_mapping.keys()):
                raise ValueError('Configuration: Mismatch between size of RSS and RSS_to_ref_mapping attributes.')

        return











