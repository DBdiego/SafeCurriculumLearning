import numpy as np
import time

#Home made
import Excitement_signals_generator as EXCITE
import Supervisor_Controllers as Supervisors
import Reference_signal_generator as REF
import System_identification as SystemID
import Mass_Spring_Damper_System as MSD
import Measurement_Noise as MEAS_noise
import Curriculum_Learning as CURR
import Output_Generator as OutGen
import Transfer_Learning as TL
import Environment as ENV
import Quadrotor as Quad
import Configurations
import Agents
import Safety
import Models


# Set numpy print options to longer floats and larger print lines
np.set_printoptions(precision=5, linewidth=300)
np.random.seed(9)


# Define simulation and curricular outline
num_curricula = 1
num_curricular_steps = 3


# Boolean stating the addition of noise or not
impose_flat_learning = True
add_measurement_noise = False
add_safety_filter = False

# Selection of the system
system = 'MSD'  # also 'Quadrotor'
# system = 'Quadrotor'

configuration_class = Configurations.select_configuration_class(system,
                                                                impose_flat_learning,
                                                                add_measurement_noise,
                                                                add_safety_filter)

# Chose simulation configuration based on system and noise choice
if system =='MSD':
    # Define system class
    system_class = MSD.N_MSD

    # Set supervisor class to none, since no supervisor is used in MSD simulations
    supervisor_class = None

    # Set transfer learning class to the one with MSD semantic mapping
    transfer_learning_class = TL.TransferLearningMSD

    # Making sure only one curricular step is used for flat learning
    if impose_flat_learning:
        num_curricular_steps = 1

elif system == 'Quadrotor':
    # Define system
    system_class = Quad.QuadrotorSystem

    # Define supervisor
    supervisor_class = Supervisors.PController

    # Define transfer learning class
    transfer_learning_class = TL.TransferLearningQuadrotor

    # Making sure only one curricular step is used for flat learning
    if impose_flat_learning:
        num_curricular_steps = 1

else:
    raise ValueError('No configuration implemented for the desired system')

# Chose select measurement noise class
if add_measurement_noise:
    measurement_noise_class = MEAS_noise.MeasurementNoise
else:
    measurement_noise_class = None

# Assign safety filter class
if add_safety_filter:
    safety_filter_class = Safety.SHERPA
else:
    safety_filter_class = None


print('Simulation started')
simulation_start = time.time()

# Create output f older tree
curr_folders, curr_step_folders = OutGen.create_output_folder_tree(num_curricula, num_curricular_steps, system)

# Run simulation
succesful_count = 0
for curriculum_ind in range(num_curricula):
    print('Curriculum {:2d}/{}\n'.format(curriculum_ind+1, num_curricula))

    # Create curriculum instance
    curriculum = CURR.Curriculum(curr_folders[curriculum_ind],
                                 num_curricular_steps,
                                 transfer_learning_class=transfer_learning_class)

    # Construct Curriculum
    curriculum.construct(curr_step_folders[curriculum_ind],
                         configuration_class,
                         system_class,
                         Agents.Agent,
                         ENV.Environment,
                         Models.SS_model,
                         REF.ReferenceSignal,
                         EXCITE.ExcitationSignal,
                         measurement_noise_class=measurement_noise_class,
                         safety_filter_class=safety_filter_class,
                         system_id_class=SystemID.SystemIdentificationSS,
                         supervisor_class=supervisor_class)

    # Execute the curriculum
    curriculum.run_simulation()

    # Count successful curricula
    succesful_count += curriculum.converged * 1
    print('')

    step = curriculum.current_step

# Saving stats to a txt file
stats_directory = '/'.join(curr_folders[0].split('/')[:4] + ['Simulation_statistics'])
OutGen.save_statistics(succesful_count, num_curricula, filename=stats_directory)

# Printing end statement
simulation_time = round(time.time()-simulation_start, 2)
print('\nSimulation: DONE (' + str(simulation_time) + 's)\n')

# step = curriculum.current_step
# curriculum.current_step.generate_output(curriculum.curricular_steps_configs[0])


# step = curriculum.current_step
# actual = step.safety_filter.actual_state_evolution
# a = np.zeros((12,0))
# for i in actual:
#     a = np.hstack((a, np.asarray(i)))
#
# projected = step.safety_filter.projected_state_evolution
# p = np.zeros((12,0))
# for i in projected:
#     p = np.hstack((p, np.asarray(i)))




