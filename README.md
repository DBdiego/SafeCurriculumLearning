# Safe Curriculum Learning for Systems With Unknown Dynamics
## Thesis D.D.C. De Buysscher
This thesis considers the implementation of safe curriculum learning on systems for which inherent dynamics are unknown. The dimensions of the state and input are known. The goal is to control any system such that it tracks a reference signal, which is defined externally. Examples of applications for such setup are following predefined flight paths for aircraft or following a racing line for cars, etc. The scope is not limited to these examples, though.

The full thesis is outlined in this [document](https://www.overleaf.com/read/ttzvhrdxgynn), a technical report provides a better insight towards the theory supporting the work developed for this solution. It contains a literature study on the 5 main subjects of the thesis:

- Reinforcement Learning (RL)
- Optimal Control
- Curriculum Learning (CurL)
- Safe Learning (SL)
- System Identification

Furthermore, the literature study is complemented with a preliminary analysis on a reinforcement learning environment based on a N-Mass-Spring-Damper (N-MSD) system and a final analysis performed on a linearised quadcopter model. For the safety aspect, the SHERPA algorithm is used.

Should the reader be interested in the specifics, he/she is invited to read this [document](https://www.overleaf.com/read/ttzvhrdxgynn).



## Python Implementation

The python implementation has been designed with modularity in mind. The idea is that the modules can be swapped or removed as easily as possible.  It comes with a slight additional complexity due to the few rules that have to be followed strictly for the simulation to work.

### How to run?

In order to run the code, the ```Main.py``` file has to be run with Python 3.5 (or higher). Furthermore, to ensure an optimal code execution, it is required that the following packages are available to the Python version:

- Numpy

- Scipy

- Matplotlib (with Patches)

To select which system to simulate, the ```Main.py``` file contains a variable called ```system```, which defines whether to simulate the agent in a N-MSD or linearised quadrotor environment. Additionally, depending on the system, a set of configuration files are pre-defined in the reposistory already. Additional Configurations can be added to the ```./Configurations``` module/directory. The configuration to be used is selected based on three booleans:

- ```impose_flat_learning```: When true, the number of curricular steps will be reduced to 1 and a flat learning configuration chosen.
- ```add_measurement_noise```: When true, a configuration with the required hyperparameters for measurement noise will be chosen.
- ```use_safety_filter```: When true, a configuration with the required hyperparameters for the safety filter will be chosen.

In case a combination of these "decision booleans" does not have a corresponding configuration, a ValueError is raised. The error message states the configuration that has been requested.



### Communication Channels between modules

For a modular implementation, the configuration parameters can not be hardcoded in the general framework. Consequently, a system of communication channels has been put in place. All configuration parameters are placed in the configuration file (located in ```./Configurations```). Each module has a specific configuration class that is the grouped in the general simulation configuration. In order to avoid "knowledge leaks", the modules only have access to the configuration hyperparameters in their respective configuration class.

An example of a configuration file is given below, where the simulation configuration assigns itself all the module configurations as attributes. In the example below, only the mendatory attributes of each module configurations are given. However, more attributes can be defined in these classes. They will be availble in the respective modules as ```configuration.$name_of_attribute```.

```Python
class Configuration(object):
  def __init__(curr_step_ind):
    self.train_agent = True
    self.duration = 50
    self.dt = 0.005
    
    # Set up the configuration of the system
    self.system_config = SystemConfig(curr_step_ind)

    # Set up the configuration of the agent
    self.agent_config = AgentConfig(curr_step_ind, self.system_config)

    # Set up the configuration of the environment
    self.env_config = EnvironmentConfig(curr_step_ind)

    # Set up the configuration of the excitation signal
    self.excitation_signal_config = ExcitationSignalConfig(curr_step_ind)

    # Set up the configuration of the reference signal
    self.reference_signal_config = ReferenceSignalConfig(curr_step_ind, self.dt)

    # Set up the configuration of the safety filter
    self.safety_filter_config = SafetyFilterConfig(curr_step_ind, self.dt, self.agent_config.action_bounds)

    # Set up the configuration of the supervisor controller
    self.supervisor_config = SupervisorConfig(curr_step_ind)

    # Set up the configuration of the system
    self.system_config = SystemConfig(curr_step_ind)

    # Set up the configuration of the system identifier
    self.system_id_config = SystemIDConfig(curr_step_ind)

    # Set up the configuration of transfer learning
    self.transfer_learning_config = TransferLearningConfig(curr_step_ind, self.system_config.xi)

# ---> Agent
class AgentConfig(object):
    def __init__(self, curr_step_ind, system_config):
        self.save_H_values = True

        self.learning_rate = 0.8

        self.method = 'VI'  # 'PI' also available
        self.sample_factor = 1
        self.action_bounds = np.array([-20, 20])

        return
    
# ---> Environment
class EnvironmentConfig(object):
    def __init__(self, curr_step_ind):
        self.run_system_checks = True

        self.Q_diag_values = np.array([1e5, 1e2])
        self.R_diag_values = np.array([1e1])
        self.R_diag_value = 1e1

        self.RMSE_limit = 0.5

        return

# ---> Excitation Signal
class ExcitationSignalConfig(object):
    def __init__(self, curr_step_ind):
        self.signal_type = 'frequency_limited' # Also other?

        self.means = np.array([0.0, 0.0, 0.0, 0.0])
        self.amplitudes = np.array([10.5, 13, 13, 21.0])
        self.frequency_ranges = np.array([[50, 100],  # [rad/s]
                                          [50, 100],  # [rad/s]
                                          [50, 100],  # [rad/s]
                                          [50, 100]]) # [rad/s]
        return


class ReferenceSignalConfig(object):
    def __init__(self, curr_step_ind, dt):
        # Define dt
        self.dt = dt
				
        # Define reference signal properties 
        self.signal_type = 'sinusoidal'  # OTHERS?
        self.amplitudes = np.array([np.pi/2])
        self.phase = np.random.random((1,)) * np.pi
        self.frequencies = np.array([3]) * 1/(2*np.pi)

        self.reference_map = np.array([[-1, 0],
                                       [ 0,-1]])

        return

# ---> Safety Filter
class SafetyFilterConfig(object):
    def __init__(self, curr_step_ind, dt, action_bounds):
        self.activated = True

        # Only define more detailed attributes when the filter is active
        if self.activated:
            # Setting dt
            self.dt = dt

            # Setting action bounds (taken from agent configuration)
            self.action_bounds = np.copy(action_bounds)

            # Plot policy simulations
            self.plot_backup_policy_projection = True

            # Define what SHERPA does when no backup policy is found an no safe projection can be found.
            self.last_resort_action = 'pass-through'  # also 'random'

            # Define whether the policy search is processed sequentially or in parallel
            self.policy_search_processing = 'parallel'  # also "sequentlial"

            # Iteration parameters
            self.num_iterations_input = 10  # Number of tries to find a backup policy
            self.num_iterations_backup = 10  # Number of policies to be checked
            self.backup_size = 40  # Number of time steps are followed per policy (when using backup policy)
            self.backup_projection_size = np.array([10, 120])

            # Range in which the parameters of a randomly generated policy (in policy generation)
            self.policy_parameter_range = np.array([-10, 10])
            
            if curr_step_ind == 0:
                self.states_closeness_condition = np.array([0, 3]) 
                self.RSS = np.array([0])
                self.RSS_to_ref_mapping = {0:2}

                self.FSS = np.array([[-0.6, 0.6],
                                     [ 0.0, 0.0],
                                     [ 0.0, 0.0],
                                     [ 0.0, 0.0]])

                self.sensor_reach = np.array([[-0.2, 0.2],
                                              [ 0.0, 0.0],
                                              [ 0.0, 0.0],
                                              [ 0.0, 0.0]])

                self.v_epsilon = np.array([[-0.06, 0.06],
                                           [ 0.00, 0.00],
                                           [ 0.00, 0.00],
                                           [-0.04, 0.04]])
               
            elif curr_step_ind == 1:
                self.states_closeness_condition = None
                self.RSS = None
                self.FSS = None
                self.sensor_reach = None
                self.v_epsilon = None

            else:
                raise ValueError('Curricular step {} not implemented'.format(curr_step_ind+1))

        return

# ---> Supervisor
class SupervisorConfig(object):
    def __init__(self, curr_step_ind):
      # Define mask for which actions this controller has a say on
      self.action_mask = np.array([True, True, True, False])

      # # Define Gain matrix
      K = np.zeros((3,10))
      K[0,6] = -1.1784e+00  # Cancel vertical rate (w)
      K[0,7] = -3.1538e-01  # Cancel vertical position (z)
      K[1,0] = 1.4996e+00  # Cancel roll (phi)
      K[1,3] = 4.3504e-01  # Cancel roll rate (p)
      K[2,1] = 1.5192e+00  # Cancel pitch (theta)
      K[2,4] = 4.4361e-01  # Cancel pitch rate (q)

      if np.sum(self.action_mask) == K.shape[0]:
        self.K = np.asmatrix(K)
      else:
        raise ValueError('''Supervisor Gain matrix does not have the correct size. According to the action 
        mask, it controls {} actions, however, the gain matrix has {} rows.'''.format(np.sum(self.action_mask), 				K.shape[0]))

        return

# ---> System
class SystemConfig(object):
    def __init__(self, curr_step_ind):
        # Plotting Booleans
        self.plot_static = True
        self.plot_animation = False

        # Define initial conditions
        self.init_bounds = np.array([[-np.pi/36, np.pi/36],
                                     [-np.pi/36, np.pi/36],
                                     [-np.pi/2 , np.pi/2 ],
                                     [-np.pi/24, np.pi/24],
                                     [-np.pi/24, np.pi/24],
                                     [-np.pi/4 , np.pi/4 ],
                                     [-0.05, 0.05],
                                     [-0.02, 0.02]])

        # Define C matrix
        self.C = np.zeros((2, 8))
        self.C[0, 2] = 1
        self.C[1, 5] = 1

        return

# --> System Identifier
class SystemIDConfig(object):
    def __init__(self, curr_step_ind):
        # Boolean whether to use sliding window principle or not
        self.use_sliding_window = True

        # Define sample factor for each curricular step
        self.sample_factor = 2

        return

# --> Transfer Learning
class TransferLearningConfig(object):
    def __init__(self, curr_step_ind, xi):
        self.curr_step_ind = curr_step_ind

        # Boolean denoting the use of the supervisor policy as initial value for agent policy
        self.use_supervisor_policy_as_basis = True

        return

```



Besides the configuration, the output generation has also been designed with modularity in mind. Each module is equipped with its own ```generate_output``` function. Taking as input, the module's configuration class (for communication channel logic explained above), as well as the curricular step simulated instance. Each module has access to all the results of the simulated agent-environment interaction. It is understood that this decision allows more condensed plots to show results that might relate to multiple modules.



### Modules

In this section a brief explanation on the multiple modules is given.

#### Agent

The agent module is what defines the agent's role in Reinforcement learning interaction. In this implementation, the agent module is expected to learn a control policy using state samples. The mendatory functions that have to be included in the agent's class/module along with their respective in/outputs are given in the table below:

Mendatory functions for the ***Agent*** module

| Function            | Input(s)                                     | Output(s)        |
| ------------------- | -------------------------------------------- | ---------------- |
| action_saturation() | action                                       | action           |
| collect_samples()   | state, action                                | None             |
| evaluate_policy()   | None                                         | None             |
| improve_policy()    | None                                         | None             |
| policy()            | state, policy_parameters                     | action           |
| uncertain_policy()* | uncertain_state, uncertain_policy_parameters | uncertain_action |

​	*only necessary when a safety filter is present that requires this functionality (such as SHERPA)



Additionally, the some of the Agent's class attributes are used in other modules such as the safety filter. Therefore, some class attributes are mandatory for the simulation to function correctly. These attribytes, are given in the table below, along with their meaning and in which module they're used.

| Attribute         | Meaning/role                                                 |
| ----------------- | ------------------------------------------------------------ |
| policy_parameters | contains all the parameters required to define its policy    |
| num_samples       | Defines after how many time steps the VI or PI is performed. (i.e. number of samples collected) |
| n                 | state vector dimensions                                      |
| m                 | input vector dimension                                       |
| p                 | reference vector dimension                                   |



#### Environment

In essence, the environment is responsible for state propagation along with a measure of how favorable this transition is. The latter is contained in some form of reward or cost that is computed within the environment. Additionally, the environment can serve as a tool to analyse the model provided by the operator. Tests such as open and closed loop stability, observability, controllability etc, can all be performed in the environment. 

For the code to function correctly, the environment module/class requires certain functions that are to be included in the class instance. These are given in the table below:

Mendatory functions for the ***Environment*** module

| Function          | Input(s)      | Output(s) |
| ----------------- | ------------- | --------- |
| calculate_cost()  | state, action | cost      |
| propagate_state() | state, action | new_state |
|                   |               |           |
|                   |               |           |
|                   |               |           |
|                   |               |           |



Mendatory attributes for the ***Environment*** module/class

| Attribute         | Meaning/role                                                 |
| ----------------- | ------------------------------------------------------------ |
| policy_parameters | contains all the parameters required to define its policy    |
| num_samples       | Defines after how many time steps the VI or PI is performed. (i.e. number of samples collected) |
| n                 | state vector dimensions                                      |
| m                 | input vector dimension                                       |
| p                 | reference vector dimension                                   |



### Safety filter

some explanation about the safety filter



### Agent-Environment Interaction

some explanation about the RL interaction



### Model

Model reperesentation module. This module is used to transfer the model estimate information from the system identification module to the safety filter module.



Mendatory functions for the ***Model*** module

| Function          | Input(s)      | Output(s) |
| ----------------- | ------------- | --------- |
| ????              | state, action | cost      |
| propagate_state() | state, action | new_state |
|                   |               |           |
|                   |               |           |
|                   |               |           |
|                   |               |           |



Mendatory attributes for the ***Model*** module/class

| Attribute   | Meaning/role                                                 |
| ----------- | ------------------------------------------------------------ |
| ????        | contains all the parameters required to define its policy    |
| num_samples | Defines after how many time steps the VI or PI is performed. (i.e. number of samples collected) |
| n           | state vector dimensions                                      |
| m           | input vector dimension                                       |
| p           | reference vector dimension                                   |



### System

some explanation about the system (i.e. class defining the MSD system etc)

Requied functions:



### System Identifier

some explanation about the system id class


```

```