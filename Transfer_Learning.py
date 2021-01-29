import numpy as np
import numpy.linalg
import scipy
import scipy.linalg


class TransferLearningMSD(object):
    '''

    '''

    def __init__(self, configuration):
        self.curr_step_ind = configuration.curr_step_ind
        self.use_supervisor_policy_as_basis = configuration.use_supervisor_policy_as_basis

        return

    def transfer_knowledge(self, previous_agent, new_agent, previous_supervisor):
        '''

        :param SS_dimensions: The state space dimensions
        :param H_source: The source kernel matrix which is to be mapped
        :return: H_target, the mapped kernel matrix
        '''

        # Map kernel matrix
        new_agent.H_j = self.semantic_mapping(previous_agent.H_j, new_agent.n, new_agent.m, new_agent.p)

        # Update agent's initial policy parameters based on the kernel mapping
        new_agent.policy_parameters = new_agent.improve_policy()

        # Save mapped H as new initial value in the kernel storage attribute H_storage
        H_vector = new_agent.H_to_H_1D()
        new_agent.H_storage = np.copy(H_vector)

        return new_agent

    def semantic_mapping(self, H_source, n, m, p):
        '''

        :param H_source: The source kernel matrix from which values are mapped
        :param n: Dimension of the A matrix in the new curricular step (i.e. n from the new agent)
        :param m: Dimension of the B matrix in the new curricular step (i.e. m from the new agent)
        :param p: Dimension of the F matrix in the new curricular step (i.e. p from the new agent)
        :return: H_target, the mapped kernel matrix
        '''

        # Creating the basis for H_target
        H_target = np.identity(n + m + p)

        # Computing mapping characteristics
        new_num_masses = int(n/2)
        num_added_masses = int((H_target.shape[0] - H_source.shape[0]) / 3)
        old_num_masses = new_num_masses - num_added_masses

        # Define mapping mask
        mapping_mask = self.semantic_mapping_mask(p, old_num_masses, num_added_masses)

        # Assign values to H_new according to the mapping mask
        H_target[mapping_mask] = np.asarray(H_source).reshape((H_source.shape[0]**2,))

        return H_target

    def semantic_mapping_mask(self, p, old_m, added_m):
        '''
        Semantic mapping of the kernel matrix. The elements in the kernel matrix related to the dynamics
        of masses that have remained from previous curricular stepped are mapped onto the new kernel matrix.
        Other values will be zero, except for diagonal elements which are 1.

        :param p: dimension of reference signal
        :param old_m:
        :param added_m:
        :return:
        '''

        # Creating mapping mask
        init_mask_shape = (3*(old_m)+p, 3*(old_m)+p)
        semantic_mask = np.ones(init_mask_shape).astype('bool')

        # Define row and column indices where false are inserted
        ind_set_1 = np.arange(added_m)
        ind_set_2 = 1 * old_m + added_m + np.arange(added_m)
        ind_set_3 = 2 * old_m + 2 * added_m + p + np.arange(added_m)
        indices = np.hstack((ind_set_1, ind_set_2, ind_set_3))

        # Insert False valued rows and columns at given indices
        for ind in indices:
            semantic_mask = np.insert(semantic_mask, ind, False, axis=1)
            semantic_mask = np.insert(semantic_mask, ind, False, axis=0)

        return semantic_mask



class TransferLearningQuadrotor(object):
    '''
    Class containing all the required logic and mapping strategies for a linear quadrotor model
    '''

    def __init__(self, configuration):
        self.curr_step_ind = configuration.curr_step_ind
        self.use_supervisor_policy_as_basis = configuration.use_supervisor_policy_as_basis
        self.xi = configuration.xi

        return

    def transfer_knowledge(self, previous_agent, new_agent, previous_supervisor):
        '''
        Transfers the kernel matrix from one curricular step to another based on the changing state and action
        space representation of the agent

        :param previous_agent:
        :param new_agent:
        :return:
        '''

        if self.curr_step_ind in [0, 1, 2, 3]:
            # Map kernel matrix
            new_agent.H_j = self.semmantic_input_mapping(previous_agent.H_j,
                                                         previous_agent.action_mask,
                                                         new_agent.action_mask,
                                                         previous_agent.n,
                                                         previous_agent.p,
                                                         new_agent.n,
                                                         new_agent.p)

        elif self.curr_step_ind == 4:
            new_agent.H_j = self.force_torque_to_RPM_mapping(previous_agent.H_j)


        elif self.curr_step_ind in [5, 6]:
            new_agent.H_j = self.semantic_mapping_xy_dims(previous_agent.H_j, self.curr_step_ind)

        else:
            raise ValueError('No mapping strategy provided between curricular steps {} and {}'.format(self.curr_step_ind,
                                                                                                      self.curr_step_ind+1))

        print(new_agent.H_j)
        print()

        # Update agent's initial policy parameters based on the kernel mapping
        new_agent.policy_parameters = new_agent.improve_policy()

        # Use supervisor policy as basis for new actions available to the agent
        if previous_supervisor is not None and self.use_supervisor_policy_as_basis:
            # Find the row indices of the supervisor gain to use as basis
            sup_K_inds, added_action_inds = self.agent_policy_basis_from_supervisor(previous_agent.action_mask,
                                                                                    new_agent.action_mask)

            # Find the row indices of the agent policy gain onto which the supervisor policy is to be mapped
            new_policy_param_rows = np.sum(np.asarray(new_agent.policy_parameters), axis=1) == 0

            # Map supervisor policy to agent's policy
            new_agent.policy_parameters[new_policy_param_rows,:9] = previous_supervisor.K[sup_K_inds, :9]

        print(new_agent.policy_parameters)

        # Save mapped H as new initial value in the kernel storage attribute H_storage
        H_vector = new_agent.H_to_H_1D()
        new_agent.H_storage = np.copy(H_vector)

        return new_agent

    def semmantic_input_mapping(self, H_source, input_mask_1, input_mask_2, prev_n, prev_p, new_n, new_p):
        '''

        :param H:
        :param input_mask_1:
        :param input_mask_2:
        :param n:
        :param p:
        :return:
        '''

        if input_mask_1.shape != input_mask_2.shape:
            raise ValueError("Input mask 1 and 2 don't have the same dimensions")

        # Retain the non-input related parts of H
        H_new = H_source[:(prev_n+prev_p), :(prev_n+prev_p)]

        # Create intermediate input column mapping array
        H_input_cols_all = np.zeros((prev_n+prev_p, input_mask_1.shape[0]))
        H_input_cols_all[:, input_mask_1] = H_source[:(prev_n+prev_p), -np.sum(input_mask_1):]

        # New H columns related to the input
        H_new_input_cols = H_input_cols_all[:, input_mask_2]

        # Remaining H_source values
        H_remaining = H_source[(prev_n+prev_p):, (prev_n+prev_p):]

        input_mask_1_int = np.asmatrix(input_mask_1, dtype='int')
        mask = np.matmul(input_mask_1_int.T, input_mask_1_int).astype('bool')
        mask = mask[input_mask_2,:][:,input_mask_2]

        H_new_remaining = np.identity(int(np.sum(input_mask_2)))
        H_new_remaining[mask] = np.array(H_remaining).reshape(H_new_remaining[mask].shape)

        # New H rows related to the input
        H_new_input_rows = np.transpose(H_new_input_cols)
        H_new_input_rows = np.hstack((H_new_input_rows, H_new_remaining))

        # Complete H_new
        H_new = np.hstack((H_new, H_new_input_cols))
        H_new = np.vstack((H_new, H_new_input_rows))

        # Add empty rows and columns for new reference signal if new one is added
        if (prev_n == new_n) and (prev_p < new_p):
            H_final = np.identity(H_new.shape[0]+(new_p-prev_p))
            final_mask = np.ones(H_final.shape).astype('bool')
            final_mask[prev_n+prev_p:prev_n+new_p,:] = 0
            final_mask[:, prev_n+prev_p:prev_n+new_p] = 0

            H_final[final_mask] = np.array(H_new).reshape((H_new.size,))
        else:
            H_final = np.copy(H_new)

        return H_final

    def force_torque_to_RPM_mapping(self, H_source):

        H_uu_source = np.copy(H_source[-4:,-4:])
        # H_Xu_source = np.copy(H_source[:-4,-4:])
        # H_uX_source = np.copy(H_source[-4:,:-4])

        H_uu_target = np.matmul(self.xi, H_uu_source)
        # H_Xu_target = np.matmul(H_Xu_source, self.xi)
        # H_uX_target = np.matmul(self.xi, H_uX_source)

        H_target = np.copy(H_source)
        H_target[-4:,-4:] = H_uu_target
        # H_target[:-4,-4:] = H_Xu_target
        # H_target[-4:,:-4] = H_uX_target

        return H_target

    def semantic_mapping_xy_dims(self, H_source, curr_step_ind):
        '''

        :param H_source:
        :return:
        '''

        # # Create Basis for H target (identity matrix)
        # H_target = np.identity(H_source.shape[0]+2)
        #
        # # Define indices of rows and columns to be kept at zero
        # state_inds = np.array([-6, -(curr_step_ind+4)])
        #
        # # Create semmantic map
        # semmantic_map = np.ones(H_target.shape).astype('bool')
        # semmantic_map[state_inds, :] = 0
        # semmantic_map[:, state_inds] = 0
        #
        # # Map H_source to H_target
        # H_target[semmantic_map] = np.asarray(H_source).reshape((H_source.shape[0]**2,))

        semmantic_mask = np.ones(H_source.shape).astype('bool')

        semmantic_mask = np.insert(semmantic_mask, -5, False, axis=1)
        semmantic_mask = np.insert(semmantic_mask, -5, False, axis=0)

        H_target = np.identity(H_source.shape[0]+1)

        H_target[semmantic_mask] = np.asarray(H_source).reshape((H_source.shape[0]**2,))

        # H_target[-5:,-2] = 0
        # H_target[-2,-5:] = 0
        # H_target[-2,-2] = 5

        return H_target

    def agent_policy_basis_from_supervisor(self, source_agent_action_mask, target_agent_action_mask):
        '''

        :param source_agent_action_mask:
        :param target_agent_action_mask:
        :return: supervisor_K_row_inds, added_actions_inds
        '''

        # Define the supervisor action mask of the source task
        source_supervisor_action_mask = np.invert(source_agent_action_mask)

        # Derive which action dimensions were added to the agent between curricular steps
        added_actions_inds = source_agent_action_mask ^ target_agent_action_mask

        # Define mapping indices between the action masks and the supervisor gain
        K_all_row_inds = np.ones(source_supervisor_action_mask.shape)*-1
        K_all_row_inds[source_supervisor_action_mask] = np.arange(np.sum(source_supervisor_action_mask))

        # Derive the desired row indices of the supervisor gain to be used as the basis for the agent's policy
        supervisor_K_row_inds = K_all_row_inds[added_actions_inds].astype('int')

        return supervisor_K_row_inds, added_actions_inds
