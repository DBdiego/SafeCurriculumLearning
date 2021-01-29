import numpy as np
import os
import json
import datetime

def create_output_folder_tree(num_curricula, num_curricular_steps, system_name):
    '''

    :param num_curricula:
    :param num_curricular_steps:
    :return:
    '''

    if not os.path.isdir('./Output'):
        os.mkdir('./Output')

    parent_folder = './Output/' + system_name + '/'
    if not os.path.isdir('./Output/'+ system_name + '/'):
        os.mkdir(parent_folder)

    simulation_folder = parent_folder + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    # Setup output folder
    if os.path.isdir(simulation_folder):
        os.rmdir(simulation_folder)
    os.mkdir(simulation_folder)

    # Creat Curriculum folders
    curriculum_folders = []
    curricular_step_folders = []
    for curriculum_ind in range(num_curricula):
        curriculum_folder_name = simulation_folder + '/Curriculum ' + str(curriculum_ind+1)
        curriculum_folders.append(curriculum_folder_name)
        os.mkdir(curriculum_folder_name)

        step_folder_list = []
        for curricular_step_ind in range(num_curricular_steps):
            step_folder_name = curriculum_folder_name + '/' + 'Step ' + str(curricular_step_ind+1)
            step_folder_list.append(step_folder_name)
            os.mkdir(step_folder_name)
        curricular_step_folders.append(step_folder_list)

    return curriculum_folders, curricular_step_folders


def save_statistics(successes, total, filename=None):
    '''

    :param successes:
    :param total:
    :param filename:
    :return:
    '''

    if filename is None:
        print('Success rate: {:2d}/{:2d} ({:3.2f}%)'.format(successes, total, successes/total*100))
    else:
        string2write = ''
        string2write += 'Success rate: {:2d}/{:2d} ({:3.2f}%)'.format(successes, total, successes/total*100)
        file = open(filename+'.txt', 'w')
        file.write(string2write)
        file.close()

    return

def load_existing_sim_options(filename):
    '''

    :param filename:
    :return:
    '''
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    options_dict = np.load(filename)
    options_dict = options_dict.reshape((1,))
    options_dict = options_dict[0]
    return options_dict

def save_dict(dictionary, filename):
    np.save(filename, dictionary)



