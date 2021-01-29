import numpy as np

def get_initial_conditions(n_masses):
    k = 200
    c = 2
    all_k = np.ones(n_masses)*k
    all_c = np.ones(n_masses)*c
    if n_masses == 1:
        masses = np.array([0.4])
        x0 = (np.random.rand(n_masses)-0.5)*0.5  # np.array([-1, 0, 0])
        x0_d = (np.random.rand(n_masses)-0.5)*0.5  # np.array([0, 0, 0])
        # x0 = np.array([0.0190115])
        # x0_d = np.array([-0.243905])
        all_k = np.array([4])
        all_c = np.array([3])

    elif n_masses == 2:
        masses = np.array([0.4, 0.8])
        x0 = (np.random.rand(n_masses)-0.5)*0.3  # np.array([-1, 0, 0])
        x0_d = (np.random.rand(n_masses)-0.5)*0.2  # np.array([0, 0, 0])
        # x0 = np.array([-0.1769, -0.0315])
        # x0_d = np.array([-0.1741, -0.1446])
        # x0 = np.array([0.08849641, 0.13033923])
        # x0_d = np.array([-0.10473509,-0.22518014])
        # x0_d = np.array([-0.10473509, -0.022518014])
        all_k = np.array([ 3, 4])
        all_c = np.array([1, 3])

    elif n_masses == 3:
        masses = np.array([0.3, 0.8, 0.4])
        x0 = (np.random.rand(n_masses)-0.5)*0.3  # np.array([-1, 0, 0])
        x0_d = (np.random.rand(n_masses)-0.5)*0.2 # np.array([0, 0, 0])
        all_k = np.array([-1,  3, 4])
        all_c = np.array([ 6, -1, 3])

    elif n_masses == 4:
        masses = np.array([10, 10, 10, 10])
        x0 = np.array([-3, 0, 0, 0])
        x0_d = np.array([0, 0, 0, 0])
        x0_dd = np.array([0, 0, 0, 0])

    elif n_masses == 5:
        masses = np.array([10, 10, 10, 10, 10])
        x0 = np.array([-1, 0, 0, 0, 0])
        x0_d = np.array([0, 0, 0, 0, 0])
        x0_dd = np.array([2, 0, 0, 0, 0])

    elif n_masses == 6:
        masses = np.array([10, 10, 10, 10, 10, 10])
        x0 = np.array([-1, 0, 0, 0, 0, 0])
        x0_d = np.array([0, 0, 0, 0, 0, 0])
        x0_dd = np.array([2, 0, 0, 0, 0, 0])

    elif n_masses == 7:
        masses = np.array([10, 10, 10, 10, 10, 10, 10])
        x0 = np.array([-1, 0, 0, 0, 0, 0, 0])
        x0_d = np.array([0, 0, 0, 0, 0, 0, 0])
        x0_dd = np.array([2, 0, 0, 0, 0, 0, 0])

    elif n_masses == 8:
        masses = np.array([10, 10, 10, 10, 10, 10, 10, 10])
        x0 = np.array([-1, 0, 0, 0, 0, 0, 0, 0])
        x0_d = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        x0_dd = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    else:
        masses = np.ones(n_masses)*10
        x0 = 4 * np.random.random(n_masses) - 2
        x0_d = np.ones(n_masses)*0
        x0_dd = np.ones(n_masses) * 0
        #raise ValueError('No initial conditions have been defined for the requested number of masses')

    state_init = np.hstack((x0, x0_d)).reshape((2*n_masses, 1))
    return masses, state_init, all_k, all_c