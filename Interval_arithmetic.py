import numpy as np
import itertools

def construct_uncertainty_matrix(matrix, uncertainty, uncertainty_type='absolute'):
    '''
    Consturcts a column interval matrix of uncertainties of its respective
    elements. Basically replacing its elements by their uncertainty interval.
    As a result, the number of columns in the matrix are dubbled.

    :param matrix: matrix with of which the parametric uncertainty is known.
    :param uncertainty: matrix of same size as "matrix" containing all parametric uncertainties of the matrix elements.
    :param uncertainty_type: default to "relative" but can also be "absolute". It defines the type of uncertainty
                             given in uncertainty.
    :returns uncertain_matrix: the column-interval matrix
    '''

    # Compute absolute uncertainty (absolute value is taken when element is negative)
    if uncertainty_type == 'absolute':
        absolute_uncertainty = uncertainty
    elif uncertainty_type == 'relative':
        absolute_uncertainty = np.abs(np.multiply(matrix, uncertainty))
    else:
        raise ValueError('Input uncertainty_type not valid. Should be "relative" or "absolute".')

    # Find lower and upper bounds of matrix element intervals
    lower_bounds = matrix - absolute_uncertainty
    upper_bounds = matrix + absolute_uncertainty

    # Creating matrix containing all the intervals (twice the amount of columns of initial matrix)
    uncertain_matrix = np.zeros((matrix.shape[0], int(matrix.shape[1]*2)))

    # Creating mapping masks for the lower and upper bounds of the intervals in the uncertain matrix
    even_column_mask = np.zeros(uncertain_matrix.shape).astype('bool')
    even_column_mask[:,::2] = np.ones(matrix.shape).astype('bool')
    odd_column_mask = np.invert(even_column_mask)

    # Map lower and upper interval bounds to matrix
    uncertain_matrix[even_column_mask] = np.asarray(lower_bounds).reshape((lower_bounds.size,))
    uncertain_matrix[odd_column_mask] = np.asarray(upper_bounds).reshape((upper_bounds.size,))

    return uncertain_matrix

def interval_vectmul(vector1, vector2):
    '''
    This function takes two interval vectors as input. An example of such format
    is the row interval vector given below:

        vec_1 = [[v_{1,a}, v_{1,b}], [v_{2,a}, v_{2,b}]]

    Essentially an interval vector has twice the amount of columns it would have normally. The
    even indices in the vector will obtain lower interval bounds, and the uneven indices will
    obtain the upper bounds of the vector.

    :param vector1: first vector (requires an even size)
    :param vector2: second vector (requires an even size)
    :return resulting_interval: interval representing the new bounds of the vector multiplication
    '''

    # Create empty resulting interval
    resulting_interval = np.zeros((1, 2))

    # Insuring the the array format of variables (Eliminates the 1-D array problems)
    vector1 = np.asmatrix(vector1)
    vector2 = np.asmatrix(vector2)

    # Change both of the vectors into column interval vectors
    vector1 = shape_to_column_interval_vector(vector1)
    vector2 = shape_to_column_interval_vector(vector2)

    # Verify compliance of vector shapes
    if vector1.shape[0] != vector2.shape[0]:
        raise ValueError('Vector sizes do not match: vector1: {}x{}; vector2: {}x{}'.format(*vector1.shape,
                                                                                            *vector2.shape))

    for row_index in range(vector1.shape[0]):
        # Collecting the intervals to multiply
        vec1_interval = list(np.array(vector1[row_index, :]))
        vec2_interval = list(np.array(vector2[row_index, :]))

        # Compute the product of all the permutations
        all_products = [x[0] * x[1] for x in itertools.product(vec1_interval, vec2_interval, repeat=1)]

        # Find min and max for defining the new interval
        new_interval = np.array([np.min(all_products), np.max(all_products)])

        # adding newly found interval to the final result
        resulting_interval = resulting_interval + new_interval

    return resulting_interval

def interval_matmul(matrix1, matrix2):
    '''
    Multiplies two interval matrices. For each matrix, the matrix elements are represented by an
    interval.

    :param matrix1: first interval matrix
    :param matrix2: second interval matrix
    :return result_matrix: resulting interval matrix
    '''

    # Converting both inputs to np.matrix() format
    matrix1 = np.asmatrix(matrix1)
    matrix2 = np.asmatrix(matrix2)

    # Dimension checks
    if int(matrix1.shape[1]/2) != matrix2.shape[0]:
        raise ValueError('Matrix sizes do not match: matrix1 -> {}x{}; matrix2 -> {}x{}'.format(*matrix1.shape,
                                                                                                *matrix2.shape))

    # Create emtpy array to be filled with new intervals
    result_matrix = np.empty((matrix1.shape[0], matrix2.shape[1]))
    result_matrix = np.asmatrix(result_matrix)

    # Go over all rows of matrix 1
    for row_mat1_index in range(matrix1.shape[0]):

        # Select a row from interval matrix 1
        matrix1_row = matrix1[row_mat1_index, :]

        # Go over all columns of matrix 2
        for col_mat2_index in range(0, matrix2.shape[1], 2):
            # Select a row from interval matrix B
            matrix2_col = matrix2[:, col_mat2_index:col_mat2_index + 2]

            # Calculate new element interval in result_matrix
            new_interval = interval_vectmul(matrix1_row, matrix2_col)

            # Completing result matrix
            result_matrix[row_mat1_index, col_mat2_index:col_mat2_index + 2] = new_interval

    return result_matrix

def shape_to_column_interval_vector(vector):
    '''
    Shapes a vector of even size (bigger than zero) into a column interval matrix.

    :param vector: input vector to be shaped
    :return new_vector: reshaped vector (if dimensions and size of input vector allow it)
    '''

    # Check if vector already in column interval format
    if vector.shape[1] == 2:
        new_vector = np.copy(vector)

    elif np.size(vector) % 2 == 0:
        new_vector = vector.reshape((int(max(vector.shape)/2), 2))

    else:
        raise ValueError('Vector has inappropriate shape for conversion to column interval vector.')

    new_vector = np.asarray(new_vector)

    return new_vector
