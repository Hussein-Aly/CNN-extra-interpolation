import numpy as np


def data_reader(image_array: np.ndarray, offset: tuple, spacing: tuple):
    """

    :param image_array: A numpy array of shape (M, N, 3) and numeric datatype, which contains the RGB image data
    :param offset: A tuple containing 2 int values. These two values specify the offset of the first grid point in x and
    y direction from the top left side
    :param spacing: A tuple containing 2 int values. These two values specify the spacing between two successive
    grid points in x and y direction
    :return:
    tuple -> (input array, known array , target array)

    input: a 3D numpy array of shape(3, M, N)and the same datatype as image_array. Note that the channel dimension moved
    to the front. input_array should have the same pixel values as image array, with the exception that the
    to-be-removed pixel values off the specified grid are set to 0

    known array: a 3D numpy array of same shape and datatype as input_array, where pixels on the specified grid should
    have value 1 and all other unknown pixels have value 0.

    target array: a 1D numpy array of the same datatype as image_array. It should hold the R-, G-, and B-pixel values at
     the off-grid locations (the pixels that were set to 0 in input_array).
     The order of the R-, G-, and B-pixels in target_array should be the same as when using the inverted known_array
     as boolean mask on image_array (like image[known_array < 1]), which yields a flattened array
     of all matching R-pixels, followed by all matching G-pixels and all matching B-pixels
     (the length of the 1D numpy array target_array is thus the number of removed pixels times 3)

    """
    # Check for TypeError in image_array if not numpy array
    if not isinstance(image_array, np.ndarray):
        raise TypeError("Image is not a numpy array!")

    # Check if image_array is 3D or not
    if len(np.shape(image_array)) != 3 or np.shape(image_array)[2] != 3:  # Shape -> (M, N, 3) == (H, W, 3)
        raise NotImplementedError("numpy image_array is not in a 3D form and/or 3rd dimension's size is not equal to 3")

    # Check for offset and spacing eligibility
    if not str(offset[0]).isdigit() or offset[0] < 0 or offset[0] > 32 or not str(offset[1]).isdigit() or \
            offset[1] < 0 or offset[1] > 32 or not str(spacing[0]).isdigit() or spacing[0] < 2 or spacing[0] > 8 or \
            not str(spacing[1]).isdigit() or spacing[1] < 2 or spacing[1] > 8:
        raise ValueError("The values of offset and spacing are not convertible to int or out of range!")

    # Set input array to image array with the same values and transpose to -> (3, M, N)
    input_array = np.full_like(image_array, image_array)
    input_array = np.transpose(input_array, (2, 0, 1))

    # Initializing target array as input array and known array
    target_array = np.full_like(input_array, input_array)
    known_array = np.full_like(input_array, 1)

    # set pixels inside the offset to 0 in input_array and known_array
    input_array[:, :, :offset[0]] = 0
    input_array[:, :offset[1], :] = 0

    known_array[:, :, :offset[0]] = 0
    known_array[:, :offset[1], :] = 0

    # set pixels every spacing to 0 along the x-axis
    start_pixel = offset[0]
    # while the starting position + spacing won't be bigger than the dimension N (width)
    while start_pixel + spacing[0] < input_array.shape[2]:
        input_array[:, :, start_pixel + 1: start_pixel + spacing[0]] = 0
        known_array[:, :, start_pixel + 1: start_pixel + spacing[0]] = 0
        start_pixel += spacing[0]

    input_array[:, :, start_pixel + 1: input_array.shape[2]] = 0
    known_array[:, :, start_pixel + 1: input_array.shape[2]] = 0

    # set pixels every spacing to 0 along the y-axis
    start_pixel = offset[1]
    # while the starting position + spacing won't be bigger than the dimension N (width)
    while start_pixel + spacing[1] < input_array.shape[1]:
        input_array[:, start_pixel + 1: start_pixel + spacing[1], :] = 0
        known_array[:, start_pixel + 1: start_pixel + spacing[1], :] = 0
        start_pixel += spacing[1]

    input_array[:, start_pixel + 1: input_array.shape[1], :] = 0
    known_array[:, start_pixel + 1: input_array.shape[1], :] = 0

    # masking the unknown values to be the target using known_array
    target_array = target_array[known_array == 0]

    if np.count_nonzero(known_array) / 3 < 144:
        raise ValueError(f'The number of known pixels after removing must be at least 144 but is '
                         f'{np.count_nonzero(input_array)}')

    return input_array, known_array, target_array

