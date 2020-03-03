import numpy as np

def pad_2d(input_, padding):
    """Performs zero padding of a 2D input image
    
    Arguments:
        input_ {np.ndarray} -- input image 
        padding {int} -- padding coefficient
    
    Returns:
        [np.ndarray] -- zero-padded image
    """
    if padding == 0:
        return input_
    side_pad = np.zeros((input_.shape[0], padding))
    side_padded = np.concatenate([side_pad, input_, side_pad])
    ceiling_pad = np.zeros(padding, input_.shape[1] + 2*padding)
    full_padded = np.concatenate([ceiling_pad, side_padded, ceiling_pad],axis=1)
    return full_padded

def output_shape(input_height, input_width, filter_size, padding=0, stride=1):
    """Calculates the first two dimensions of the shape of an output of a convolutional layer
    
    Arguments:
        input_height {int} -- height of the input image
        input_width {int} -- width of the input image
        filter_size {int} -- size of the filter (the filter is assumed to be square)
    
    Keyword Arguments:
        padding {int} -- zero-padding coefficient (default: {0})
        stride {int} -- stride coefficient (default: {1})
    
    Returns:
        [int, int] -- height and width of the output array of the conv filter
    """
    output_height = int((input_height + 2*padding - filter_size)/stride) + 1
    output_width = int((input_width + 2*padding - filter_size)/stride) + 1
    return output_height, output_width