import numpy as np

def pad_1d(input_, padding):
    zeros = np.array([0])
    zeros = np.repeat(zeros, padding)
    return np.concatenate([zeros, input_, zeros])

def pad_2d(input_, padding):
    if padding == 0:
        return input_
    side_pad = np.zeros((input_.shape[0], padding))
    side_padded = np.concatenate([side_pad, input_, side_pad])
    ceiling_pad = np.zeros(padding, input_.shape[1] + 2*padding)
    full_padded = np.concatenate([ceiling_pad, side_padded, ceiling_pad],axis=1)
    return full_padded

def batch_pad_2d(batch, padding):
    outs = [pad_2d(elt, padding) for elt in batch]
    return np.stack(outs)

def output_shape(input_height, input_width, filter_size, padding=0, stride=1):
    output_height = int((input_height + 2*padding - filter_size)/stride) + 1
    output_width = int((input_width + 2*padding - filter_size)/stride) + 1
    return output_height, output_width