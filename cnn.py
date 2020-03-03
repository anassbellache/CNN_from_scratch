import numpy as np
from layers.conv import Conv2D
from layers.max_pool import MaxPooling2D
from layers.softmax import Softmax
from utils import output_shape

class ConvNet:
    def __init__(self, filter_size, 
                num_filters,
                pool_size, 
                input_shape, 
                out_dim,
                padding=0,
                stride=1):
                """Convolutional neural network with one convolutional layer, a maxpooling layer,
                then a dense and softmax layer.
                
                Arguments:
                    filter_size {int} -- filter size for the convolutional layer
                    num_filters {int} -- number of filters in convulutional layer
                    pool_size {int} -- size of maxpooling filter
                    input_shape {(int, int)} -- width and height of input images
                    out_dim {int} -- number of possible labels
                
                Keyword Arguments:
                    padding {int} -- padding coefficient (default: {0})
                    stride {int} -- stride coefficient (default: {1})
            """
                
                self.conv = Conv2D(num_filters, filter_size, padding, stride)
                self.pooling = MaxPooling2D(pool_size)
                H_1, W_1 = output_shape(input_shape[0], input_shape[1], filter_size, padding, stride)
                H_2, W_2 = H_1//pool_size, W_1//pool_size
                softmax_in = H_2*W_2*num_filters
                self.softmax = Softmax(softmax_in, out_dim)
    
    def forward(self, image):
        """forward pass throught the CNN"""

        out = self.conv.forward(image)
        out = self.pooling.forward(out)
        out = self.softmax.forward(out)
        return out
    
    def backward(self, gradient, learning_rate):
        """Backprop through the CNN"""
        grad_back = self.softmax.backward(gradient, learning_rate)
        grad_back = self.pooling.backward(grad_back)
        grad_back = self.conv.backward(grad_back, learning_rate)
        return grad_back