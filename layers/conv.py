import numpy as np
from utils import pad_2d, output_shape

class Conv2D:
    def __init__(self, num_filters, filter_size, padding=0, stride=1):
        """2D convolutional layer class
        
        Arguments:
            num_filters {int} -- number of convolutional filters in array
            filter_size {int} -- the size of each filter (assuming the filters are square)
        
        Keyword Arguments:
            padding {int} -- zero parring  (default: {0})
            stride {int} -- stride of filter (default: {1})
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.padding = padding
        self.stride = stride
        self.conv_filter = np.random.rand(num_filters, filter_size, filter_size)/(filter_size*filter_size)
    
    def image_patch(self, image):
        """A generator of filter_size x filter_size patches from the input images
        
        Arguments:
            image {np.ndarray} -- input image
        
        Yields:
            [np.ndarray, int, int] -- image patch and its corresponding indexes
        """
        height, width = image.shape
        self.current_image = image
        H_out, W_out = output_shape(height, width, self.filter_size, self.padding, self.stride)
        for j in range(H_out):
            for k in range(W_out):
                image_patch = image[j*self.stride : (j*self.stride + self.filter_size), k*self.stride:(k*self.stride+self.filter_size)]
                yield image_patch, j, k
    
    def forward(self, image):
        """Performs forward pass of an image through the convulutional layer
        
        Arguments:
            image {np.ndarray} -- input image
        
        Returns:
            [np.ndarray] -- output of the layer
        """
        height, width = image.shape
        H_out, W_out = output_shape(height, width, self.filter_size, self.padding, self.stride)
        output = np.zeros((H_out, W_out, self.num_filters))
        padded_image = pad_2d(image, self.padding)
        for patch, i, j in self.image_patch(padded_image):
            output[i,j] = np.sum(patch*self.conv_filter, axis=(1,2))
        return output

    def backward(self, previous_grad, learning_rate):
        """Performs a backward propagation pass through the convolutional layer.
        During the pass the gradient with respect to filter parameters are calculated and 
        the parameters are also updated
        
        Arguments:
            previous_grad {np.ndarray} -- incoming gradient from the upcoming layer
            learning_rate {int} -- the learning rate for updating filter parameters
        
        Returns:
            [np.ndarray] -- backprop gradient
        """
        conv_grad = np.zeros(self.conv_filter.shape)
        for patch, i, j in self.image_patch(self.current_image):
            for k in range(self.num_filters):
                conv_grad[k] += patch*previous_grad[i,j,k]
        
        self.conv_filter -= learning_rate*conv_grad
        return conv_grad




