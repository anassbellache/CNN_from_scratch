import numpy as np
from utils import pad_1d, pad_2d

class Conv2D:
    def __init__(self, num_filters, filter_size, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.padding = padding
        self.conv_filter = np.random.rand(num_filters, filter_size, filter_size)/(filter_size*filter_size)
    
    def image_patch(self, image):
        height, width, _ = image.shape
        self.current_image = image
        for j in range(height - self.filter_size + 1):
            for k in range(width - self.filter_size + 1):
                image_patch = image[j : (j + self.filter_size), k:(k+self.filter_size)]
                yield image_patch, j, k
    
    def forward(self, image):
        padded_image = pad_2d(image, self.padding)
        height, width = padded_image.shape
        output = np.zeros((height - self.filter_size + 1, width - self.filter_size + 1, self.num_filters))
        for patch, i, j in self.image_patch(image):
            output[i,j] = np.sum(patch*self.conv_filter, axis=(1,2))
        return output

    def backward(self, previous_grad, learning_rate):
        conv_grad = np.zeros(self.conv_filter.shape)
        for patch, i, j in self.image_patch(self.current_image):
            for k in range(self.num_filters):
                conv_grad[k] += patch*previous_grad[i,j,k]
        
        self.conv_filter -= learning_rate*conv_grad
        return conv_grad




