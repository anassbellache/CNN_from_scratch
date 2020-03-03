import numpy as np
from utils import pad_2d

class Conv2D:
    def __init__(self, num_filters, filter_size, padding=0, stride=1):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.padding = padding
        self.stride = stride
        self.conv_filter = np.random.rand(num_filters, filter_size, filter_size)/(filter_size*filter_size)
    
    def image_patch(self, image):
        height, width = image.shape
        self.current_image = image
        H_out = int((height - self.filter_size)/self.stride) + 1
        W_out = int((width - self.filter_size)/self.stride) + 1
        for j in range(H_out):
            for k in range(W_out):
                image_patch = image[j*self.stride : (j*self.stride + self.filter_size), k*self.stride:(k*self.stride+self.filter_size)]
                yield image_patch, j, k
    
    def forward(self, image):
        #padded_image = pad_2d(image, self.padding)
        padded_image = image
        height, width = padded_image.shape
        H_out = int((height - self.filter_size)/self.stride) + 1
        W_out = int((width - self.filter_size)/self.stride) + 1
        output = np.zeros((H_out, W_out, self.num_filters))
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




