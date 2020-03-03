import numpy as np

class MaxPooling2D:
    def __init__(self, filter_size):
        """Max Pooling layer class
        
        Arguments:
            filter_size {int} -- size of the max pooling filter
        """
        self.filter_size = filter_size
    
    def image_patch(self, image):
        """Generates an image patch for the maxpooling operation. We did not implement stride and zero
        padding for this operation. 
        
        Arguments:
            image {np.ndarray} -- input tensor
        
        Yields:
            [np.ndarray] -- a patch of the tensor
        """
        out_height = image.shape[0] // self.filter_size
        out_width = image.shape[1] // self.filter_size
        self.current_image = image

        for i in range(out_height):
            for j in range(out_width):
                patch = image[i*self.filter_size:(i+1)*self.filter_size, j*self.filter_size:(j+1)*self.filter_size]
                yield patch, i, j
    
    def forward(self, image):
        """Performs the forward operation for 2D maxpooling
        
        Arguments:
            image {np.ndarray} -- input tensor
        
        Returns:
            [np.ndarray] -- output of max pooling layer
        """
        height, width, num_filters = image.shape
        output = np.zeros((height//self.filter_size, width//self.filter_size, num_filters))

        for patch, i, j in self.image_patch(image):
            output[i,j] = np.amax(patch, axis=(0,1))
        
        return output
    
    def backward(self, pervious_grad):
        """Performs backpropagation through maxpooling layer. Since this step does not have parameters 
        there is no update and only the gradient is returned 
        
        Arguments:
            pervious_grad {np.ndarray} -- gradient from upcoming layer
        
        Returns:
            [np.ndarray] -- backprop gradient
        """
        pooling_grad = np.zeros(self.current_image.shape)
        for patch, i, j in self.image_patch(self.current_image):
            height, width, num_filters = patch.shape
            maximum_val = np.amax(patch, axis=(0,1))

            for n in range(height):
                for m in range(width):
                    for o in range(num_filters):
                        if patch[n,m,o] == maximum_val[o]:
                            pooling_grad[i*self.filter_size+n, j*self.filter_size+m, o] = pervious_grad[i,j,o]
        
        return pooling_grad
