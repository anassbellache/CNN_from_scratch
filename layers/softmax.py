import numpy as np

class Softmax:
    def __init__(self, input_size, output_size):
        """Softmax layer
        
        Arguments:
            input_size {int} -- input size of the layer
            output_size {int} -- dimension of the labels space
        """
        self.weights = np.random.randn(input_size, output_size)/input_size
        self.bias = np.zeros(output_size)
    
    def forward(self, image):
        """Performs forward pass through the layer.
        The pass involves flattening the input tensor, going through a fully connnected layer 
        followed by a softmax function
        
        Arguments:
            image {np.ndarray} -- input tennor
        
        Returns:
            [type] -- [description]
        """
        self.current_image = image
        self.flattened_image = image.flatten()
        self.output = np.dot(self.flattened_image, self.weights) + self.bias
        return np.exp(self.output)/np.sum(np.exp(self.output), axis=0)
    
    def backward(self, pervious_grad, learning_rate):
        """Performs backward propagation through the layer
        
        Arguments:
            pervious_grad {np.ndarray} -- gradient from upcoming layer
            learning_rate {int} -- rate for parameter updates
        
        Returns:
            [np.ndarray] -- backprop gradient
        """
        for i, grad in enumerate(pervious_grad):
            if grad == 0:
                continue
            numerator = np.exp(self.output)
            denom = np.sum(numerator)

            d_softmax_dz = -numerator[i]*numerator / (denom**2)
            d_softmax_dz[i] = numerator[i]*(denom - numerator[i])/(denom**2)

            dz_dw = self.flattened_image
            dz_db = 1
            dz_d_input = self.weights

            dL_dz = grad*d_softmax_dz
            dL_dw = dz_dw[np.newaxis].T @ dL_dz[np.newaxis]
            dL_db = dL_dz*dz_db
            dL_d_input = dz_d_input @ dL_dz
            
            self.weights -= learning_rate*dL_dw
            self.bias -= learning_rate*dL_db

            return dL_d_input.reshape(self.current_image.shape)


