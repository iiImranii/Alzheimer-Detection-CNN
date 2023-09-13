import math
import numpy as np
import numba
import cv2
from numba import jit, cuda
from timeit import default_timer as timer   
import time

class ConvolutionalLayer:
    def __init__(self, kernal_size, num_of_kernels, stride, padding=1, lambd=0.01):
        # Initializes the ConvolutionalLayer with kernel size, number of kernels, stride, padding, and lambda value
        # kernel_size: Size of the convolutional kernel
        # num_of_kernels: Number of kernels in the layer
        # stride: Stride value for convolution
        # padding: Padding value for convolution (default: 1)
        # lambd: Lambda value for regularization (default: 0.01)

        self.num_of_kernals = num_of_kernels
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = padding
        self.lambd = lambd
        self.kernels = np.random.randn(kernal_size, kernal_size, num_of_kernels) * np.sqrt(2 / (kernal_size**2 + num_of_kernels))
        
    def Convolute(self, img):
        # Applies convolution to the input image and yields feature maps along with their corresponding positions
        # img: Input image
        
        image_height, image_width, z = img.shape
        padded_image = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        
        for h in range(0, image_height + self.padding*2 - self.kernal_size + 1, self.stride):
            for w in range(0, image_width + self.padding*2 - self.kernal_size + 1, self.stride):
                feature_map = padded_image[h:(h+self.kernal_size), w:(w+self.kernal_size)]
                yield feature_map, h, w
    
    def forward_prop(self, img):
        # Performs forward propagation through the ConvolutionalLayer
        # img: Input image
        
        image_height, image_width, _ = img.shape
        conv_out_height = int((image_height + 2*self.padding - self.kernal_size) / self.stride) + 1
        conv_out_width = int((image_width + 2*self.padding - self.kernal_size) / self.stride) + 1
        conv_out = np.zeros((conv_out_height, conv_out_width, self.num_of_kernals))
        
        for f_map, h, w in self.Convolute(img):
            conv_out[h // self.stride, w // self.stride] = np.sum(f_map.transpose((2, 0, 1)).dot(self.kernels))
            conv_out[h // self.stride, w // self.stride] = np.maximum(conv_out[h // self.stride, w // self.stride], 0)
        
        self.image = conv_out
        conv_out -= np.min(conv_out)
        max_value = np.max(conv_out)
        
        if max_value != 0:
            conv_out /= max_value
        else:
            conv_out = np.zeros_like(conv_out)
        
        return conv_out
    
    def back_prop(self, gradient_out, learning_rate):
        # Performs backward propagation through the ConvolutionalLayer
        # gradient_out: Gradient of the loss with respect to the layer's output
        # learning_rate: Learning rate for gradient descent
        
        filter_grad = np.zeros((self.kernels.shape))
        input_grad = np.zeros((self.image.shape[0]+2*self.padding, self.image.shape[1]+2*self.padding, self.image.shape[2]))
        
        for feature_map, h, w in self.Convolute(self.image):
            for f in range(self.num_of_kernals):
                h_start, w_start = h*self.stride, w*self.stride
                h_end, w_end = h_start + self.kernal_size, w_start+self.kernal_size
                filter_grad[:, :, f] += gradient_out[h // self.stride, w // self.stride, f] * feature_map[:, :, f]
                input_grad[h_start:h_end, w_start:w_end, :] += np.dot(self.kernels[:, :, f, np.newaxis], gradient_out[h // self.stride, w // self.stride, f]) #self.kernels[:, :, f] * gradient_out[h // self.stride, w // self.stride, f]
        if self.padding > 0:
            input_grad = input_grad[self.padding:-self.padding, self.padding:-self.padding, :]
            
        self.kernels -= (learning_rate * filter_grad )     
        #self.kernels -= (learning_rate * filter_grad + self.lambd * self.kernels)      
        #print("Output Conv Layer: "+str(input_grad.shape))
        #print("Conv out Max Val: "+str(np.amax(input_grad)))
        #print("-------------")
        return input_grad
   
   
    
class MaxPoolingLayer:
    def __init__(self, kernal_size, stride):
        # Initializes the MaxPoolingLayer with kernel size and stride
        # kernal_size: Size of the pooling kernel
        # stride: Stride value for pooling
        
        self.kernal_size = kernal_size
        self.stride = stride
    
    def extract_regions(self, img):
        # Extracts non-overlapping regions from the image for pooling
        # img: Input image
        
        h = (img.shape[0] - self.kernal_size) // self.stride
        w = (img.shape[1] - self.kernal_size) // self.stride
        
        for i in range(h):
            for j in range(w):
                region = img[(i*self.kernal_size):(i*self.kernal_size+self.kernal_size), (j*self.kernal_size): (j*self.kernal_size+self.kernal_size)]
                yield region, i, j
    
    def forward_prop(self, image):
        # Performs forward propagation through the MaxPoolingLayer
        # image: Input image
        
        self.prev_image = image
        image_height, image_width, num_of_kernels = image.shape
        max_pool_map = np.zeros(((image_height - self.kernal_size) // self.stride, (image_width - self.kernal_size) // self.stride, num_of_kernels))
        
        for f_map, h, w in self.extract_regions(image):
            max_pool_map[h, w] = np.amax(f_map, axis=(0, 1))
        
        return max_pool_map
        
    def back_prop(self, loss_gradient): 
        # Performs backward propagation through the MaxPoolingLayer
        # loss_gradient: Gradient of the loss with respect to the layer's output
        
        gradient_out = np.zeros(self.prev_image.shape)
        
        for region, i, j in self.extract_regions(self.prev_image):
            h, w, f = region.shape
            amax = np.amax(region, axis=(0, 1))
            
            for pool_h in range(h):
                for pool_w in range(w):
                    for filters in range(f):
                        if region[pool_h, pool_w, filters] == amax[filters]:
                            if loss_gradient is not None:
                                gradient_out[i*self.kernal_size+pool_h, j*self.kernal_size+pool_w] = loss_gradient[i, j, filters]
        
        return gradient_out
    
  
    
    
class DropoutLayer:
    def __init__(self, p):
        # Constructor for DropoutLayer class
        # p: The probability of dropping out a unit in the layer
        self.p = p
        self.mask = None  # Variable to store the dropout mask
        
    def forward_prop(self, image):
        # Performs forward propagation through the dropout layer
        # image: Input image or activation from the previous layer
        in_train = True  # Flag to indicate if the model is in training mode
        if in_train == True:
            # Generate a dropout mask during training
            self.mask = np.random.binomial(1, 1 - self.p, size=image.shape) / (1 - self.p)
            return image * self.mask  # Apply dropout by element-wise multiplication
        else:
            return image  # Return the image as is during inference
        
    def back_prop(self, loss_grad):
        # Performs backward propagation through the dropout layer
        # loss_grad: Gradient of the loss with respect to the output of the dropout layer
        
        return loss_grad * self.mask  # Backpropagate the gradients multiplied by the dropout mask
    
    
    
class FlattenLayer:
    def __init__(self):
        # Constructor for FlattenLayer class
        pass
    def forward_prop(self, inputs):
        # Performs forward propagation through the flatten layer
        # inputs: Input tensor or activation from the previous layer
        self.inputs_shape = inputs.shape  # Store the shape of the input tensor
        self.output = inputs.flatten()  # Flatten the input tensor
        return self.output
    def back_prop(self, dvalues):
        # Performs backward propagation through the flatten layer
        # dvalues: Gradient of the loss with respect to the output of the flatten layer
        return dvalues.reshape(self.inputs_shape)  # Reshape the gradient to match the input shape
    
class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.biases = np.zeros(output_size)
        self.activation = None

    def forward_prop(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        if self.activation == 'relu':
            self.output = np.maximum(self.output, 0, self.output)
        return self.output

    def back_prop(self, dvalues, learning_rate):
        if self.activation == 'relu':
            dvalues = np.where(self.output <= 0, 0, dvalues)

        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases

        return self.dinputs
    
    
class SoftmaxDenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        self.bias = np.zeros(output_size)

    def forward_prop(self, image):
        # reshape the image to a 2D array
        self.last_image_2d = image
        forward_output = np.dot(image, self.weights) + self.bias
        self.prev_result = forward_output
        exp_out = np.exp(forward_output-np.max(forward_output))

        softmax_activation = exp_out / np.sum(exp_out)
        return softmax_activation
    
    def back_prop(self, forward_result_grad, learning_rate):
        gradient_image = np.zeros(self.last_image_2d.shape)
        for i, gradient in enumerate(forward_result_grad):
            #We want to calculate gradient for the correct class (nonzero value)
            if gradient == 0:
                continue
            last_forward_result_exp = np.exp(self.prev_result)
            S = np.sum(last_forward_result_exp)
            gradient_forward = -last_forward_result_exp[i] * last_forward_result_exp / (S**2)
            gradient_forward[i] = last_forward_result_exp[i] * (S- last_forward_result_exp[i]) / (S**2)
            gradient_loss = gradient * gradient_forward
            gradient_weights = self.last_image_2d[np.newaxis].T @ gradient_loss[np.newaxis] 
            gradient_bias = gradient_loss * 1
            gradient_image += self.weights @ gradient_loss


            self.weights -= learning_rate * gradient_weights
            self.bias -= learning_rate * gradient_bias

        # Return shape to the same as it was originally in the forward propogation
        return gradient_image.reshape(self.last_image_2d.shape)
    
        
        
def Predict(file_path,model_info):
    img = cv2.imread(file_path)
    
    # Downsample image size for predicting
    if "input_size" in model_info:
        img = cv2.resize(img, (model_info["input_size"], model_info["input_size"]))
    else:
        img = cv2.resize(img, (208, 208))
        
    foundFlatten = False
    for i in range(len(model_info["model"])):
        if type(model_info["model"][i]) == FlattenLayer:
            foundFlatten = True
    
    print("Found flatten layer: "+str(foundFlatten))
    if foundFlatten == False:
        print("Inserting flattening")
        dense_layer_index = len(model_info["model"]) - 1
        model_info["model"].insert(dense_layer_index, FlattenLayer())
    input = img/255.
    for layer in model_info["model"]:
        print(layer)
        output = layer.forward_prop(input)
    return output
    
    
    
    
def Validate(img, label, layers):
    # Validates the network on a single image-label pair
    # img: Input image
    # label: Ground truth label of the image
    # layers: List of layers in the network
    
    output = img / 255.  # Normalize the input image
    for layer in layers:
        output = layer.forward_prop(output)  # Perform forward propagation through each layer
    
    loss = -np.log(np.maximum(output[label], 1e-10))  # Calculate the loss using cross-entropy
    accuracy = 1 if np.argmax(output) == label else 0  # Calculate the accuracy
    
    print("Correct label: " + str(label))
    print(output)
    
    return loss, accuracy
    
     
def TrainNetwork(img, label, layers, label_table, learning_rate=.0015):
    # Trains the network on a single image-label pair
    # img: Input image
    # label: Ground truth label of the image
    # layers: List of layers in the network
    # label_table: Table mapping labels to class names
    # learning_rate: Learning rate for gradient descent (default: 0.0015)
    output = img / 255.  # Normalize the input image
    for layer in layers:
        output = layer.forward_prop(output)  # Perform forward propagation through each layer

    loss = -np.log(np.maximum(output[label], 1e-10))  # Calculate the loss using categorical cross-entropy
    accuracy = 1 if np.argmax(output) == label else 0  # Calculate the accuracy
    
    print("Correct label: " + str(label))
    print(output)
    
    gradient = np.zeros(4)  # Initialize the gradient
    
    if output[label] == 0:
        gradient[label] = 0
    else:
        gradient[label] = -1 / output[label]  # Calculate the gradient of the loss with respect to the output
    
    for layer in layers[::-1]:
        # Perform backward propagation through each layer
        if type(layer) in [ConvolutionalLayer, SoftmaxDenseLayer, DenseLayer]:
            gradient = layer.back_prop(gradient, learning_rate)
        elif type(layer) in [MaxPoolingLayer, DropoutLayer, FlattenLayer]:
            gradient = layer.back_prop(gradient)
            
    return loss, accuracy