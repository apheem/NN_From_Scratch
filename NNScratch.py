import numpy as np
import matplotlib
''''a single neuron'''
'''This is also none as dot opperator'''
inputs = [1,2,3,2.5]
weights = [0.2,0.8,-0.5,1]
bias = 2
#outer layer of forward prop
output = (inputs[0]*weights[0]+
          inputs[1]*weights[1]+
          inputs[2]*weights[2]+
          inputs[3]*weights[3]+bias)

print(output)

inputs = [1,2,3,2.5]

weights1 = [0.2,0.8,-0.5,1]
weights2 = [0.5,-0.91,0.26,-0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

outputs = [inputs[0]*weights1[0]+
          inputs[1]*weights1[1]+
          inputs[2]*weights1[2]+
          inputs[3]*weights1[3]+bias1,
          # Neuron 2
          inputs[0]*weights2[0]+
          inputs[1]*weights2[1]+
          inputs[2]*weights2[2]+
          inputs[3]*weights2[3]+bias2,
          # Neuron 3
          inputs[0]*weights3[0]+
          inputs[1]*weights3[1]+
          inputs[2]*weights3[2]+
          inputs[3]*weights3[3]+bias3]

#or

weights = [[0.2,0.8,-0.5,1], [0.5,-0.91,0.26,-0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2,3,0.5]

#Output of current layer
layer_outputs = []
#using zip to go into a list of list to pull deeper values of each weight at each index
#and to connect each bias to a row of weights (layer of neurons)
for neuron_weights, neuron_bias in zip(weights, biases):
    #this would look like [bias1,[weights1]]
    neuron_output = 0
    #going into the each individual weight and nueron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
        print('n before')
        print(neuron_output)
    neuron_output += neuron_bias
   
    layer_outputs.append(neuron_output)
    
'''dot opperator'''
inputs = [1,2,3,2.5]
weights = [0.2,0.8,-0.5,1]
bias = 2
#multiplying iputs by weights like above
outputs = np.dot(inputs, weights) + bias

inputs = [[1,2,3,2.5], [2,5,-1,2], [-1.5,2.7,3.3,-0.8]]
weights = [[0.2,0.8,-0.5,1], [0.5,-0.91,0.26,-0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2,3,0.5]
outputs = np.dot(weights, inputs) + biases #the order matters. The transposed has to be on the right

'''you have to transpose the weights for it to work properly'''
'''making it (n,1) instead of (1,n)'''

a = [[1,2,3]]
b = [[1,2,3]]

a = np.array(a)
b = np.array(b).T
outputs = np.dot(a, b)

inputs = [[1,2,3,2.5], [2,5,-1,2], [-1.5,2.7,3.3,-0.8]]
weights = [[0.2,0.8,-0.5,1], [0.5,-0.91,0.26,-0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2,3,0.5]
outputs = np.dot(inputs, np.array(weights).T) + biases #the order matters. The transposed has to be on the right


'''Adding layers'''
#layer 1
inputs = [[1,2,3,2.5], [2,5,-1,2], [-1.5,2.7,3.3,-0.8]]
weights = [[0.2,0.8,-0.5,1], [0.5,-0.91,0.26,-0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2,3,0.5]

weights2 = [[0.1,-0.14,0.5], [-0.5,0.12,-0.33], [-0.44, 0.73, -0.13]]
biases2 = [-1,2,-0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2)) + biases2

import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data #random dataset gen
import nnfs
nnfs.init()

'''Dense Layer Class'''
class Layer_Dense:
    #weigths and bias init
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
#creating data set  
x, y = spiral_data(samples=100, classes=3)
#creating dense layer with 2 input features and 3 output values
dense1= Layer_Dense(2, 3)
#perforward pass
dense1.forward(x)
print(dense1.output[:5])
print(np.zeros((2,3)))#just to compare the shapes
print(x.shape)

'''Relu function'''
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []
for i in inputs:
    if i > 0:
        output.append(i)
    else:
        output.append(0)
#or
for i in inputs:
    output.append(max(0, i))
#or
output = np.maximum(0, inputs)

class Activation_Relu:
    
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
'''applying activation function to dense model'''
#creating data
x,y = spiral_data(samples= 100, classes= 3)
#setting up the model
dense1 = Layer_Dense(2,3)
activation1 = Activation_Relu()
#forward passing and applying relu
dense1.forward(x)
activation1.forward(dense1.output)
results = activation1.output[:5]
    
'''Softmax function'''
#every value in the vector is an exponiant of E and then devided by the sum
layer_output = [4.8, 1.21, 2.385]
E = 2.718281846
exp_values = []
for output in layer_output:
    exp_values.append(E ** output) 
#normalized values
norm_base = sum(exp_values)
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)
print(norm_values)

#or numpy version

exp_values = np.exp(layer_output)
norm_values = exp_values / np.sum(exp_values)

#need to - max value in the inputs to
exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

class Activation_Softmax:
    def forward(self, inputs):
        #getting unnormalized values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #normalization
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
#        
 #       
  #        
   #     
'''applying everything so far'''
import numpy as np
import  nnfs
from nnfs.datasets import spiral_data
nnfs.init()

#dense layer
class Layer_Dense:
    #init layer
    def __init__(self, n_inputs, n_neurons):
        #init biases and weights
        self.weights = 0.01* np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        #calculating values from inputs, neurons and biases at each neuron
        self.output = np.dot(inputs, self.weights)+ self.biases
        
#relu
class Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) #if the input is < 0 then output = 0 if input > 0 output = input
        
#softmax
class Softmax:
    def forward(self, inputs):
        #unnormalized values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 
        #normalize
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims= True)
        
        self.output = probabilities
        
#create dataset
x,y = spiral_data(100, 3)
#creating layer
dense1 = Layer_Dense(2,3)
#
activation1 = Relu()
#creating a dense layer to match the output dim of the preview output and determining output neurons 
dense2 = Layer_Dense(3,3)
#input dense1 2,3 = 3,3 dense2 3,3  
activation2 = Softmax()
#each new step takes in the output of the last
dense1.forward(x)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)


print(activation2.output[:5])
    #
   #
  #
 # 

#loss = math.log(activation_output[0])*target_output[0]+ math.log(activation_output[1])*target_output[1]
'''how log works'''
import math
b = 5.2
logb = np.log(b)
 #or
unlog = math.e ** logb

'''Calcuating catagorical cross entropy with batch softmax output and finding the mean loss'''
softmax_outputs = [[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]]
class_targets = [0,1,1] # 0 = dog 1 = cat first indices is dog second and third cat

for targ_inx, distribution in zip(class_targets, softmax_outputs):
    print(distribution[targ_inx]) #softmax_outputs at class_targets index

#numpy version
softmax_outputs = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
class_targets = [0,1,1]
print(softmax_outputs[range(len(softmax_outputs)), class_targets])
#applying log to find loss of each 'guess'
neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
#the higher the -log output the further away the pred is from correct
#applying mean
average_loss = np.mean(neg_log)

'''calculating batch of 1 hot encoded targets'''
import numpy as np

softmax_outputs = [[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]]
class_targets = np.array([[1,0,0],[0,1,0],[0,1,0]])
#checking to see if the target is 1 hot encoded matrix or just a list of binary results
if  len(class_targets.shape) == 1:
    correct_confidence = softmax_outputs[range(len(softmax_outputs)), class_targets]
elif len(class_targets.shape) == 2:
    correct_confidence = np.sum(softmax_outputs * class_targets, axis=1)
    
#calc loss
neg_log = -np.log(correct_confidence)

average_loss = np.mean(neg_log)


#        
 #       
  #        
   #  
'''everything to thing point'''
'''forward prop'''
import numpy as np
import nnfs
from nnfs.datasets  import spiral_data

nnfs.init()

class Layer_Dense:
    
    def __init__ (self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
class Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        probabilities = exp_values/ np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities
        
class Loss:
    
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
        
class Loss_CatagoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_target):
        samples = len(y_pred)
        #clipping the data to prevent dividing by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_target.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_target]
        elif len(y_target.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped*y_target, axis=1)
        
        neg_log_likelyhood = -np.log(correct_confidence)
        return neg_log_likelyhood
    
x, y = spiral_data(100, 3)

dense1 = Layer_Dense(2,3)
activation1 = Relu()
dense2 = Layer_Dense(3,3)
activation2 = Softmax()
loss_function = Loss_CatagoricalCrossEntropy() 

dense1.forward(x)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output)
loss = loss_function.calculate(activation2.output, y)

prediction = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

print(loss)
print('accuracy:', np.mean(prediction == y))
    #
   #
  #
 # 
 
'''accuracy'''
import numpy as np
softmax_outputs = [[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]]
class_targets = np.array([[1,0,0],[0,1,0],[0,1,0]])

pred = np.argmax(softmax_outputs, axis=1 )
#only needed if its a matrix
class_targets = np.argmax(class_targets, axis=1)
print(np.mean(pred == class_targets))

'''back prop'''
# sample back prop function
#currentLayer/previousLayer * previousLayer/prepreviousLayer * etc
#dinputs = inputs for next layer in the chain
#dvalues = values from previous layer in the chain
class Layer_Dense:
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros(1, neurons)
        
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues):
        #since you multiply the weights and the inputs you take other and multiply it by the dvalues from the previous layer
        #nd keep the same dim
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdim=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        
class Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        
'''softmax and cat cross-entr back prop'''
class Loss:
    
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
        
class Loss_CatagoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_target):
        samples = len(y_pred)
        #clipping the data to prevent dividing by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_target.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_target]
        elif len(y_target.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped*y_target, axis=1)
        
        neg_log_likelyhood = -np.log(correct_confidence) 
        return neg_log_likelyhood
    
    def backward(self, dvalues, y_target):
        samples = len(dvalues)
        labels = len(dvalues[0])
        #if not 2 dim
        if len(y_target.shape) == 1:
            #converting it into a 2d array
            y_target = np.eye(labels)[y_target] #number columns and rows. returns nxn at where y_target = 1
        self.dinputs = -y_target/dvalues
        #normalize the gradient
        self.dinputs = self.dinputs/samples
        
#softmax back prop
class Softmax:
    def forward(self, inputs):
    
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        probabilities = exp_values/ np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities
        
    def backward(self, dvalues):
        #init an array
        self.dinputs = np.empty_like(dvalues)
        
        #enumerating outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #flattening the array
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            #calc sample-wise gradient and adding it to the array
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
            
'''code up to now'''
'''updting weights without optimizer using gradient decent'''
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

#Dense layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # init weights with randn which takes in dims
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.inputs = inputs #to remember the inputs for grad decent later
        self.outputs = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues):#dvalues are dinputs( deriv outputs from the layer next in line)
        self.dweights = np.dot(self.inputs.T, dvalues) #since in this layer forard pass we multiply
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.dweights.T)
        
class Relu:
    def forward(self, inputs):
        #saving a copy for grad decent later on
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs) #returns everything from 0 and up and converts neg to 0
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        #if previous inputs are negitave make them 0 and keep anything above 
        self.dinputs[self.inputs <= 0] = 0 #any spot where inputs are less than 0 set to 0
        
class Softmax:
    def forward(self, inputs):
        #saving copy for back prop later
        self.inputs = inputs
        #unnormalized prob
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))# substracting to keep it close to 0
        probability = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probability
        
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        
        #going thru each value and perfroming math 1 pair at a time
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)): #enumerate is simular to range as it counts iterations
            #flatten array
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T) #diagflat does np.eye * input
            
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
            
class Loss:
    def calculate(self, outputs, y):
        sample_loss = self.forward(outputs, y)
        data_loss = np.mean(sample_loss)
        
        return data_loss

class Loss_CrossEntro(Loss):
    def forward(self, y_pred, y_target):
        samples = len(y_pred)
        #clipping the data to prevent dividing by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_target.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_target]
        elif len(y_target.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped*y_target, axis=1)
        
        neg_log_likelyhood = -np.log(correct_confidence) 
        return neg_log_likelyhood
    
    def backward(self, dvalues, y_target):
        samples = len(dvalues)
        #number of  labels or rows 
        labels = len(dvalues[0])
        #if not 2 dim
        if len(y_target.shape) == 1:
            #converting it into a 2d array or 1 hot encoding them if they are not already
            y_target = np.eye(labels)[y_target] #number columns and rows. returns nxn at where y_target = 1
        self.dinputs = -y_target/dvalues
        #normalize the gradient
        self.dinputs = self.dinputs/samples
        
#combining softmax classifer with  cross entropy for faster processing (7x faster)
class Softmax_CrossEntro():
    def __init__(self):
        self.activation = Softmax()
        self.loss = Loss_CrossEntro()
        
    def forward(self, inputs, y_target):
        self.activation.forward(inputs)
        self.output = self.activation.output
        
        return self.loss.calculate(self.output, y_target)
    
    def backward(self, dvalues, y_target):
        sample = len(dvalues)
        #if labels are 1 hot incoded convert them to a 1d array
        if len(y_target.shape) == 2:
            y_target = np.argmax(y_target, axis=1)
            
        self.dinputs = dvalues.copy()
        print('soft dinputs', self.dinputs[0:5])
        self.dinputs[range(sample), y_target] -= 1
        print('y_target: ', y_target)
        print('dinputs after range -=1', self.dinputs)
        self.dinputs = self.dinputs/sample
        print('dinputs after norm', self.dinputs[:5])

x,y = spiral_data(100, 3)
print('x: ', x[:5])
print('y: ', y[:5])
#layer with 2 features and 3 output values or neurons
dense1 = Layer_Dense(2, 3)  
activation1 = Relu()
#layer with the inputs of previous layer and then how ever many neurons you want or outputs
dense2 = Layer_Dense(3,3)      
loss_activation = Softmax_CrossEntro() #since softmax is a classifier in the final dense layer we can combine them

dense1.forward(x)
activation1.forward(dense1.outputs)
dense2.forward(activation1.outputs)
loss = loss_activation.forward(dense2.outputs, y)

print('loss act output: ', loss_activation.output[:5])
print('Loss: ', loss)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

print('acc: ', accuracy)

loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print('D weights and biases')
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)
'''You minus 1 in the soft_crossentro.backward to the place where the correct values should be '''

