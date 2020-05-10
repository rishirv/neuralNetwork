import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit as sigmoid

#Neural network class def
class neuralNetwork:

    #initialize network
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes
        self.lr = learningRate
        #Weights matricies
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        #Activation function is sigmoid function
        self.activ_func = sigmoid
    #Train network
    def train(self, inputs_list, targets_list):
        targets = np.array(targets_list, ndmin=2).T
        final_outputs = self.query(inputs_list)
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T,output_errors)
        self.who+=self.lr*np.dot((output_errors*final_outputs*(1-final_outputs)), np.transpose(hidden_outputs))
        self.wih+=self.lr*np.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)), np.transpose(inputs))
    #Query network
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activ_func(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activ_func(final_inputs)

data_file = open("MNIST data/mnist_train_100.csv", 'r')
data_list = data_file.readlines()
data_file.close()

all_values = data_list[0].split(',')
print(all_values[0])
image_array=np.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array, cmap='Greys', interpolation='none')