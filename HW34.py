
# coding: utf-8

# In[1]:

#Make batch 
#Add momentum
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit as sigmoid
from sklearn.model_selection import train_test_split


# In[2]:

#Neural network class def
class neuralNetwork:

    #initialize network
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        hiddenNodes.insert(0, inputNodes)
        hiddenNodes.append(outputNodes)
        self.nodes = hiddenNodes
        self.n = len(self.nodes) - 1 #Number of layers, not including input
        self.batchSize = 0
        self.lr = learningRate
        #Weights matricies created with random numbes
        #Chosen from a normal distribution with center 0 and std dev (1/sqrt(n))
        #Where n is number of nodes inputting into that neuron
        self.weights = []
        self.gradients = []
        for i in range(self.n):
            self.weights.append(np.random.normal(0.0, pow(self.nodes[i+1], -0.5), (self.nodes[i+1], self.nodes[i])))
            self.gradients.append(np.zeros((self.nodes[i+1], self.nodes[i])))
        self.errors = [0 for i in range(self.n+1)]
        self.outputs = [0 for i in range(self.n+1)]

    def gradient(self, i):
        return self.lr * np.dot((self.errors[i+1]*self.outputs[i+1]*(1-self.outputs[i+1])), self.outputs[i].T)
            
    #Train network
    def train(self, inputs_list, targets_list):
        self.batchSize += 1
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        self.query(inputs_list)
        self.errors[self.n] = self.outputs[self.n] - targets
        #Propogate errors backwards
        for i in range(self.n, 1, -1):
            self.errors[i-1] = np.dot(self.weights[i-1].T, self.errors[i])
        #Update weights given errors
        for i in range(self.n):
            self.gradients[i] -= self.lr * self.gradient(i)
        return self.outputs

    def update(self):
        for i in range(self.n):
            self.weights[i] += self.gradients[i]/self.batchSize
            self.gradients[i].fill(0)
        self.batchSize = 0
    
    #Query network
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        self.outputs[0] = inputs
        for i in range(self.n):
            self.outputs[i+1] = sigmoid(np.dot(self.weights[i], self.outputs[i]))
        return self.outputs


# In[3]:

#Read the training data
X = np.genfromtxt("MNIST data/MNISTnumImages5000.txt", dtype='float64')
y = np.genfromtxt("MNIST data/MNISTnumLabels5000.txt", dtype=int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
train_size = len(X_train)
test_size = len(X_test)

# image_array = np.asfarray(data_list[0].split()).reshape((28, 28)).T
# plt.imshow(image_array, cmap='Greys', interpolation='none')


# In[37]:

# ------------------------------ MNIST Recognition ----------------------------------------------
hiddenNodes = [150]
eps = [] #Epochs for error graph
ers = [] #Errors for error graph

#Create the network
n = neuralNetwork(784, hiddenNodes[:], 10, .45)
correct = 0
conf_train = np.zeros((10, 10), dtype=int)

#Train the network
epochs = 11
for e in range(epochs):
    correct = 0
    conf_train = np.zeros((10, 10), dtype=int)
    for i in range(train_size):
        #All the output nodes are 0.01, except for the correct value, which is 0.99
        targets = np.zeros(10)+0.01
        targets[y_train[i]]=0.99
        predicted = np.argmax(n.train(X_train[i], targets)[-1])
        n.update()
        conf_train[predicted][y_train[i]]+=1
        correct += (predicted == y_train[i])
    print(e, correct/train_size)
    eps.append(e)
    ers.append(1-correct/train_size)
        
hr_train = correct/train_size


# In[38]:

#Test data
conf_test = np.zeros((10, 10), dtype=int)
correct = 0

for i in range(test_size):
    predicted = np.argmax(n.query(X_test[i])[-1])
    conf_test[predicted][y_test[i]]+=1
    if predicted == y_test[i]:
        correct+=1

hr_test = correct/test_size

print("Train Data Hit Rate:", hr_train)
print("Train Data Confusion Matrix:")
print(conf_train)
print("Test Data Hit rate:", hr_test)
print("Test Data Confusion Matrix:")
print(conf_test)
plt.plot(eps, ers)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Train Data Error by Epoch')
plt.show()


# In[6]:

# ----------------------------------- MINST Autoencoder -------------------------------
ae = neuralNetwork(784, hiddenNodes[:], 784, 0.01)

eps = []
ers = []
err_train = 0
err_digit_train = np.zeros(10, dtype=int)
num_digit_train = np.zeros(10, dtype=int)
epochs = 6

#Train the network
for e in range(epochs):
    err_train = 0
    err_digit_train = np.zeros(10, dtype=int)
    num_digit_train = np.zeros(10, dtype=int)
    for i in range(train_size):
        err = np.sum((X_train[i] - ae.train(X_train[i], X_train[i])[-1])**2)/2 #Add J2 loss function
        ae.update()
        err_train += err
        err_digit_train[y_train[i]] += err
        num_digit_train[y_train[i]] += 1
    print(e, err_train/train_size)
    eps.append(e)
    ers.append(err_train/train_size)


# In[7]:

#Test data
err_test = 0
err_digit_test = np.zeros(10, dtype=int)
num_digit_test = np.zeros(10, dtype=int)

for i in range(test_size):
    err = np.sum((X_test[i] - ae.query(X_test[i])[-1])**2)/2 #Add J2 loss function
    err_test += err
    err_digit_test[y_test[i]] += err
    num_digit_test[y_test[i]] += 1

err_train = err_train/train_size
err_test = err_test/test_size

print("Average Train Data Error:", err_train)
print("Average Test Data Error:", err_test)
plt.plot(eps, ers)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Average Train Data Error by Epoch')
plt.show()
fig, ax = plt.subplots()
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
width = .2
plt.bar(x-width/2, err_digit_train/num_digit_train, width=width, label='Train', align='center')
plt.bar(x+width/2, err_digit_test/num_digit_test, width=width, label='Test', align='center')
ax.set_xticks(x)
ax.set_xticklabels(x)
plt.autoscale(tight=True)
plt.legend()
plt.title('Average Error by Digit')
plt.xlabel("Digit")
plt.ylabel("Error")
plt.show()


# In[8]:

y = 10
x = hiddenNodes[0]//y

fig, ax = plt.subplots(x,y)
for i in range(x):
    for j in range(y):
        ax[i,j].imshow(ae.weights[0][y*i+j].reshape(28, 28), cmap='Greys', interpolation='none')
        ax[i,j].axis('off')
fig.subplots_adjust(hspace=0.1, wspace=.01)
plt.savefig('AutoEncoderFeatures.jpg')
plt.show()


# In[9]:

#Regularized neural network class def
class regularizedNeuralNetwork(neuralNetwork):
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate, rho, beta, lambd):
        self.nodeActivation = [np.zeros(layer) for layer in hiddenNodes] #Only for hidden nodes
        self.avgNodeActivation = [np.full(layer, rho) for layer in hiddenNodes] #Initialize with sparseness target
        self.rho = rho
        self.beta = beta
        self.lambd = lambd
        neuralNetwork.__init__(self, inputNodes, hiddenNodes, outputNodes, learningRate)
        
    def update(self):
        self.avgNodeActivation = [layer/self.batchSize for layer in self.nodeActivation]
        self.nodeActivation = [np.zeros(layer) for layer in self.nodes[1:-1]]
        neuralNetwork.update(self)
        
    def gradient(self, i):
        if i != 0: #Update node activations only for hidden nodes
            self.nodeActivation[i-1] += self.outputs[i].flatten()
        deriv = self.outputs[i+1]*(1-self.outputs[i+1])
        if i != self.n-1: #Sparse KL divergence only for hidden nodes
            sparse = ((1-self.rho)/(1-self.avgNodeActivation[i])-self.rho/self.avgNodeActivation[i]).reshape((self.nodes[i+1],1))
        else:
            sparse = 0 #No sparseness constrain on output nodes
        delta = (self.errors[i+1]+self.beta*sparse)*deriv
        return np.dot(delta, self.outputs[i].T) + self.lambd*self.weights[i] #Weight decay


# In[10]:

#----------------------------------MNIST Autoencoder with Regularization --------------------------------
rae = regularizedNeuralNetwork(784, hiddenNodes[:], 784, .08, .05, .8, .001)

eps = []
ers = []
err_train = 0
err_digit_train = np.zeros(10, dtype=int)
num_digit_train = np.zeros(10, dtype=int)
epochs = 21

#Train the network
for e in range(epochs):
    err_train = 0
    err_digit_train = np.zeros(10, dtype=int)
    num_digit_train = np.zeros(10, dtype=int)
    for i in range(train_size):
        err = np.sum((X_train[i] - rae.train(X_train[i], X_train[i])[-1])**2)/2 #Add J2 loss function
        err_train += err
        err_digit_train[y_train[i]] += err
        num_digit_train[y_train[i]] += 1
    rae.update()
    print(e, err_train/train_size)
    eps.append(e)
    ers.append(err_train/train_size)


# In[11]:

#Test data
err_test = 0
err_digit_test = np.zeros(10, dtype=int)
num_digit_test = np.zeros(10, dtype=int)

for i in range(test_size):
    err = np.sum((X_test[i] - rae.query(X_test[i])[-1])**2)/2 #Add J2 loss function
    err_test += err
    err_digit_test[y_test[i]] += err
    num_digit_test[y_test[i]] += 1

err_train = err_train/train_size
err_test = err_test/test_size

print("Average Train Data Error:", err_train)
print("Average Test Data Error:", err_test)
plt.plot(eps, ers)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Average Train Data Error by Epoch')
plt.show()
fig, ax = plt.subplots()
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
width = .2
plt.bar(x-width/2, err_digit_train/num_digit_train, width=width, label='Train', align='center')
plt.bar(x+width/2, err_digit_test/num_digit_test, width=width, label='Test', align='center')
ax.set_xticks(x)
ax.set_xticklabels(x)
plt.autoscale(tight=True)
plt.legend()
plt.title('Average Error by Digit')
plt.xlabel("Digit")
plt.ylabel("Error")
plt.show()


# In[12]:

y = 10
x = hiddenNodes[0]//y

fig, ax = plt.subplots(x,y)
for i in range(x):
    for j in range(y):
        ax[i,j].imshow(rae.weights[0][y*i+j].reshape(28, 28), cmap='Greys', interpolation='none')
        ax[i,j].axis('off')
fig.subplots_adjust(hspace=0.1, wspace=.01)
plt.savefig('RegularizedAutoEncoderFeatures.jpg')
plt.show()


# In[20]:

#-----------------------------------------MINST Recognition with autoencoder features------------------------------------------
na = neuralNetwork(hiddenNodes[0], [], 10, .4)
eps = [] #Epochs for error graph
ers = [] #Errors for error graph

correct = 0
conf_train = np.zeros((10, 10), dtype=int)

#Train the network
epochs = 201
for e in range(epochs):
    correct = 0
    conf_train = np.zeros((10, 10), dtype=int)
    for i in range(train_size):
        #All the output nodes are 0.01, except for the correct value, which is 0.99
        targets = np.zeros(10)
        targets[y_train[i]]=1
        predicted = np.argmax(na.train(sigmoid(np.dot(ae.weights[0], X_train[i])), targets)[-1])
        na.update()
        conf_train[predicted][y_train[i]]+=1
        correct += (predicted == y_train[i])
    if(e%10 == 0):
        print(e, correct/train_size)
        eps.append(e)
        ers.append(1-correct/train_size)
    
hr_train = correct/train_size


# In[21]:

#Test data
conf_test = np.zeros((10, 10), dtype=int)
correct = 0

for i in range(test_size):
    predicted = np.argmax(na.query(sigmoid(np.dot(ae.weights[0], X_test[i])))[-1])
    conf_test[predicted][y_test[i]]+=1
    if predicted == y_test[i]:
        correct+=1

hr_test = correct/test_size

print("Train Data Hit Rate:", hr_train)
print("Train Data Confusion Matrix:")
print(conf_train)
print("Test Data Hit rate:", hr_test)
print("Test Data Confusion Matrix:")
print(conf_test)
plt.plot(eps, ers)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Train Data Error by Epoch')
plt.show()


# In[27]:

#-----------------------------MINST Recognition with regularized autoencoder features-----------------------------------
nb = neuralNetwork(hiddenNodes[0], [], 10, .8)
eps = [] #Epochs for error graph
ers = [] #Errors for error graph

correct = 0
conf_train = np.zeros((10, 10), dtype=int)

#Train the network
epochs = 81
for e in range(epochs):
    correct = 0
    conf_train = np.zeros((10, 10), dtype=int)
    for i in range(train_size):
        #All the output nodes are 0.01, except for the correct value, which is 0.99
        targets = np.zeros(10)
        targets[y_train[i]]=1
        predicted = np.argmax(nb.train(sigmoid(np.dot(rae.weights[0], X_train[i])), targets)[-1])
        nb.update()
        conf_train[predicted][y_train[i]]+=1
        correct += (predicted == y_train[i])
    if(e%10 == 0):
        print(e, correct/train_size)
        eps.append(e)
        ers.append(1-correct/train_size)
    
hr_train = correct/train_size


# In[28]:

#Test data
conf_test = np.zeros((10, 10), dtype=int)
correct = 0

for i in range(test_size):
    predicted = np.argmax(nb.query(sigmoid(np.dot(rae.weights[0], X_test[i])))[-1])
    conf_test[predicted][y_test[i]]+=1
    if predicted == y_test[i]:
        correct+=1

hr_test = correct/test_size

print("Train Data Hit Rate:", hr_train)
print("Train Data Confusion Matrix:")
print(conf_train)
print("Test Data Hit rate:", hr_test)
print("Test Data Confusion Matrix:")
print(conf_test)
plt.plot(eps, ers)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Train Data Error by Epoch')
plt.show()

