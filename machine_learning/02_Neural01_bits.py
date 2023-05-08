# https://gist.github.com/a-i-dan/8d0a40b8690b40d9a46c4cb1d326fce5
import numpy as np # helps with the math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # to plot error during training


# input data
inputs = np.array([[0, 1, 0],
                   [0, 1, 1],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [1, 0, 1]])
# output data
outputs = np.array([[0],  [0], [0],  [1], [1], [1]])
#hidden = np.array ([[1],  [0], [1],  [0], [0], [1]])
#error  = np.array ([[-1], [0], [-1], [1], [1], [0]])

# create NeuralNetwork class
class NeuralNetwork:

    # intialize variables in class
    def __init__(self, inputs, outputs):
        self.inputs  = inputs
        self.outputs = outputs
        # initialize weights as .50 for simplicity
        self.weights = np.array([[.50], [.50], [.50]])
        self.error_history = []
        self.epoch_list = []

    #activation function ==> S(z) = 1/1+e^(-z)
    def sigmoid(self, z, deriv=False):
        if deriv == True:
            return z * (1 - z)
        return 1 / (1 + np.exp(-z))

    # data will flow through the neural network.
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    # going backwards through the network to update weights
    def backpropagation(self):
        self.error  = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    # train the neural net for 25,000 iterations
    def train(self, epochs=25000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            print(f"epochs = {epoch}",end='\t')
            print(f'Antes={self.weights}')
            self.backpropagation()
            print(f'Después={self.weights}')
            print('###############')
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)
        return self.weights

    # function to predict output on new and unseen input data
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction


# create neural network
NN = NeuralNetwork(inputs, outputs)
# train neural network
pesos  = NN.train(5000)
print("Pesos finales=",pesos)
# create two new examples to predict
example = np.array([[1, 1, 0]])
example_2 = np.array([[0, 0, 1]])

# print the predictions for both examples                                   
print(example,NN.predict(example), ' - Correct: ', example[0][0])
print(example_2,NN.predict(example_2), ' - Correct: ', example_2[0][0])

# plot the error over the entire training duration
#https://stackoverflow.com/questions/56656777/userwarning-matplotlib-is-currently-using-agg-which-is-a-non-gui-backend-so
plt.figure(figsize=(15,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()