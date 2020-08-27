import numpy as np
import nnfs
from nnfs.datasets import spiral_data  # Used for data generation for testing my model
nnfs.init()


def create_spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.05
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


def create_simple_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))  # index in class
        X[ix] = np.c_[np.random.randn(points)*.1 + (class_number)/3, np.random.randn(points)*.1 + 0.5]
        y[ix] = class_number
    return X, y


class DenseLayer:
    def __init__(self, num_of_inputs, num_of_neurons):
        self.weights = 0.01 * np.random.randn(num_of_inputs, num_of_neurons)
        # The parameters of randn are just the dimensions of the matrix it makes
        self.biases = np.zeros((1, num_of_neurons))
        # creates a matrix of height 1 and length num_of_neurons filled with 0's
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases

    def backward(self, d_error__d_denselayer):
        d_denselayer__d_weights = self.inputs.T
        d_denselayer__d_biases = np.ones((1, len(self.inputs)))
        # ^^ produces a matrix of 1x(number of samples) dimensions

        self.d_error__d_weights = np.dot(d_denselayer__d_weights,
                                         d_error__d_denselayer)
        self.d_error__d_biases = np.dot(d_denselayer__d_biases,
                                        d_error__d_denselayer)
        d_denselayer__d_input = self.weights.T
        self.d_error__d_inputs = np.dot(d_error__d_denselayer, d_denselayer__d_input)


class ActivationReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, d_error__d_relu):
        d_relu__d_inputs = np.ones((self.output.shape[0], self.output.shape[1]))
        for row_index in range(len(d_relu__d_inputs)):
            row = d_relu__d_inputs[row_index]
            for val_index in range(len(row)):
                if self.output[row_index][val_index] <= 0:
                    d_relu__d_inputs[row_index][val_index] = 0

        # element wise product
        self.d_error__d_inputs = np.multiply(d_error__d_relu, d_relu__d_inputs)


class ActivationSoftmax:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        exponentiated_values = np.exp(inputs-np.max(inputs, axis=1,
                                                    keepdims=True))
        # the np.max part ensures all exponents are <=0, while actually
        # resulting in the same probability outputs!
        self.output = exponentiated_values / np.sum(exponentiated_values,
                                                    axis=1, keepdims=True)
        # we use axis=1 in both lines to make sure it does all these operations
        # based on rows.

    def backward(self, d_error__d_softmax):
        num_of_nodes = self.output.shape[1]

        gradient_matrices_for_d_softmax__d_inputs = []
        for sample in self.output:
            gradient_matrix = np.zeros((num_of_nodes, num_of_nodes))
            for row_index in range(len(gradient_matrix)):
                row = gradient_matrix[row_index]
                for val_index in range(len(row)):
                    if row_index == val_index:
                        row[val_index] = sample[val_index] * (1-sample[val_index])
                    else:
                        row[val_index] = -sample[row_index] * sample[val_index]
            gradient_matrices_for_d_softmax__d_inputs.append(gradient_matrix)

        d_softmax__d_inputs = np.array(gradient_matrices_for_d_softmax__d_inputs)

        gradient_matrices_for_d_error__d_inputs = []
        for row, sub_matrix in zip(d_error__d_softmax, d_softmax__d_inputs):
            gradient_matrices_for_d_error__d_inputs.append(
                np.dot(row, sub_matrix))

        self.d_error__d_inputs = np.array(gradient_matrices_for_d_error__d_inputs)


class LossCategoricalCrossEntropy:
    def __init__(self):
        # obligatory iS tHiS lOsS reference
        """
        │     │ |

        │ │   │ __
        """
        pass

    def forward(self, network_output, correct_output):
        self.network_output = network_output
        self.correct_output = correct_output
        # correct_output are the correct
        # classifications. If correct_output is [0, 1, 1], then that means the
        # first sample was category 0, the second sample was cat. 1, etc.

        neg_logs = []
        for output, actual in zip(network_output, correct_output):
            neg_logs.append(-np.log(output[actual]))

        self.error = np.mean(neg_logs)

    def backward(self):
        self.d_error__d_inputs = np.zeros((self.network_output.shape[0],
                                    self.network_output.shape[1]))
        for sample_index in range(len(self.network_output)):
            correct_output_class_for_this_sample = self.correct_output[sample_index]

            for neuron_index in range(self.network_output.shape[1]):
                if neuron_index == correct_output_class_for_this_sample:
                    self.d_error__d_inputs[sample_index][neuron_index] = \
                        -1/(self.network_output[sample_index][neuron_index]*self.network_output.shape[0])


class OptimizerSGD:
    def __init__(self, learning_rate=1.0, decay=0.1):
        self.starting_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def update_parameters(self, list_of_layers):
        self.current_learning_rate = self.starting_learning_rate / \
                                     (1+(self.decay*self.iterations))

        for layer_object in list_of_layers:
            if not hasattr(layer_object, 'previous_d_error__d_weights') and \
                    hasattr(layer_object, 'previous_d_error__d_biases'):
                layer_object.previous_d_error__d_weights = np.zeros_like(layer_object.d_error__d_weights)
                layer_object.previous_d_error__d_biases = np.zeros_like(layer_object.d_error__d_biases)


            layer_object.weights += -self.current_learning_rate * layer_object.d_error__d_weights
            layer_object.biases += -self.current_learning_rate * layer_object.d_error__d_biases

        self.iterations += 1


#########################################################################################################
# TESTING EXAMPLES#
#########################################################################################################

# Simple example
'''
inputs = np.array([[1, 2, 0.3],
                   [0.4, .2, 1.1],
                   [1, 1, 0.8],
                   [3.2, .49, .899]])

correct_outputs = np.array([0, 1, 1, 2])

dense1 = DenseLayer(3, 4)
relu1 = ActivationReLU()
softmax = ActivationSoftmax()
CE = LossCategoricalCrossEntropy()

dense1.forward(inputs)
relu1.forward(dense1.output)
softmax.forward(relu1.output)
CE.forward(softmax.output, correct_outputs)

dense1.backward(relu1.backward(softmax.backward(CE.backward())))
'''

X, y = create_spiral_data(points=100, classes=3)

# Initializing layers
dense1 = DenseLayer(2, 64)
activation1 = ActivationReLU()
dense2 = DenseLayer(64, 3)
activation2 = ActivationSoftmax()
loss_function = LossCategoricalCrossEntropy()
optimizer = OptimizerSGD(1, 0.001)

for epoch in range(10001):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    loss_function.forward(activation2.output, y)
    loss = loss_function.error

    if epoch % 100 == 0:
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == y)
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}' + '\t' +
              f'current_learning_rate: {optimizer.current_learning_rate:.3f}')


    loss_function.backward()
    activation2.backward(loss_function.d_error__d_inputs)
    dense2.backward(activation2.d_error__d_inputs)
    activation1.backward(dense2.d_error__d_inputs)
    dense1.backward(activation1.d_error__d_inputs)

    optimizer.update_parameters([dense1, dense2])

