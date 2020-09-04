import numpy as np
import time  # You can use this to time how long different parts take to run--find inefficiencies
from test import X, y  # Data set and labels used for testing purposes.
#np.random.seed(0)
np.set_printoptions(threshold=100000)

def get_data_from_acath_csv():
    """
    Data found at: http://biostat.mc.vanderbilt.edu/wiki/Main/DataSets

    For description of data, see: http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/acath.html

    For label explanations, see: http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/Cacath.html
    I've edited the column names to better reflect what the values mean


    Outputs:
    [(training_data, training_sigdz_lables, training_tvdlm_labels),
    (validation_data, validation_sigdz_labels, validation_tvdlm_labels),
    (testing_data, testing_sigdz_labels, testing_tvdlm_labels)]

    Data is shuffled before being allocated to training/validation/testing.
    """
    lines = []
    for line in open('acath.csv'):
        line = (line.replace('\n', ''))
        line = line.split(',')
        lines.append(line)

    lines = lines[1:]
    np.random.shuffle(lines)  # shuffles the same way each time its run
    samples = []
    significant_coronary_disease_lables = []
    three_vessel_or_left_main_disease_labels = []
    for line in lines:
        if line[3] == '':
            line[3] = '0'  # If there's no cholesterol data, set it to 0. There are 1246 times we do this.
        if not line[5] == '':  # Excluding data points with no data for three vessel or left main disease
            # There are 3 empty values for this, so we are ignoring those 3 samples. See:
            # http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/Cacath.html
            sample = [float(val) for val in line]  # Turning string values into actual floats
            samples.append(sample[0:4])  # Including sex, age, symptom duration, and cholesterol as features
            significant_coronary_disease_lables.append(sample[4])
            three_vessel_or_left_main_disease_labels.append(sample[5])

    # Since the data has already been shuffled, we'll just take out the pieces we want for the data sets

    training_data = np.array(samples[0:int(len(samples)*0.6)])
    training_sigdz_lables = np.array(significant_coronary_disease_lables[0:int(len(samples)*0.6)])
    training_tvdlm_labels = np.array(three_vessel_or_left_main_disease_labels[0:int(len(samples)*0.6)])

    validation_data = np.array(samples[round(len(samples)*0.6):round(len(samples)*0.8)])
    validation_sigdz_labels = np.array(significant_coronary_disease_lables
                                       [round(len(samples)*0.6):round(len(samples)*0.8)])
    validation_tvdlm_labels = np.array(three_vessel_or_left_main_disease_labels
                                       [round(len(samples)*0.6):round(len(samples)*0.8)])

    testing_data = np.array(samples[round(len(samples)*0.8):])
    testing_sigdz_labels = np.array(significant_coronary_disease_lables[round(len(samples)*0.8):])
    testing_tvdlm_labels = np.array(three_vessel_or_left_main_disease_labels[round(len(samples)*0.8):])

    train = (training_data, training_sigdz_lables, training_tvdlm_labels)
    validation = (validation_data, validation_sigdz_labels, validation_tvdlm_labels)
    test = (testing_data, testing_sigdz_labels, testing_tvdlm_labels)

    return [train, validation, test]

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

        d_relu__d_inputs[self.inputs <= 0] = 0
        # The above line is a fast implementation of this, below is a slower but more intuitive implementation

        '''
        for row_index in range(len(d_relu__d_inputs)):
            row = d_relu__d_inputs[row_index]
            for val_index in range(len(row)):
                if self.output[row_index][val_index] <= 0:
                    d_relu__d_inputs[row_index][val_index] = 0
        '''

        # element wise product
        self.d_error__d_inputs = np.multiply(d_error__d_relu, d_relu__d_inputs)


class ActivationSigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1/(1+np.e**(-self.inputs))

    def backward(self, d_error__d_sigmoid):
        d_sigmoid__d_inputs = np.multiply(self.output, (1-self.output))
        # element wise product
        self.d_error__d_inputs = np.multiply(d_error__d_sigmoid, d_sigmoid__d_inputs)


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
        '''
        for sample in self.output:
            gradient_matrix = np.tile(sample, (len(sample), 1))  # creates a matrix of len(sample)xlen(sample)
                                                               # dim. where each row is = to sample
            for row_val_index in range(num_of_nodes):
                gradient_matrix[row_val_index] *= -sample[row_val_index]
                gradient_matrix[row_val_index][row_val_index] = sample[row_val_index]*(1-sample[row_val_index])

            gradient_matrices_for_d_softmax__d_inputs.append(gradient_matrix)

        '''  # An intuitive implementation, though less efficient

        for sample in self.output:
            gradient_matrix = np.zeros((num_of_nodes, num_of_nodes))
            for row_index in range(len(gradient_matrix)):
                for val_index in range(len(gradient_matrix[0])):
                    if row_index == val_index:
                        gradient_matrix[row_index][val_index] = sample[val_index] * (1 - sample[row_index])
                    else:
                        gradient_matrix[row_index][val_index] = -sample[row_index] * sample[val_index]

            gradient_matrices_for_d_softmax__d_inputs.append(gradient_matrix)

        d_softmax__d_inputs = np.array(gradient_matrices_for_d_softmax__d_inputs)

        gradient_matrices_for_d_error__d_inputs = []
        for row, sub_matrix in zip(d_error__d_softmax, d_softmax__d_inputs):
            gradient_matrices_for_d_error__d_inputs.append(
                np.dot(sub_matrix, row))

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
        self.network_output = np.clip(network_output, 1e-7, 1 - 1e-7)
        self.correct_output = correct_output
        # correct_output are the correct
        # classifications. If correct_output is [0, 1, 1], then that means the
        # first sample was category 0, the second sample was cat. 1, etc.

        neg_logs = []
        for output, actual in zip(self.network_output, self.correct_output):
            neg_logs.append(-np.log(output[actual]))

        self.error = np.mean(neg_logs)

    def backward(self):
        self.d_error__d_inputs = np.zeros((self.network_output.shape[0],
                                    self.network_output.shape[1]))
        for sample_index in range(len(self.network_output)):
            correct_output_class_for_this_sample = self.correct_output[sample_index]
            self.d_error__d_inputs[sample_index][correct_output_class_for_this_sample] = \
                -1/(self.network_output[sample_index][correct_output_class_for_this_sample] *
                    self.network_output.shape[0])


class OptimizerVanillaSGD:
    def __init__(self, learning_rate=1.0, decay=0.1):
        self.starting_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def update_parameters(self, list_of_layers):
        self.current_learning_rate = self.starting_learning_rate / \
                                     (1+(self.decay*self.iterations))

        for layer_object in list_of_layers:
            update_weights = -(self.current_learning_rate * layer_object.d_error__d_weights)

            update_biases = -(self.current_learning_rate * layer_object.d_error__d_biases)

            layer_object.weights += update_weights
            layer_object.biases += update_biases

        self.iterations += 1


class OptimizerSGDWithMomentum:
    def __init__(self, learning_rate=1.0, decay=0.1, momentum=0.0):
        self.starting_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def update_parameters(self, list_of_layers):
        self.current_learning_rate = self.starting_learning_rate / \
                                     (1+(self.decay*self.iterations))

        for layer_object in list_of_layers:
            if not hasattr(layer_object, 'previous_update_weights'):

                layer_object.previous_update_weights = np.zeros_like(layer_object.weights)
                layer_object.previous_update_biases = np.zeros_like(layer_object.biases)

            update_weights = (self.momentum * layer_object.previous_update_weights) +\
                               -(self.current_learning_rate * layer_object.d_error__d_weights)

            update_biases = (self.momentum * layer_object.previous_update_biases) +\
                            -(self.current_learning_rate * layer_object.d_error__d_biases)

            layer_object.previous_update_weights = update_weights
            layer_object.previous_update_biases = update_biases

            layer_object.weights += update_weights
            layer_object.biases += update_biases

        self.iterations += 1


class OptimizerAdaGrad:
    def __init__(self, learning_rate=1.0, decay=0.1, epsilon=1e-7):
        self.starting_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def update_parameters(self, list_of_layers):
        self.current_learning_rate = self.starting_learning_rate / \
                                     (1+(self.decay*self.iterations))

        for layer_object in list_of_layers:
            if not (hasattr(layer_object, 'weight_cache') or hasattr(layer_object, 'bias_cache')):

                layer_object.weight_cache = np.zeros_like(layer_object.weights)
                layer_object.bias_cache = np.zeros_like(layer_object.biases)

            layer_object.weight_cache += layer_object.d_error__d_weights**2
            layer_object.bias_cache += layer_object.d_error__d_biases**2

            weight_learning_rates = self.current_learning_rate /\
                                   (np.sqrt(layer_object.weight_cache)+self.epsilon)
            bias_learning_rates = self.current_learning_rate /\
                                   (np.sqrt(layer_object.bias_cache)+self.epsilon)

            layer_object.weights += (-weight_learning_rates * layer_object.d_error__d_weights)

            layer_object.biases += (-bias_learning_rates * layer_object.d_error__d_biases)

        self.iterations += 1


class OptimizerRMSProp:
    def __init__(self, learning_rate=.001, decay=0.1, epsilon=1e-7, rho=0.9):
        # NOT SURE WHY WE SET SUCH A LOW STARTING LEARNING RATE
        self.starting_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def update_parameters(self, list_of_layers):
        self.current_learning_rate = self.starting_learning_rate / \
                                     (1+(self.decay*self.iterations))

        for layer_object in list_of_layers:
            if not (hasattr(layer_object, 'weight_cache') or hasattr(layer_object, 'bias_cache')):

                layer_object.weight_cache = np.zeros_like(layer_object.weights)
                layer_object.bias_cache = np.zeros_like(layer_object.biases)

            # Think of rho as the percent of the cache that we keep around. If we are changing the cache
            # less (i.e. higher rho value) then the cache value changes in a smoother way.
            # Smoother cache => smoother learning rates => smoother changes to weights/biases.
            layer_object.weight_cache = self.rho*layer_object.weight_cache + \
                                         (1-self.rho)*(layer_object.d_error__d_weights**2)

            layer_object.bias_cache = self.rho*layer_object.bias_cache + \
                                       (1-self.rho)*(layer_object.d_error__d_biases**2)

            weight_learning_rates = self.current_learning_rate /\
                                   (np.sqrt(layer_object.weight_cache)+self.epsilon)
            bias_learning_rates = self.current_learning_rate /\
                                   (np.sqrt(layer_object.bias_cache)+self.epsilon)

            layer_object.weights += (-weight_learning_rates * layer_object.d_error__d_weights)

            layer_object.biases += (-bias_learning_rates * layer_object.d_error__d_biases)

        self.iterations += 1

'''
class OptimizerAdam:
    def __init__(self, learning_rate=.001, decay=0.1, epsilon=1e-7, rho=0.9, momentum=0):
        self.starting_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
        self.momentum = momentum

    def update_parameters(self, list_of_layers):
        self.current_learning_rate = self.starting_learning_rate / \
                                     (1+(self.decay*self.iterations))

        for layer_object in list_of_layers:
            if not (hasattr(layer_object, 'weight_cache') or hasattr(layer_object, 'bias_cache')):

                layer_object.weight_cache = np.zeros_like(layer_object.weights)
                layer_object.bias_cache = np.zeros_like(layer_object.biases)

                layer_object.wei

            # Think of rho as the percent of the cache that we keep around. If we are changing the cache
            # less (i.e. higher rho value) then the cache value changes in a smoother way.
            # Smoother cache => smoother learning rates => smoother changes to weights/biases.
            layer_object.weight_cache = self.rho*layer_object.weight_cache + \
                                         (1-self.rho)*(layer_object.d_error__d_weights**2)

            layer_object.bias_cache = self.rho*layer_object.bias_cache + \
                                       (1-self.rho)*(layer_object.d_error__d_biases**2)

            weight_learning_rates = self.current_learning_rate /\
                                   (np.sqrt(layer_object.weight_cache)+self.epsilon)
            bias_learning_rates = self.current_learning_rate /\
                                   (np.sqrt(layer_object.bias_cache)+self.epsilon)

            layer_object.weights += (-weight_learning_rates * layer_object.d_error__d_weights)

            layer_object.biases += (-bias_learning_rates * layer_object.d_error__d_biases)

        self.iterations += 1
'''


class ClassificationNeuralNetwork:
    def __init__(self, number_of_input_features, number_of_dense_layers, lengths_for_each_dense_layer,
                 activation_layer_types, cost_function_type, number_of_output_nodes):
        assert number_of_dense_layers == len(activation_layer_types)
        dense_layer_dimensions = [number_of_input_features] + lengths_for_each_dense_layer + \
                                 [number_of_output_nodes]
        self.dense_layer_objects = []
        for index in range(len(dense_layer_dimensions)):
            try:
                self.dense_layer_objects.append(DenseLayer(dense_layer_dimensions[index],
                                                      dense_layer_dimensions[index+1]))
            except IndexError:
                pass

        self.activation_layer_objects = []
        for val in activation_layer_types:
            if val == 0:
                self.activation_layer_objects.append(ActivationReLU())
            elif val == 1:
                self.activation_layer_objects.append(ActivationSigmoid())
            elif val == 2:
                self.activation_layer_objects.append(ActivationSoftmax())
        assert len(self.activation_layer_objects) == len(self.dense_layer_objects)
        self.cost_function_object = None
        if cost_function_type == 0:
            self.cost_function_object = LossCategoricalCrossEntropy()

    def forward_pass(self, network_inputs, correct_outputs):
        self.dense_layer_objects[0].forward(network_inputs)
        self.activation_layer_objects[0].forward(self.dense_layer_objects[0].output)
        current_output = self.activation_layer_objects[0].output
        for dense_layer_object, activation_layer_object in \
            zip(self.dense_layer_objects[1:], self.activation_layer_objects[1:]):
            dense_layer_object.forward(current_output)
            activation_layer_object.forward(dense_layer_object.output)
            current_output = activation_layer_object.output

        self.cost_function_object.forward(current_output, correct_outputs)
        self.error = self.cost_function_object.error
        predictions = np.argmax(self.activation_layer_objects[-1].output, axis=1)
        self.accuracy = np.mean(predictions==correct_outputs)

        self.network_output = current_output

    def backward_pass(self):
        self.cost_function_object.backward()
        current_d_error__d_inputs = self.cost_function_object.d_error__d_inputs

        for index in reversed(range(0, len(self.activation_layer_objects))):
            activation_layer_object = self.activation_layer_objects[index]
            dense_layer_object = self.dense_layer_objects[index]

            activation_layer_object.backward(current_d_error__d_inputs)
            dense_layer_object.backward(activation_layer_object.d_error__d_inputs)

            current_d_error__d_inputs = dense_layer_object.d_error__d_inputs


class Model:
    def __init__(self, number_of_layers, neuron_range, training_data, training_labels, validation_data,
                 validation_labels, testing_data, testing_labels, number_of_outputs_nodes):
        self.number_of_layers = number_of_layers
        self.neuron_range = neuron_range  # The range of the number of neurons per layer
        self.training_data = training_data
        self.training_labels = training_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.testing_data = testing_data
        self.testing_labels = testing_labels
        self.number_of_output_nodes = number_of_outputs_nodes
        self.neural_networks = []
        self.optimizer_objects = []

    def train(self):
        for val in range(3):  # We will have 3 network architectures we try
            layer_lengths = list(np.random.randint(self.neuron_range[0], self.neuron_range[1],
                                                   self.number_of_layers-1))  # The last dense layer is output
            activation_layer_types = list(np.random.randint(0, 2, self.number_of_layers))
            cost_function_type = 0  # For CE
            number_of_output_nodes = self.number_of_output_nodes
            number_of_input_features = self.training_data.shape[1]

            for i in range(2):  # Creating 2 instances so we have multiple starting points parameter wise.
                self.neural_networks.append(ClassificationNeuralNetwork(number_of_input_features,
                                                                   self.number_of_layers,
                                                                   layer_lengths, activation_layer_types,
                                                                   cost_function_type,number_of_output_nodes))
                if i % 2:
                    self.optimizer_objects.append(OptimizerSGDWithMomentum(momentum=0.2))
                elif i%2 == 1:
                    self.optimizer_objects.append(OptimizerAdaGrad())

        # Time to train!
        counter = 1
        for neural_network, optimizer in zip(self.neural_networks, self.optimizer_objects):
            print('Network: ', counter)
            for epoch in range(10001):
                neural_network.forward_pass(self.training_data, self.training_labels)
                neural_network.backward_pass()
                optimizer.update_parameters(neural_network.dense_layer_objects)

                if epoch % 100 == 0:
                    print(f'epoch: {epoch}, ' +
                          f'acc: {neural_network.accuracy:.3f}, ' +
                          f'loss: {neural_network.error:.3f}, ' +
                          f'lr: {optimizer.current_learning_rate}')
                    for index in range(len(neural_network.dense_layer_objects)):
                        layer = neural_network.dense_layer_objects[index]
                        print(f'Layer {index} max weight: {np.amax(layer.weights)}, ' +
                        f'Layer {index} min weight: {np.amin(layer.weights):.3f}, ' +
                        f'Layer {index} max bias: {np.amax(layer.biases):.3f}, ' +
                        f'Layer {index} min bias: {np.amin(layer.biases)}')
            counter += 1

    def test(self):
        best_network = sorted(self.neural_networks, key=lambda network: network.accuracy)[0]
        for network in self.neural_networks:
            assert best_network.accuracy >= network
        best_network.forward(self.testing_data, self.testing_labels)
        print()
        print()
        print(f'The best network had an accuracy of {best_network.accuracy} and loss of {best_network.error}')
        print()
        print('The following is a loop through of all of the layers in this '
              'network and the weights and biases')
        for layer in best_network.dense_layer_objects:
            print('Weights:')
            print(layer.weights)
            print()
            print('Biases:')
            print(layer.biases)
            print()


data = get_data_from_acath_csv()
training_data, training_sigdz_labels = data[0][0], data[0][1].astype(int)

validation_data, validation_sigdz_labels = data[1][0], data[1][1].astype(int)

testing_data, testing_sigdz_labels = data[2][0], data[2][1].astype(int)

model = Model(number_of_layers=2, neuron_range=(10, 15), training_data=training_data,
              training_labels=training_sigdz_labels, validation_data=validation_data,
              validation_labels=validation_sigdz_labels,
              testing_data=testing_data, testing_labels=testing_sigdz_labels, number_of_outputs_nodes=2)

model.train()
model.test()


'''
# Create Dense layer with 2 input features and 64 output values
dense1 = DenseLayer(2, 64)


# Create ReLU activation (to be used with Dense layer):
activation1 = ActivationReLU()

# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = DenseLayer(64, 3)


# Create Softmax classifier's combined loss and activation
activation2 = ActivationSoftmax()
CE = LossCategoricalCrossEntropy()

# Create optimizer
my_SGD = OptimizerSGD(learning_rate=1, decay=1e-3, momentum=0.9)
my_AdaGrad = OptimizerAdaGrad(decay=1e-4)
their_AdaGrad = Optimizer_Adagrad(decay=1e-4)
my_RMSProp = OptimizerRMSProp(learning_rate=.02, decay=1e-5, rho=0.999)

# Train in loop
for epoch in range(10001):

    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)

    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)

    activation2.forward(dense2.output)
    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    CE.forward(activation2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions==y)
    loss = CE.error
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {my_RMSProp.current_learning_rate}')

    # Backward pass
    CE.backward()
    activation2.backward(CE.d_error__d_inputs)
    dense2.backward(activation2.d_error__d_inputs)
    activation1.backward(dense2.d_error__d_inputs)
    dense1.backward(activation1.d_error__d_inputs)

    # Update parameters
    my_RMSProp.update_parameters([dense1, dense2])

    #SOFTMAX_BACKPROP_TIME += (activation2_backprop_time_end - activation2_backprop_time_start)
    #OPTIMIZER_UPDATE_TIME += (optimizer_update_parameters_time_end - optimizer_update_parameters_time_start)
    #CE_BACKPROP_TIME += (loss_function_backprop_time_end-loss_function_backprop_time_start)
    #DENSE1_BACKPROP_TIME += (dense1_backprop_time_end - dense1_backprop_time_start)
    #DENSE2_BACKPROP_TIME += (dense2_backprop_time_end - dense2_backprop_time_start)
    #RELU_BACKPROP_TIME += (activation1_backprop_time_end - activation1_backprop_time_start)


END_TIME = time.time()

print()
print('TOTAL TIME: ', END_TIME-START_TIME)

print()
print('SOFTMAX BACKPROP TIME: ', SOFTMAX_BACKPROP_TIME)
print('OPTIMIZER UPDATE TIME: ', OPTIMIZER_UPDATE_TIME)
print('CE BACKPROP TIME: ', CE_BACKPROP_TIME)
print('RELU BACKPROP TIME: ', RELU_BACKPROP_TIME)
print('DENSE1 BACKPROP TIME: ', DENSE1_BACKPROP_TIME)
print('DENSE2 BACKPROP TIME: ', DENSE2_BACKPROP_TIME)
'''
