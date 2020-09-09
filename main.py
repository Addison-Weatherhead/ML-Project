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
    [(training_data, training_sigdz_labels, training_tvdlm_labels),
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
    significant_coronary_disease_labels = []
    three_vessel_or_left_main_disease_labels = []
    for line in lines:
        if line[3] == '':
            line[3] = '0'  # If there's no cholesterol data, set it to 0. There are 1246 times we do this.
        if not line[5] == '':  # Excluding data points with no data for three vessel or left main disease
            # There are 3 empty values for this, so we are ignoring those 3 samples. See:
            # http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/Cacath.html
            sample = [float(val) for val in line]  # Turning string values into actual floats
            samples.append(sample[0:4])  # Including sex, age, symptom duration, and cholesterol as features
            significant_coronary_disease_labels.append(int(sample[4]))
            three_vessel_or_left_main_disease_labels.append(int(sample[5]))

    # Since the data has already been shuffled, we'll just take out the pieces we want for the data sets

    training_data = np.array(samples[0:int(len(samples)*0.6)])
    training_sigdz_labels = np.array(significant_coronary_disease_labels[0:int(len(samples)*0.6)])
    training_tvdlm_labels = np.array(three_vessel_or_left_main_disease_labels[0:int(len(samples)*0.6)])

    validation_data = np.array(samples[round(len(samples)*0.6):round(len(samples)*0.8)])
    validation_sigdz_labels = np.array(significant_coronary_disease_labels
                                       [round(len(samples)*0.6):round(len(samples)*0.8)])
    validation_tvdlm_labels = np.array(three_vessel_or_left_main_disease_labels
                                       [round(len(samples)*0.6):round(len(samples)*0.8)])

    testing_data = np.array(samples[round(len(samples)*0.8):])
    testing_sigdz_labels = np.array(significant_coronary_disease_labels[round(len(samples)*0.8):])
    testing_tvdlm_labels = np.array(three_vessel_or_left_main_disease_labels[round(len(samples)*0.8):])

    # normalizing the data
    training_data = training_data/np.max(training_data, axis=0)
    validation_data = validation_data/np.max(validation_data, axis=0)
    testing_data = testing_data/np.max(testing_data, axis=0)

    train = (training_data, training_sigdz_labels, training_tvdlm_labels)
    validation = (validation_data, validation_sigdz_labels, validation_tvdlm_labels)
    test = (testing_data, testing_sigdz_labels, testing_tvdlm_labels)

    return [train, validation, test]


def get_data_from_falldetection_csv():
    """
    Data set found at https://www.kaggle.com/pitasr/falldata/version/1#
    """
    lines = []
    for line in open('falldetection.csv'):
        line = (line.replace('\n', ''))
        line = line.split(',')
        lines.append(line)
    column_names = lines.pop(0)
    np.random.shuffle(lines)
    labels = []
    samples = []
    for line in lines:
        labels.append(int(line[0]))
        samples.append([float(val) for val in line[1:]])

    all_data = [(label, sample) for label, sample in zip(labels, samples)]
    organized_data = {}
    for data_point in all_data:
        if data_point[0] in organized_data:
            organized_data[data_point[0]].append(data_point[1])
        else:
            organized_data[data_point[0]] = [data_point[1]]

    training_dict = {}
    rest_dict = {}
    for label in organized_data:
        if label != 3:
            training_dict[label] = organized_data[label][0:int(len(organized_data[3]) * 0.8/5)]
            rest_dict[label] = organized_data[label][int(len(organized_data[3]) * 0.8/5):]
    training_dict[3] = organized_data[3][0:int(len(organized_data[3]) * 0.8)]
    rest_dict[3] = organized_data[3][int(len(organized_data[3]) * 0.8):]



    training_data_and_labels = []
    rest_data_and_labels = []
    for key in training_dict:
        for sample in training_dict[key]:
            if key == 3:
                training_data_and_labels.append((1, sample))
            else:
                training_data_and_labels.append((0, sample))

        for sample2 in rest_dict[key]:
            if key == 3:
                rest_data_and_labels.append((1, sample2))
            else:
                rest_data_and_labels.append((0, sample2))

    np.random.shuffle(training_data_and_labels)
    np.random.shuffle(rest_data_and_labels)
    rest_data = [tup[1] for tup in rest_data_and_labels]
    rest_labels = [tup[0] for tup in rest_data_and_labels]

    training_data = np.array([tup[1] for tup in training_data_and_labels])
    training_labels = np.array([tup[0] for tup in training_data_and_labels])

    validation_data = np.array(rest_data[0:int(len(rest_data) * 0.3)])
    validation_labels = np.array(rest_labels[0:int(len(rest_data) * 0.3)])

    testing_data = np.array(rest_data[int(len(rest_data) * 0.3):])
    testing_labels = np.array(rest_labels[int(len(rest_data) * 0.3):])

    # normalizing the data
    #training_data = training_data / np.max(training_data, axis=0)
    #validation_data = validation_data / np.max(validation_data, axis=0)
    #testing_data = testing_data / np.max(testing_data, axis=0)

    train = (training_data, training_labels)
    validation = (validation_data, validation_labels)
    test = (testing_data, testing_labels)

    return [train, validation, test]


def get_data_from_heart_failure_clinical_records_dataset_csv():
    lines = []
    for line in open('heart_failure_clinical_records_dataset.csv'):
        lines.append(line.replace('\n', '').split(','))
    lines.pop(0)
    np.random.shuffle(lines)
    samples = []
    labels = []
    for line in lines:
        samples.append([float(val) for val in line[:-1]])
        labels.append(int(line[-1]))

    edited_samples = [sample[0:-1] for sample in samples]  # removes time
    assert len(labels) == len(samples)
    training_data = np.array(samples[0:int(len(samples)*0.7)])
    training_labels = np.array(labels[0:int(len(samples)*0.7)])

    validation_data = np.array(samples[int(len(samples)*0.7):int(len(samples)*0.8)])
    validation_labels = np.array(labels[int(len(samples)*0.7):int(len(samples)*0.8)])

    testing_data = np.array(samples[int(len(samples)*0.8):])
    testing_labels = np.array(labels[int(len(samples)*0.8):])

    training_data = training_data / np.max(training_data, axis=0)
    validation_data = validation_data / np.max(validation_data, axis=0)
    testing_data = testing_data / np.max(testing_data, axis=0)

    train = (training_data, training_labels)
    validate = (validation_data, validation_labels)
    test = (testing_data, testing_labels)

    return train, validate, test



class DenseLayer:
    def __init__(self, num_of_inputs, num_of_neurons, l2_weight_lambda=0.0,
                 l2_bias_lambda=0.0):
        self.weights = 0.01 * np.random.randn(num_of_inputs, num_of_neurons)
        # The parameters of randn are just the dimensions of the matrix it makes
        self.biases = np.zeros((1, num_of_neurons))
        # creates a matrix of height 1 and length num_of_neurons filled with 0's
        self.output = None

        # Factors used to penalize weights/biases of large magnitude. See the regularization_loss method in
        # the CE loss class
        self.l2_weight_lambda = l2_weight_lambda
        self.l2_bias_lambda = l2_bias_lambda

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases

    def backward(self, d_error__d_denselayer):
        d_denselayer__d_weights = self.inputs.T
        d_denselayer__d_biases = np.ones((1, len(self.inputs)))
        # ^^ produces a matrix of 1x(number of samples) dimensions

        self.d_error__d_weights = np.dot(d_denselayer__d_weights,
                                         d_error__d_denselayer)
        if self.l2_weight_lambda > 0:
            self.d_error__d_weights += 2*self.l2_weight_lambda*self.weights
        self.d_error__d_biases = np.dot(d_denselayer__d_biases,
                                        d_error__d_denselayer)
        if self.l2_bias_lambda > 0:
            self.d_error__d_biases += 2*self.l2_bias_lambda*self.biases

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

    def l2_regularization(self, layer):
        regularization_loss = 0
        if layer.l2_weight_lambda > 0:
            regularization_loss += layer.l2_weight_lambda * np.sum(layer.weights**2)
        if layer.l2_bias_lambda > 0:
            regularization_loss += layer.l2_bias_lambda * np.sum(layer.biases**2)

        return regularization_loss


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
                 activation_layer_types, cost_function_type, number_of_output_nodes, l2_weight_lambda,
                 l2_bias_lambda):
        if cost_function_type == 0:
            self.cost_function_object = LossCategoricalCrossEntropy()
            activation_layer_types[-1] = 2

        assert number_of_dense_layers == len(activation_layer_types)
        dense_layer_dimensions = [number_of_input_features] + lengths_for_each_dense_layer + \
                                 [number_of_output_nodes]

        self.l2_weight_lambda = l2_weight_lambda
        self.l2_bias_lambda = l2_bias_lambda

        self.dense_layer_objects = []
        for index in range(len(dense_layer_dimensions)):
            try:
                print(dense_layer_dimensions[index], dense_layer_dimensions[index+1])
                self.dense_layer_objects.append(DenseLayer(dense_layer_dimensions[index],
                                                      dense_layer_dimensions[index+1],
                                                           l2_weight_lambda=self.l2_weight_lambda,
                                                           l2_bias_lambda=self.l2_bias_lambda))
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

        print(len(self.dense_layer_objects))
        print(len(self.activation_layer_objects))

        assert len(self.activation_layer_objects) == len(self.dense_layer_objects)
        self.optimizer = None

    def forward_pass(self, network_inputs, correct_outputs):
        self.dense_layer_objects[0].forward(network_inputs)
        self.activation_layer_objects[0].forward(self.dense_layer_objects[0].output)
        current_output = self.activation_layer_objects[0].output
        regularization_error = self.cost_function_object.l2_regularization(self.dense_layer_objects[0])
        for dense_layer_object, activation_layer_object in \
            zip(self.dense_layer_objects[1:], self.activation_layer_objects[1:]):
            dense_layer_object.forward(current_output)
            activation_layer_object.forward(dense_layer_object.output)
            current_output = activation_layer_object.output
            regularization_error += self.cost_function_object.l2_regularization(dense_layer_object)

        self.regularization_error = regularization_error
        self.cost_function_object.forward(current_output, correct_outputs)
        self.error = self.cost_function_object.error + regularization_error

        predictions = np.argmax(self.activation_layer_objects[-1].output, axis=1)
        self.accuracy = np.mean(predictions == correct_outputs)

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
    def __init__(self, number_of_layers, neuron_range, activation_range, training_data, training_labels,
                 validation_data, validation_labels, testing_data, testing_labels, number_of_outputs_nodes,
                 number_of_network_architectures, number_of_network_instances, momentum_range, rho_range,
                 learning_rate_range, l2_weight_lambda, l2_bias_lambda, decay_range):
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
        self.number_of_network_architectures = number_of_network_architectures
        self.number_of_network_instances = number_of_network_instances
        self.momentum_range = momentum_range
        self.rho_range = rho_range
        self.learning_rate_range = learning_rate_range
        self.l2_weight_lambda = l2_weight_lambda
        self.l2_bias_lambda = l2_bias_lambda
        self.decay_range = decay_range
        self.activation_range = activation_range
        self.best_network = None

    def train(self):
        for val in range(self.number_of_network_architectures):  # Trying multiple network architectures
            layer_lengths = list(np.random.randint(self.neuron_range[0], self.neuron_range[1],
                                                   self.number_of_layers))  # The last dense layer is output
            activation_layer_types = list(np.random.randint(self.activation_range[0],
                                                            self.activation_range[1], self.number_of_layers))
            cost_function_type = 0  # For CE
            number_of_output_nodes = self.number_of_output_nodes
            number_of_input_features = self.training_data.shape[1]

            for i in range(self.number_of_network_instances):  # Creating multiple instances so we have
                # multiple starting points parameter wise.

                self.neural_networks.append(ClassificationNeuralNetwork(
                    number_of_input_features=number_of_input_features,
                    number_of_dense_layers=self.number_of_layers,
                    lengths_for_each_dense_layer=layer_lengths, activation_layer_types=activation_layer_types,
                    cost_function_type=cost_function_type, number_of_output_nodes=number_of_output_nodes,
                    l2_bias_lambda=self.l2_bias_lambda, l2_weight_lambda=self.l2_weight_lambda))
                if i % 3 == 0:
                    self.optimizer_objects.append(OptimizerSGDWithMomentum(
                        momentum=np.random.uniform(self.momentum_range[0], self.momentum_range[1]),
                        learning_rate=np.random.uniform(self.learning_rate_range[0],
                                                        self.learning_rate_range[1]),
                        decay=np.random.uniform(self.decay_range[0], self.decay_range[1])))

                elif i % 3 == 1:
                    self.optimizer_objects.append(OptimizerAdaGrad(
                        learning_rate=np.random.uniform(self.learning_rate_range[0],
                                                        self.learning_rate_range[1]),
                        decay=np.random.uniform(self.decay_range[0], self.decay_range[1])))
                elif i % 3 == 2:
                    self.optimizer_objects.append(OptimizerRMSProp(rho=np.random.uniform(self.rho_range[0],
                                                                                         self.rho_range[1]),
                        decay=np.random.uniform(self.decay_range[0], self.decay_range[1])))

            print('number of networks: ', len(self.neural_networks))
            print('number of optimizers: ', len(self.optimizer_objects))

        # Time to train!
        counter = 1
        for neural_network, optimizer in zip(self.neural_networks, self.optimizer_objects):
            neural_network.optimizer = optimizer
            print('Network: ', counter)
            for layer in neural_network.activation_layer_objects:
                print(type(layer))

            last_1000_epoch_accuracies = []
            should_break = False
            for epoch in range(10001):
                if should_break:
                    print('STUCK IN A MIN, STOPPING TRAINING')
                    break
                neural_network.forward_pass(self.training_data, self.training_labels)
                neural_network.backward_pass()
                optimizer.update_parameters(neural_network.dense_layer_objects)

                if epoch % 100 == 0:
                    max_weight = 0
                    min_weight = 0
                    max_bias = 0
                    min_bias = 0
                    for layer in neural_network.dense_layer_objects:
                        if np.amax(layer.weights) > max_weight:
                            max_weight = np.amax(layer.weights)
                        if np.amin(layer.weights) < min_weight:
                            min_weight = np.amin(layer.weights)

                        if np.amax(layer.biases) > max_bias:
                            max_bias = np.amax(layer.biases)
                        if np.amin(layer.biases) < min_bias:
                            min_bias = np.amin(layer.biases)

                    training_accuracy = neural_network.accuracy
                    training_error = neural_network.error

                    neural_network.forward_pass(self.validation_data, self.validation_labels)
                    validation_accuracy = neural_network.accuracy
                    neural_network.validation_accuracy = validation_accuracy
                    print(f'epoch: {epoch}, ' +
                          f'acc: {training_accuracy:.3f}, ' +
                          f'validation acc: {validation_accuracy:.3f},  ' +
                          f'loss: {training_error:.3f}, ' +
                          f'lr: {optimizer.current_learning_rate}  ' +
                          f'weight range: {(min_weight, max_weight)}  ' +
                          f'bias range: {(min_bias, max_bias)}')

                    rounded_accuracy = round(neural_network.accuracy, 3)
                    if len(last_1000_epoch_accuracies) < 10:
                        last_1000_epoch_accuracies.append(rounded_accuracy)
                    if len(last_1000_epoch_accuracies) == 10:
                        if last_1000_epoch_accuracies.count(rounded_accuracy) == len(last_1000_epoch_accuracies):
                            should_break = True
                        else:
                            last_1000_epoch_accuracies.pop(0)
                            last_1000_epoch_accuracies.append(rounded_accuracy)

            counter += 1

    def validate(self, num_of_dense_layers, activation_function_types, num_of_outputs,
                 layer_lengths, optimizer_object, cost_function_type, l2_weight_lambda, l2_bias_lambda):

        neural_network = ClassificationNeuralNetwork(number_of_input_features=self.validation_data.shape[1],
                                                     number_of_dense_layers=num_of_dense_layers,
                                                     lengths_for_each_dense_layer=layer_lengths,
                                                     activation_layer_types=activation_function_types,
                                                     cost_function_type=cost_function_type,
                                                     number_of_output_nodes=num_of_outputs,
                                                     l2_weight_lambda=l2_weight_lambda,
                                                     l2_bias_lambda=l2_bias_lambda)

        for epoch in range(10001):
            neural_network.forward_pass(self.training_data, self.training_labels)
            neural_network.backward_pass()
            optimizer_object.update_parameters(neural_network.dense_layer_objects)

            if epoch % 100 == 0:
                max_weight = 0
                min_weight = 0
                max_bias = 0
                min_bias = 0
                for layer in neural_network.dense_layer_objects:
                    if np.amax(layer.weights) > max_weight:
                        max_weight = np.amax(layer.weights)
                    if np.amin(layer.weights) < min_weight:
                        min_weight = np.amin(layer.weights)

                    if np.amax(layer.biases) > max_bias:
                        max_bias = np.amax(layer.biases)
                    if np.amin(layer.biases) < min_bias:
                        min_bias = np.amin(layer.biases)

                training_accuracy = neural_network.accuracy
                training_error = neural_network.error

                neural_network.forward_pass(self.testing_data, self.testing_labels)
                testing_accuracy = neural_network.accuracy

                neural_network.forward_pass(self.validation_data, self.validation_labels)
                validation_accuracy = neural_network.accuracy

                print(f'epoch: {epoch}, ' +
                      f'acc: {training_accuracy:.3f}, ' +
                      f'validation acc: {validation_accuracy:.3f},  ' +
                      f'testing acc: {testing_accuracy:.3f},  ' +
                      f'loss: {training_error:.3f}, ' +
                      f'lr: {optimizer_object.current_learning_rate}  ' +
                      f'weight range: {(min_weight, max_weight)}  ' +
                      f'bias range: {(min_bias, max_bias)}')

    def test(self):
        self.best_network = sorted(self.neural_networks, key=lambda network: network.validation_accuracy)[-1]
        print([net.accuracy for net in sorted(self.neural_networks, key=lambda network: network.validation_accuracy)])

        self.best_network.forward_pass(self.testing_data, self.testing_labels)
        print()
        print()
        print(f'The best network had a test accuracy of {self.best_network.accuracy} and loss of {self.best_network.error}')
        print()
        print('The following is a loop through of all of the layers in this '
              'network, providing its size, weights, and biases')
        for layer in self.best_network.dense_layer_objects:
            print('Number of inputs: ', layer.inputs.shape[1])  # Assumes .shape doens't return something like
            print('Number of outputs: ', layer.output.shape[1])  # (5,). This will error if #inputs/outputs=1
            print()
            print('Weights:')
            print(layer.weights)
            print()
            print('Biases:')
            print(layer.biases)
            print()




data = get_data_from_acath_csv()

training_data, training_labels = data[0][0], data[0][1]
validation_data, validation_labels = data[1][0], data[1][1]
testing_data, testing_labels = data[2][0], data[2][1]


model = Model(number_of_layers=3, neuron_range=(20, 25), activation_range=(0, 2), training_data=training_data,
              training_labels=training_labels, validation_data=validation_data,
              validation_labels=validation_labels,
              testing_data=testing_data, testing_labels=testing_labels, number_of_outputs_nodes=2,
              number_of_network_architectures=4, number_of_network_instances=4, momentum_range=(0.7, 0.9),
              rho_range=(0.6, 0.9), learning_rate_range=(0.7, 1), l2_weight_lambda=.00005,
              l2_bias_lambda=.00005, decay_range=(0.002, 0.04))


# model.train() # Output of this showed that the best performer was a NN with 3 layers, activation functions
# of Relu, Sigmoid, and Softmax, and with 20-25 neurons per layer.

adagrad = OptimizerAdaGrad(learning_rate=1.0, decay=0.01, epsilon=1e-7)
'''
rms_prop = OptimizerRMSProp(learning_rate=1, decay=0.01, epsilon=1e-7, rho=0.7)  # acc: 0.742, validation acc: 0.753
model.validate(num_of_dense_layers=3, activation_function_types=[0, 1, 2], num_of_outputs=2,
                 layer_lengths=[22, 25], optimizer_object=rms_prop, cost_function_type=0,
               l2_weight_lambda=.0005, l2_bias_lambda=.0005)
'''
'''
sgd_momentum = OptimizerSGDWithMomentum(learning_rate=1.0, decay=0.001, momentum=0.7)  # acc: 0.763, validation acc: 0.703

model.validate(num_of_dense_layers=3, activation_function_types=[0, 1, 2], num_of_outputs=2,
                 layer_lengths=[28, 30], optimizer_object=sgd_momentum, cost_function_type=0,
               l2_weight_lambda=.0005, l2_bias_lambda=.0005)
'''

'''
rms_prop = OptimizerRMSProp(learning_rate=1, decay=0.01, epsilon=1e-7, rho=0.7)  # acc: 0.740, validation acc: 0.760

model.validate(num_of_dense_layers=3, activation_function_types=[0, 1, 2], num_of_outputs=2,
                 layer_lengths=[35, 42], optimizer_object=rms_prop, cost_function_type=0,
               l2_weight_lambda=.0005, l2_bias_lambda=.0005)
'''

'''
rms_prop = OptimizerRMSProp(learning_rate=1, decay=0.01, epsilon=1e-7, rho=0.4)  # acc: 0.736, validation acc: 0.754

model.validate(num_of_dense_layers=3, activation_function_types=[0, 1, 2], num_of_outputs=2,
                 layer_lengths=[64, 64], optimizer_object=rms_prop, cost_function_type=0,
               l2_weight_lambda=.0005, l2_bias_lambda=.0005)
'''



rms_prop = OptimizerRMSProp(learning_rate=1, decay=0.005, epsilon=1e-7, rho=0.4)  #acc: 0.749, validation acc: 0.756,  testing acc: 0.733

model.validate(num_of_dense_layers=3, activation_function_types=[0, 0, 2], num_of_outputs=2,
                 layer_lengths=[64, 64], optimizer_object=rms_prop, cost_function_type=0,
               l2_weight_lambda=.0005, l2_bias_lambda=.0005)



