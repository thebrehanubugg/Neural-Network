"""Neural Network Class."""
from activation_functions import sigmoid, dsigmoid
from random import choice
from Matrix import Matrix


class NeuralNetwork():
    """Neural Network Class."""

    def __init__(self, input_nodes, hidden_nodes, output_nodes, activations):
        """Neural Network Initialization Method."""
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
        self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)

        self.bias_h = Matrix(self.hidden_nodes, 1)
        self.bias_o = Matrix(self.output_nodes, 1)

        self.weights_ih.randomize()
        self.weights_ho.randomize()

        self.bias_h.randomize()
        self.bias_o.randomize()

        self.activation_function = activations[0]
        self.d_function = activations[1]

        self.learning_rate = 0.3

    def print(self):
        """Print out the network's information."""
        print("=== START NETWORK INFORMATION ===")
        print("# OF INPUT NODES:", self.input_nodes)
        print("# OF HIDDEN NODES:", self.hidden_nodes)
        print("# OF OUTPUT NODES:", self.output_nodes)
        print("INPUT TO HIDDEN MATRIX:")
        self.weights_ih.print()
        print("HIDDEN TO OUTPUT MATRIX:")
        self.weights_ho.print()
        print("=== END NETWORK INFORMATION ===")

    def train(self, inputs_array, targets_array):
        """Feedforward and train the network with inputs and targets."""
        inputs = Matrix.to_matrix(inputs_array)
        targets = Matrix.to_matrix(targets_array)

        hidden_inputs = self.weights_ih.dot(inputs)
        hidden_outputs = hidden_inputs.activate(self.activation_function)

        final_inputs = self.weights_ho.dot(hidden_outputs)
        final_outputs = final_inputs.activate(self.activation_function)

        hidden_ho_t = self.weights_ho.transpose()
        hidden_o_t = hidden_outputs.transpose()
        inputs_t = inputs.transpose()

        output_errors = targets.subtract(final_outputs)
        hidden_errors = hidden_ho_t.dot(output_errors)

        hidden_ho_distributed = output_errors.multiply(final_outputs)
        hidden_ih_distributed = hidden_errors.multiply(hidden_outputs)

        output_gradients = final_outputs.sub(1.0)
        output_gradients = output_gradients.multiply(hidden_ho_distributed)
        output_gradients = output_gradients.dot(hidden_o_t)

        hidden_gradients = hidden_outputs.sub(1.0)
        hidden_gradients = hidden_gradients.multiply(hidden_ih_distributed)
        hidden_gradients = hidden_gradients.dot(inputs_t)

        delta_weights_ih = hidden_gradients.scale(self.learning_rate)
        delta_weights_ho = output_gradients.scale(self.learning_rate)

        self.weights_ih.add(delta_weights_ih)
        self.weights_ho.add(delta_weights_ho)

    def predict(self, inputs_arr):
        """Neural Network Predict Method."""
        inputs = Matrix.to_matrix(inputs_arr)

        hidden_inputs = self.weights_ih.dot(inputs)
        hidden_outputs = hidden_inputs.activate(self.activation_function)

        final_inputs = self.weights_ho.dot(hidden_outputs)
        final_outputs = final_inputs.activate(self.activation_function)

        return final_outputs
