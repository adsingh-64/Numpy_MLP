# %load network.py

"""
network_matrix.py
"""
#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np


class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(Z, A, Y):
        """Return the error delta from the output layer."""
        return (A - Y) * sigmoid_prime(Z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(Z, A, Y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return A - Y


class L2_regularization(object):
    """implements update step for L2_regularized cost function"""

    def update_step(eta, lmbda, n, weights, biases, nabla_w, nabla_b, velocities, mu):
        weights = [
            (1 - eta * lmbda / n) * w - (eta) * nw for w, nw in zip(weights, nabla_w)
        ]
        biases = [b - (eta) * nb for b, nb in zip(biases, nabla_b)]
        return weights, biases, velocities


class momentum(object):
    def update_step(eta, lmbda, n, weights, biases, nabla_w, nabla_b, velocities, mu):
        velocities = [mu * v - eta * nw for v, nw in zip(velocities, nabla_w)]
        weights = [w + v for w, v in zip(weights, velocities)]
        biases = [b - (eta) * nb for b, nb in zip(biases, nabla_b)]
        return weights, biases, velocities


class L1_regularization(object):
    """implements update step for L1_regularized cost function"""

    def update_step(eta, lmbda, n, weights, biases, nabla_w, nabla_b):
        weights = [
            w - (eta * lmbda / n) * np.sign(w) - eta * nw
            for w, nw in zip(weights, nabla_w)
        ]
        biases = [b - (eta) * nb for b, nb in zip(biases, nabla_b)]
        return weights, biases


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_accuracy = 0
        self.counter = 0
        self.accuracies = []

    def should_stop(self, current_accuracy):
        self.accuracies.append(current_accuracy)
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True
        return False


class LearningRate:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_accuracy = 0
        self.counter = 0
        self.num_halves = 0
        self.accuracies = []

    def should_halve(self, current_accuracy):
        self.accuracies.append(current_accuracy)
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.counter = 0
            self.num_halves += 1
            return True
        return False

    def get_num_halves(self):
        return self.num_halves


class Network(object):

    def __init__(
        self,
        sizes,
        cost=CrossEntropyCost,
        reg=L2_regularization,
        patience=10,
    ):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron. Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
        self.reg = reg
        self.learning_rate = LearningRate(patience)

    def default_weight_initializer(self):
        """Initializes weights as N(0, 1/n_in)"""
        self.biases = [
            np.random.randn(y, 1) for y in self.sizes[1:]
        ]  # each element in list is a column vector of the biases, b^l
        self.weights = [
            np.random.randn(y, x)
            / np.sqrt(x)  # each element in list is weight matrix W^l
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]
        self.velocities = [np.zeros(w.shape) for w in self.weights]

    def SGD(
        self,
        training_data,
        epochs,
        mini_batch_size,
        eta,
        lmbda=0.0,
        mu=0.5,
        test_data=None,
    ):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n, mu)

            corrects = self.evaluate(test_data)

            if test_data:
                print("Epoch {} : {} / {}".format(j, corrects, n_test))
            else:
                print("Epoch {} complete".format(j))

            if self.learning_rate.should_halve(corrects):
                eta /= 2
                print(f"learning halved to: {eta}")

            num_halves = self.learning_rate.get_num_halves()
            print(f"num halves: {num_halves}")
            if num_halves >= 7:  # stop training when learning_rate is 1/128 of original
                break

    def update_mini_batch(self, mini_batch, eta, lmbda, n, mu):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        velocities = [np.zeros(w.shape) for w in self.weights]
        X = np.column_stack([x for x, y in mini_batch])
        Y = np.column_stack([y for x, y in mini_batch])

        nabla_b, nabla_w = self.backprop(X, Y)

        self.weights, self.biases, self.velocities = self.reg.update_step(
            eta,
            lmbda,
            n,
            self.weights,
            self.biases,
            nabla_w,
            nabla_b,
            self.velocities,
            mu,
        )

    def backprop(self, X, Y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = X
        activations = [X]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = (
                np.dot(w, activation) + b
            )  # this is the same as adding hte matrix B, it is called numpy broadcasting
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)  # 2), pg.38, Nueral Nets
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], Y)
        # step 1 of algorithm, y is (0,..1,...0)T
        nabla_b[-1] = np.mean(delta, axis=1, keepdims=True)  
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) / X.shape[1]
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = (
                np.dot(self.weights[-l + 1].transpose(), delta) * sp
            )  # for l = L-1, ..., 2, compute delta^l
            nabla_b[-l] = np.mean(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose()) / X.shape[1]
        return (nabla_b, nabla_w)
    
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [
            (
                np.argmax(self.feedforward(x)),
                y,
            )  # tuples are (larget output activation index , label)
            for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


