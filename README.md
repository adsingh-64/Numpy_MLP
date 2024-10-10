# Numpy_MLP
Feedforward network with sigmoid activation function implementing matrix-based backpropagation, choice of regularization 
(L_1 regularization, L_2 regularization, or momentum-based gradient descent), 
choice of cost function (quadratic cost or cross entropy cost), learning rate 
schedule, and early stopping.

Based off code and exercises given in Neural Networks and Deep Learning
by Michael Nielsen (http://neuralnetworksanddeeplearning.com).

## Sample usage:
```python
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network_matrix
net = network_matrix.Network([784, 30, 10], cost=network_matrix.CrossEntropyCost, reg=network_matrix.L2_regularization)
net.SGD(training_data, 30, 10, 0.5, lmbda=5.0, test_data=validation_data)

