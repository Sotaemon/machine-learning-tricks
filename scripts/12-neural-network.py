import numpy as np
import matplotlib.pyplot as plt


class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def update(self, learning_rate):
        pass


class Linear(Layer):
    def __init__(self, num_in, num_out, use_bias=True):
        self.num_in = num_in
        self.num_out = num_out
        self.use_bias = use_bias

        self.W = np.random.normal(loc=0, scale=1.0, size=(num_in, num_out))
        if use_bias:
            self.b = np.zeros((1, num_out))

    def forward(self, x):
        self.x = x
        self.y = x @ self.W
        if self.use_bias:
            self.y += self.b
        return self.y

    def backward(self, grad):
        self.grad_W = self.x.T @ grad / grad.shape[0]
        if self.use_bias:
            self.grad_b = np.mean(grad, axis=0, keepdims=True)
        grad = grad @ self.W.T
        return grad

    def update(self, learning_rate):
        self.W -= learning_rate * self.grad_W
        if self.use_bias:
            self.b -= learning_rate * self.grad_b


class Identity(Layer):

    def forward(self, x):
        return x

    def backward(self, grad):
        return grad


class Sigmoid(Layer):

    def forward(self, x):
        self.x = x
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, grad):
        return grad * self.y * (1 - self.y)


class Tanh(Layer):

    def forward(self, x):
        self.x = x
        self.y = np.tanh(x)
        return self.y

    def backward(self, grad):
        return grad * (1 - self.y**2)


class ReLU(Layer):

    def forward(self, x):
        self.x = x
        self.y = np.maximum(x, 0)
        return self.y

    def backward(self, grad):
        return grad * (self.x >= 0)

class MultiLayerPerceptron:
    pass

def main():
    data_set_path = "./data/xor_dataset.csv"
    ratio = 0.8
    np.random.seed(42)

    activation_dict = {
        "identity": Identity,
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "relu": ReLU,
    }

    data = np.loadtxt(data_set_path, delimiter=",")
    print("Data Set: ", len(data))
    print(data[:5])

    split = int(ratio * len(data))
    data = np.random.permutation(data)

    features_train = data[:split, :2]
    labels_train = data[:split, -1].reshape(-1, 1)

    features_test = data[split:, :2]
    labels_test = data[split:, -1].reshape(-1, 1)

    return 0


if __name__ == "__main__":
    main()
