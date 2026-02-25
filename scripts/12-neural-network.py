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
        self.num_in  = num_in
        self.num_out = num_out
        self.use_bias= use_bias

        self.W = np.random.normal(loc=0, scale=1.0, size=(num_in, num_out))
        if use_bias:
            self.b = np.zeros((1, num_out))
    
    def forward(self, x):
        self.x = x
        self.y = x @ self.W


def main():
    data_set_path = './data/xor_dataset.csv'
    ratio = 0.8
    np.random.seed(42)
    
    data = np.loadtxt(data_set_path, delimiter=',')
    print('Data Set: ', len(data))
    print(data[:5])

    split = int(ratio * len(data))
    data = np.random.permutation(data)

    features_train = data[:split, :2]
    labels_train   = data[:split, -1].reshape(-1, 1)

    features_test  = data[split:, :2]
    labels_test    = data[split:, -1].reshape(-1, 1)

    return 0

if __name__ == "__main__":
    main()