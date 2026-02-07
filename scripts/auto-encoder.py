import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def display(data, m, n):
    img = np.zeros((28 * m, 28 * n))
    for i in range(m):
        for j in range(n):
            img[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = data[i * m + j].reshape(
                28, 28
            )
    plt.figure(figsize=(m * 1.5, n * 1.5))
    plt.imshow(img, cmap="gray")
    plt.show()


class multiLayerPerceptron(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        num_in = layer_sizes[0]
        for num_out in layers[1:]:
            self.layers.append(nn.Linear(num_in, num_out))
            self.layers.append(nn.Sigmoid())
            num_in = num_out

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


mnist_train = pd.read_csv("./data/mnist_train.csv")
mnist_test = pd.read_csv("./data/mnist_test.csv")

x_train = mnist_train.iloc[:, 1:].to_numpy().reshape(-1, 28 * 28) / 255
x_test = mnist_test.iloc[:, 1:].to_numpy().reshape(-1, 28 * 28) / 255

print(f"训练集大小：{len(x_train)}")
print(f"测试集大小：{len(x_test)}")

display(x_test, 3, 5)
