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
        for num_out in layer_sizes[1:]:
            self.layers.append(nn.Linear(num_in, num_out))
            self.layers.append(nn.Sigmoid())
            num_in = num_out

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


def main():
    learning_rate = 0.01
    max_epoch = 10
    batch_size = 256
    display_step = 2
    np.random.seed(42)
    torch.manual_seed(42)

    mnist_train = pd.read_csv("./data/mnist_train.csv")
    mnist_test = pd.read_csv("./data/mnist_test.csv")

    features_train = mnist_train.iloc[:, 1:].to_numpy().reshape(-1, 28 * 28) / 255
    features_test = mnist_test.iloc[:, 1:].to_numpy().reshape(-1, 28 * 28) / 255

    print(f"训练集大小：{len(features_train)}")
    print(f"测试集大小：{len(features_test)}")

    display(features_test, 5, 5)

    layer_sizes = [784, 256, 128, 100]
    encoder = multiLayerPerceptron(layer_sizes)
    decoder = multiLayerPerceptron(layer_sizes[::-1])

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate
    )

    for i in range(max_epoch):
        index = np.arange(len(features_train))
        index = np.random.permutation(index)
        features_train = features_train[index]
        st = 0
        ave_loss = []

        while st < len(features_train):
            ed = min(st + batch_size, len(features_train))
            features = torch.from_numpy(features_train[st:ed]).to(torch.float32)
            latent_representation = encoder(features)
            features_reconstruction = decoder(latent_representation)
            loss = 0.5 * nn.functional.mse_loss(features, features_reconstruction)
            ave_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            st = ed

        ave_loss = np.average(ave_loss)
        if i % display_step == 0 or i == max_epoch - 1:
            print(f"训练轮数：{i}，平均损失：{ave_loss:.4f}")
            # 选取测试集中的部分图像重建并展示
            with torch.inference_mode():
                features = torch.from_numpy(features_test[: 3 * 5]).to(torch.float32)
                features_rec = decoder(encoder(features))
                features_rec = features_rec.cpu().numpy()
            display(features_rec, 3, 5)


if __name__ == "__main__":
    main()
